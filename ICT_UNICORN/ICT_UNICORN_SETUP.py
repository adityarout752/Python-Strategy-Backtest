"""
ICT Unicorn Strategy (Fractal swings + True FVG + Breaker Block + HTF filter)
- Limit order entries at breaker candle open when Unicorn condition is satisfied.
- Backtest engine simulates entry execution when LTF bars reach entry price, and SL/TP triggers.
- Input: df_ltf (pandas DataFrame) with DateTime index and columns: Open, High, Low, Close
- Optional: df_htf can be provided (same columns). If not provided, df_ltf is resampled to HTF.
"""

import pandas as pd
import numpy as np


# --------------------- USER PARAMETERS ---------------------
HTF_RESAMPLE = "1H"           # e.g., "1H", "4H"
LTF = "5T"                    # for reference only
RISK_PER_TRADE = 1         # fraction of balance
RISK_REWARD = 2.1             # TP = SL * RR
STOP_BUFFER = 0.0005          # price units to buffer stop outside breaker (adjust for instrument)
MIN_FVG_CANDLES = 3           # FVG uses 3 candles by definition
START_BALANCE = 10000.0
# -----------------------------------------------------------


# --------------------- UTIL: resample to HTF if needed ---------------------
def make_htf_from_ltf(df_ltf, htf_resample=HTF_RESAMPLE):
    df_htf = df_ltf.resample(htf_resample).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()
    return df_htf


# --------------------- FRACTAL SWING DETECTION ---------------------
def detect_fractals(df):
    """Add swing_high and swing_low columns (values are the swing price at that index)"""
    df = df.copy()
    df["swing_high"] = np.nan
    df["swing_low"] = np.nan

    # simple 1-bar fractal (needs previous and next candle)
    for i in range(1, len(df) - 1):
        if df["High"].iat[i] > df["High"].iat[i - 1] and df["High"].iat[i] > df["High"].iat[i + 1]:
            df.iat[i, df.columns.get_loc("swing_high")] = df["High"].iat[i]
        if df["Low"].iat[i] < df["Low"].iat[i - 1] and df["Low"].iat[i] < df["Low"].iat[i + 1]:
            df.iat[i, df.columns.get_loc("swing_low")] = df["Low"].iat[i]

    return df


# --------------------- TRUE FVG DETECTION ---------------------
def detect_true_fvgs(df):
    """
    Detect 3-candle FVGs and store as ranges.
    We'll store fvgs as dict {index_of_middle_candle: (fvg_low, fvg_high, direction, c1_idx, c2_idx, c3_idx)}
    direction: 'bullish' (gap below) or 'bearish' (gap above)
    True FVG = no overlap: bullish if High[c1] < Low[c3]
    """
    fvgs = {}
    for i in range(2, len(df)):  # i is index of 3rd candle (c3)
        c1 = i - 2
        c2 = i - 1
        c3 = i

        # bullish FVG check: candle1 High < candle3 Low
        if df["High"].iat[c1] < df["Low"].iat[c3]:
            fvg_low = df["High"].iat[c1]
            fvg_high = df["Low"].iat[c3]
            # only keep FVG if it remains unfilled at time of detection (current price not within)
            # We'll append; user can later check if filled before entry.
            fvgs[c2] = {
                "low": fvg_low,
                "high": fvg_high,
                "dir": "bullish",
                "c1": df.index[c1],
                "c2": df.index[c2],
                "c3": df.index[c3]
            }

        # bearish FVG check: candle1 Low > candle3 High
        if df["Low"].iat[c1] > df["High"].iat[c3]:
            fvg_low = df["High"].iat[c3]
            fvg_high = df["Low"].iat[c1]
            fvgs[c2] = {
                "low": fvg_low,
                "high": fvg_high,
                "dir": "bearish",
                "c1": df.index[c1],
                "c2": df.index[c2],
                "c3": df.index[c3]
            }

    return fvgs


# --------------------- BREAKER BLOCK DETECTION (using fractals & BOS) ---------------------
def detect_breaker_blocks(df):
    """
    We identify breaker blocks as:
      - A fractal swing (high or low) exists (prev_swing_high/low)
      - Then price makes a liquidity grab beyond that swing (wick beyond)
      - Then price confirms BOS by closing beyond the opposite side of the last opposing candle (see logic)
      - The last opposing candle before BOS is labeled as breaker block (store its index and H/L)
    Returns a dict: {index_of_breaker_candle: {"type":"bullish"/"bearish", "low":..., "high":..., "trigger_idx": i}}
    """
    breaker_blocks = {}
    prev_swing_high_idx = None
    prev_swing_low_idx = None

    # map fractal indices
    swing_high_idxs = df.index[df["swing_high"].notna()].tolist()
    swing_low_idxs = df.index[df["swing_low"].notna()].tolist()

    # For faster index-based operations use integer positions
    for i in range(2, len(df)):
        # update prev swing positions (the most recent confirmed swing)
        if df["swing_high"].iat[i] == df["swing_high"].iat[i]:  # not NaN
            prev_swing_high_idx = i
        if df["swing_low"].iat[i] == df["swing_low"].iat[i]:
            prev_swing_low_idx = i

        # Bearish breaker detection: liquidity above previous swing high, then BOS down
        if prev_swing_high_idx is not None:
            prev_swing_high_price = df["swing_high"].iat[prev_swing_high_idx]
            # liquidity grab: current high exceeds previous swing high
            if df["High"].iat[i] > prev_swing_high_price:
                # Check last opposing candle (the candle before the current one) is bullish (the one that will become breaker)
                last_bullish_idx = i - 1
                if df["Close"].iat[last_bullish_idx] > df["Open"].iat[last_bullish_idx]:
                    # BOS down: current close is below the low of the last bullish candle (breaks structure down)
                    if df["Close"].iat[i] < df["Low"].iat[last_bullish_idx]:
                        # record breaker (bearish)
                        breaker_blocks[last_bullish_idx] = {
                            "type": "bearish",
                            "low": float(df["Low"].iat[last_bullish_idx]),
                            "high": float(df["High"].iat[last_bullish_idx]),
                            "trigger_idx": i
                        }
                        prev_swing_high_idx = None  # reset to avoid repeated detections

        # Bullish breaker detection: liquidity below previous swing low, then BOS up
        if prev_swing_low_idx is not None:
            prev_swing_low_price = df["swing_low"].iat[prev_swing_low_idx]
            if df["Low"].iat[i] < prev_swing_low_price:
                last_bearish_idx = i - 1
                if df["Close"].iat[last_bearish_idx] < df["Open"].iat[last_bearish_idx]:
                    # BOS up: current close above the high of the last bearish candle
                    if df["Close"].iat[i] > df["High"].iat[last_bearish_idx]:
                        breaker_blocks[last_bearish_idx] = {
                            "type": "bullish",
                            "low": float(df["Low"].iat[last_bearish_idx]),
                            "high": float(df["High"].iat[last_bearish_idx]),
                            "trigger_idx": i
                        }
                        prev_swing_low_idx = None

    return breaker_blocks


# --------------------- HELPER: Check FVG overlap with breaker ---------------------
def fvg_overlaps_breaker(fvg, breaker):
    """
    fvg: dict with 'low','high','dir'
    breaker: dict with 'low','high','type'
    Overlap exists if ranges intersect.
    """
    # intersection test
    return not (fvg["high"] < breaker["low"] or fvg["low"] > breaker["high"])


# --------------------- MERGE HTF BIAS ---------------------
def attach_htf_bias_to_ltf(df_ltf, df_htf):
    """
    Create HTF_trend column on LTF by forward-filling htf trend for each LTF timestamp.
    HTF trend simple rule: Close > Open => bullish, Close < Open => bearish.
    """
    df_htf = df_htf.copy()
    df_htf["HTF_trend"] = np.where(df_htf["Close"] > df_htf["Open"], "bullish", "bearish")
    # Reindex HTF trend to LTF by asof/merge: use pd.merge_asof
    df_htf_reset = df_htf[["HTF_trend"]].reset_index().rename(columns={"index": "htf_time"})
    df_ltf_reset = df_ltf.reset_index().rename(columns={"index": "ltf_time"})
    merged = pd.merge_asof(df_ltf_reset.sort_values("ltf_time"), df_htf_reset.sort_values("htf_time"),
                           left_on="ltf_time", right_on="htf_time", direction="backward")
    merged = merged.set_index("ltf_time")
    # bring back remaining columns and return
    df_out = merged[df_ltf.columns.tolist() + ["HTF_trend"]]
    return df_out


# --------------------- ENTRY DETECTION (Unicorn) ---------------------
def find_unicorn_entries(df_ltf, fvgs, breakers, df_htf):
    """
    For each breaker block, check if there exists a TRUE FVG (from fvgs dict) of the same direction
    and that the FVG overlaps the breaker zone (Unicorn).
    Add an entry record: entry_price (breaker open), sl, tp, qty calculation deferred to backtest.
    """
    entries = []  # list of dicts with info and index where breaker candle is
    # build a mapping from integer index to timestamp to ease matching
    for br_idx, br in breakers.items():
        br_time = df_ltf.index[br_idx]
        br_type = br["type"]  # 'bullish' or 'bearish'
        # search for FVGs that are earlier than the breaker trigger (must exist before trigger)
        candidate_fvgs = []
        for fvg_idx, fvg in fvgs.items():
            # fvg c3 index is fvg['c3'] time; ensure fvg was created before the breaker trigger bar
            # convert fvg c3 timestamp to position: find positional index in df_ltf
            # easier: compare timestamps
            if fvg["c3"] < df_ltf.index[br["trigger_idx"]]:
                # direction must match: unicorn uses same direction FVG (for bullish unicorn -> bullish FVG)
                if fvg["dir"] == br_type:
                    # check FVG still true (not filled) at time of breaker trigger: price at br trigger must not be inside fvg
                    br_trigger_high = df_ltf["High"].iat[br["trigger_idx"]]
                    br_trigger_low = df_ltf["Low"].iat[br["trigger_idx"]]
                    # FVG filled if trigger high/low intersects FVG - skip if filled
                    if not (br_trigger_low <= fvg["high"] and br_trigger_high >= fvg["low"]):
                        # check overlap between fvg range and breaker zone
                        if fvg_overlaps_breaker(fvg, br):
                            candidate_fvgs.append((fvg_idx, fvg))
        # if any candidates, choose the most recent FVG (closest to breaker)
        if candidate_fvgs:
            candidate_fvgs.sort(key=lambda x: x[0], reverse=True)
            chosen_fvg_idx, chosen_fvg = candidate_fvgs[0]
            # entry price is breaker candle open
            entry_price = float(df_ltf["Open"].iat[br_idx])
            # stop-loss and tp:
            if br_type == "bullish":
                sl = br["low"] - STOP_BUFFER
                tp = entry_price + (entry_price - sl) * RISK_REWARD
            else:
                sl = br["high"] + STOP_BUFFER
                tp = entry_price - (sl - entry_price) * RISK_REWARD

            entries.append({
                "breaker_idx": br_idx,
                "breaker_time": br_time,
                "type": br_type,
                "entry_price": entry_price,
                "sl": sl,
                "tp": tp,
                "fvg_idx": chosen_fvg_idx,
                "fvg": chosen_fvg,
                "trigger_idx": br["trigger_idx"]
            })

    return entries


# --------------------- SIMPLE BACKTEST ENGINE ---------------------
def backtest_entries(df, entries, start_balance=START_BALANCE):
    """
    Simulate execution: for each entry (ordered by time), we wait for fill at entry_price (limit),
    then manage trade using bar high/low checks for SL/TP. Position sizing uses RISK_PER_TRADE.
    This is a simple deterministic simulation: no partial fills, no slippage, no spread. Use with care.
    """
    balance = start_balance
    trades = []
    # Sort entries by breaker time
    entries_sorted = sorted(entries, key=lambda x: x["breaker_time"])

    for e in entries_sorted:
        br_idx = e["breaker_idx"]
        # we simulate price forward from bar after breaker candle (br_idx+1) to see if price hits entry
        filled = False
        fill_bar_idx = None
        for j in range(br_idx + 1, len(df)):
            low = df["Low"].iat[j]
            high = df["High"].iat[j]
            # check if limit entry is filled within this bar
            if e["type"] == "bullish":
                # need price to reach entry_price (buy limit)
                if low <= e["entry_price"] <= high:
                    filled = True
                    fill_bar_idx = j
                    break
                # if before fill, SL was invalidated because price breaches below SL before entry, then cancel
                if low <= e["sl"]:
                    filled = False
                    fill_bar_idx = None
                    break
            else:
                # sell limit
                if low <= e["entry_price"] <= high:
                    filled = True
                    fill_bar_idx = j
                    break
                if high >= e["sl"]:
                    filled = False
                    fill_bar_idx = None
                    break

        if not filled:
            # skip entry if not filled
            continue

        # Once filled at entry_price, simulate from fill_bar_idx onwards for SL/TP
        qty = (balance * RISK_PER_TRADE) / abs(e["entry_price"] - e["sl"])
        trade_result = None
        for k in range(fill_bar_idx, len(df)):
            low = df["Low"].iat[k]
            high = df["High"].iat[k]
            if e["type"] == "bullish":
                # check SL hit
                if low <= e["sl"]:
                    pl = (e["entry_price"] - e["sl"]) * qty * -1.0
                    balance += pl
                    trade_result = {"result": "LOSS", "pl": pl, "exit_idx": k}
                    break
                # check TP hit
                if high >= e["tp"]:
                    pl = (e["tp"] - e["entry_price"]) * qty
                    balance += pl
                    trade_result = {"result": "WIN", "pl": pl, "exit_idx": k}
                    break
            else:
                # bearish trade
                if high >= e["sl"]:
                    pl = (e["sl"] - e["entry_price"]) * qty * -1.0
                    balance += pl
                    trade_result = {"result": "LOSS", "pl": pl, "exit_idx": k}
                    break
                if low <= e["tp"]:
                    pl = (e["entry_price"] - e["tp"]) * qty
                    balance += pl
                    trade_result = {"result": "WIN", "pl": pl, "exit_idx": k}
                    break

        # if neither hit till end, we close at last close
        if trade_result is None:
            close_price = df["Close"].iat[-1]
            if e["type"] == "bullish":
                pl = (close_price - e["entry_price"]) * qty
            else:
                pl = (e["entry_price"] - close_price) * qty
            balance += pl
            trade_result = {"result": "CLOSED", "pl": pl, "exit_idx": len(df)-1}

        trades.append({
            "breaker_time": e["breaker_time"],
            "type": e["type"],
            "entry": e["entry_price"],
            "sl": e["sl"],
            "tp": e["tp"],
            "pl": trade_result["pl"],
            "result": trade_result["result"],
            "balance": balance
        })

    # summary
    wins = sum(1 for t in trades if t["pl"] > 0)
    losses = sum(1 for t in trades if t["pl"] <= 0)
    winrate = wins / max((wins + losses), 1) * 100

    return {
        "start_balance": start_balance,
        "final_balance": balance,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "winrate": round(winrate, 2),
        "trades": trades
    }


# --------------------- WRAPPER: run full flow ---------------------
def run_unicorn_strategy(df_ltf, df_htf=None, debug=False):
    # ensure datetime index sorted
    df_ltf = df_ltf.sort_index()
    if df_htf is None:
        df_htf = make_htf_from_ltf(df_ltf)
    df_htf = df_htf.sort_index()

    # attach HTF trend into LTF for reference (not strictly required for detection but for filtering)
    df_ltf_with_htf = attach_htf_bias_to_ltf(df_ltf, df_htf)

    # fractals on LTF
    df_fract = detect_fractals(df_ltf_with_htf)

    # FVG detection on LTF
    fvgs = detect_true_fvgs(df_fract)

    # breaker detection
    breakers = detect_breaker_blocks(df_fract)

    # find unicorn entries (breaker + true fvg + overlap + HTF alignment)
    entries = find_unicorn_entries(df_fract, fvgs, breakers, df_htf)

    # filter entries by HTF trend (ensure direction matches HTF)
    filtered_entries = []
    for e in entries:
        # get HTF trend at breaker time by looking up in df_ltf_with_htf
        htf_trend = df_ltf_with_htf["HTF_trend"].iat[e["breaker_idx"]]
        if htf_trend == e["type"]:
            filtered_entries.append(e)

    results = backtest_entries(df_fract, filtered_entries)
    if debug:
        return {
            "df": df_fract,
            "fvgs": fvgs,
            "breakers": breakers,
            "entries": filtered_entries,
            "results": results
        }
    else:
        return results


# --------------------- HOW TO USE (example) ---------------------
# 1) Load your LTF CSV:
#    df = pd.read_csv("EURUSD_5min.csv", parse_dates=["Date"], index_col="Date")
#    Ensure columns: Open, High, Low, Close  (float)
# 2) Call:
#    out = run_unicorn_strategy(df)
#    print(out)
#
# If you want debug info with objects:
#    debug_out = run_unicorn_strategy(df, debug=True)
#
# IMPORTANT: Tune STOP_BUFFER for your instrument (e.g., 0.0005 for FX pairs with 5-digit pricing might be OK,
#            but for indices/crypto you'll need to use absolute price units appropriate to the market).
