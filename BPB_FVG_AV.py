# eurusd_fvg_backtest.py
# Strategy rules implemented EXACTLY as provided:
# 1) Trend on 15m (EMA50 + structure via swing break in step 4)
# 2) Trade only after 12:00 UTC
# 3) Mark swing high/low on 15m
# 4) Require 3-candle breakout/breakdown of same color; each candle body >= 1/3 of avg body of last 10 (15m)
# 5) Mark FVG between first & third candle (bullish: gap between High(1) and Low(3); bearish: Low(1) and High(3))
# 6) Limit order at 50% of FVG; SL below/above the breakout candle (1st of the 3); entries/SL/TP based on 5m
# 7) Target = 1 : 2.1
#
# Output columns:
# PAIR TRADED, DATE OPEN, DATE CLOSED, DAY OF WEEK, TIME, PIP RISKED, DIRECTION(LONG OR SHORT), RESULT, PIP RISK

import requests
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
import os
import time as systime
from chart import plot_trade_chart, save_trade_to_excel

# ------------------ CONFIG ------------------
API_KEY = "e04b761b8f664d108bdc60371e517703"    # Your Alpha Vantage key
PAIR = "EUR/USD"
FROM_SYMBOL = "EUR"
TO_SYMBOL = "USD"
START_DATE = "2025-01-01 00:00:00"
END_DATE   = "2025-02-27 00:00:00"
OUTPUT_FILE = "backtest_results.xlsx"

EMA_PERIOD = 50
SWING_LOOKBACK = 3          # pivot size for swing high/low on 15m
BODY_AVG_WINDOW = 10        # average body length window (15m)
RR = 2.1                    # risk:reward
# --------------------------------------------

# --------------- DATA FETCH -----------------
import requests
import pandas as pd

def fetch_intraday_fx(from_symbol: str, to_symbol: str, interval: str, api_key: str) -> pd.DataFrame:
    """
    Fetch intraday forex data from Twelve Data API.
    interval: "5min" or "15min"
    """
    print(f"Fetching {interval} data from Twelve Data...")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": f"{from_symbol}/{to_symbol}",
        "interval": interval,
        "start_date": "2025-01-01",
        "end_date": "2025-03-31",
        "apikey": api_key,
        "format": "JSON",
        "outputsize": 5000  # max allowed per request
    }

    response = requests.get(url, params=params)
    data = response.json()
    
    if "values" not in data:
        print(f"❌ No data in response: {data}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data["values"])
    df.rename(columns={
        "datetime": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    
    }, inplace=True, errors="ignore")
    
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    
   
    columns_to_convert = ["Open", "High", "Low", "Close"]
    
    df[columns_to_convert] = df[columns_to_convert].astype(float)
    
    return df



def clip_date_range(df: pd.DataFrame, start_ts: str, end_ts: str) -> pd.DataFrame:
    st = pd.to_datetime(start_ts, utc=False)
    en = pd.to_datetime(end_ts, utc=False)
    return df[(df["Datetime"] >= st) & (df["Datetime"] <= en)].copy()

# ------------- 15m UTILITIES ----------------
def add_ema(df: pd.DataFrame, period: int = EMA_PERIOD) -> pd.DataFrame:
    df = df.copy()
    df["EMA50"] = df["Close"].ewm(span=period, adjust=False).mean()
    return df

def add_swings(df: pd.DataFrame, lb: int = SWING_LOOKBACK) -> pd.DataFrame:
    """
    Marks swing highs/lows with values; NaN if not a swing.
    A swing high at i means High[i] > High[i-k] and High[i] > High[i+k] for k in [1..lb]
    Likewise for swing low.
    """
    df = df.copy()
    highs = df["High"].values
    lows  = df["Low"].values
    swing_high = [np.nan]*len(df)
    swing_low  = [np.nan]*len(df)

    for i in range(lb, len(df)-lb):
        left_high  = highs[i-lb:i]
        right_high = highs[i+1:i+1+lb]
        left_low   = lows[i-lb:i]
        right_low  = lows[i+1:i+1+lb]
        if highs[i] > max(left_high) and highs[i] > max(right_high):
            swing_high[i] = highs[i]
        if lows[i] < min(left_low) and lows[i] < min(right_low):
            swing_low[i] = lows[i]

    df["Swing_High"] = swing_high
    df["Swing_Low"]  = swing_low
    return df

def body_size(s: pd.Series) -> float:
    return abs(s["Close"] - s["Open"])

# ------------- BREAKOUT LOGIC (15m) --------
def three_candle_breakout(df15: pd.DataFrame, idx: int, trend_dir: str) -> dict | None:
    """
    At 15m index idx (the 3rd candle), check for 3 same-color candles ending at idx,
    each body >= (1/3)*avg(last 10 bodies), AND the 3rd closes beyond last swing in trend direction.
    Returns dict with c1,c2,c3 rows if breakout found; else None.
    """
    if idx < BODY_AVG_WINDOW + 2:
        return None

    # time filter: only consider if the 3rd candle time is at/after 12:00 UTC
    ts = df15.iloc[idx]["Datetime"]
    if ts.time() < dtime(12, 0):
        return None

    # candles
    c1 = df15.iloc[idx-2]
    c2 = df15.iloc[idx-1]
    c3 = df15.iloc[idx]

    # same color & body-size rule
    bavg = df15.iloc[idx-BODY_AVG_WINDOW:idx]["Close"].sub(df15.iloc[idx-BODY_AVG_WINDOW:idx]["Open"]).abs().mean()
    if bavg == 0 or np.isnan(bavg):
        return None

    if trend_dir == "LONG":
        same_color = (c1["Close"] > c1["Open"]) and (c2["Close"] > c2["Open"]) and (c3["Close"] > c3["Open"])
    else:
        same_color = (c1["Close"] < c1["Open"]) and (c2["Close"] < c2["Open"]) and (c3["Close"] < c3["Open"])

    if not same_color:
        return None

    min_body = bavg / 3.0
    if not (body_size(c1) >= min_body and body_size(c2) >= min_body and body_size(c3) >= min_body):
        return None

    # breakout beyond most recent swing level (before c1)
    # find last swing high/low prior to c1's index
    prior = df15.iloc[:idx-1]  # up to c2
    last_swing_high = prior["Swing_High"].dropna().iloc[-1] if prior["Swing_High"].dropna().size else np.nan
    last_swing_low  = prior["Swing_Low"].dropna().iloc[-1]  if prior["Swing_Low"].dropna().size  else np.nan

    if trend_dir == "LONG":
        if np.isnan(last_swing_high):
            return None
        # 3rd candle close must break above last swing high
        if c3["Close"] <= last_swing_high:
            return None
    else:
        if np.isnan(last_swing_low):
            return None
        # 3rd candle close must break below last swing low
        if c3["Close"] >= last_swing_low:
            return None

    return {"c1": c1, "c2": c2, "c3": c3, "last_swing_high": last_swing_high, "last_swing_low": last_swing_low}

# ------------- FVG (from 3-candle move) ----
def compute_fvg(bounds: dict, direction: str) -> tuple | None:
    """
    FVG defined per your rule:
      Bullish breakout: gap between High(1st) and Low(3rd); require High(1) < Low(3).
      Bearish breakdown: gap between Low(1st) and High(3rd); require Low(1) > High(3).
    Returns (fvg_low, fvg_high) as numeric bounds if valid; else None.
    """
    c1 = bounds["c1"]
    c3 = bounds["c3"]
    if direction == "LONG":
        high1 = c1["High"]
        low3  = c3["Low"]
        if high1 < low3:
            return (high1, low3)
    else:
        low1  = c1["Low"]
        high3 = c3["High"]
        if low1 > high3:
            return (high3, low1)
    return None

# ------------- TREND (15m EMA50) -----------
def current_trend(df15: pd.DataFrame, idx: int) -> str | None:
    """
    Trend by EMA50 at the 3rd candle's close.
    """
    row = df15.iloc[idx]
    if np.isnan(row["EMA50"]):
        return None
    return "LONG" if row["Close"] > row["EMA50"] else "SHORT"

# ------------- ENTRY/EXIT ON 5m ------------
def simulate_trade_on_5m(df5: pd.DataFrame,
                         entry_level: float,
                         sl_level: float,
                         tp_level: float,
                         start_time: pd.Timestamp,
                         direction: str) -> tuple | None:
    """
    After the 3rd 15m candle time, look forward on 5m:
     1) Wait for limit entry touch (Low<=entry<=High).
     2) Once in, check subsequent bars for first touch of SL or TP.
    Returns (entry_time, exit_time, result, pip_risked, pip_outcome).
    """
    # consider only bars at/after the 3rd candle close time
    fwd = df5[df5["Datetime"] >= start_time].copy()
    if fwd.empty:
        return None

    # Wait for fill
    in_trade = False
    entry_time = None
    for i in range(len(fwd)):
        row = fwd.iloc[i]
        if not in_trade:
            if row["Low"] <= entry_level <= row["High"]:
                in_trade = True
                entry_time = row["Datetime"]
                # After fill, start monitoring from SAME bar: SL/TP could be hit within this bar too
                # So evaluate SL/TP priority: conservative—assume worst case: SL first if both within same bar?
                # We'll check both; for long, if Low<=SL then SL before TP; for short, if High>=SL then SL before TP.
                # Then continue loop to check resolution.
                # Immediate resolution:
                if direction == "LONG":
                    if row["Low"] <= sl_level:
                        pip_risk = abs(entry_level - sl_level) * 10000.0
                        return (entry_time, row["Datetime"], "LOSS", pip_risk, -pip_risk)
                    if row["High"] >= tp_level:
                        pip_risk = abs(entry_level - sl_level) * 10000.0
                        pip_out = abs(tp_level - entry_level) * 10000.0
                        return (entry_time, row["Datetime"], "WIN", pip_risk, pip_out)
                else:
                    if row["High"] >= sl_level:
                        pip_risk = abs(sl_level - entry_level) * 10000.0
                        return (entry_time, row["Datetime"], "LOSS", pip_risk, -pip_risk)
                    if row["Low"] <= tp_level:
                        pip_risk = abs(sl_level - entry_level) * 10000.0
                        pip_out = abs(entry_level - tp_level) * 10000.0
                        return (entry_time, row["Datetime"], "WIN", pip_risk, pip_out)
                # If not resolved in same bar, continue to next bars
        else:
            # trade is open; check resolution
            row = fwd.iloc[i]
            if direction == "LONG":
                if row["Low"] <= sl_level:
                    pip_risk = abs(entry_level - sl_level) * 10000.0
                    return (entry_time, row["Datetime"], "LOSS", pip_risk, -pip_risk)
                if row["High"] >= tp_level:
                    pip_risk = abs(entry_level - sl_level) * 10000.0
                    pip_out = abs(tp_level - entry_level) * 10000.0
                    return (entry_time, row["Datetime"], "WIN", pip_risk, pip_out)
            else:
                if row["High"] >= sl_level:
                    pip_risk = abs(sl_level - entry_level) * 10000.0
                    return (entry_time, row["Datetime"], "LOSS", pip_risk, -pip_risk)
                if row["Low"] <= tp_level:
                    pip_risk = abs(sl_level - entry_level) * 10000.0
                    pip_out = abs(entry_level - tp_level) * 10000.0
                    return (entry_time, row["Datetime"], "WIN", pip_risk, pip_out)

    # never filled or never resolved
    return None


# ------------- MAIN BACKTEST ---------------
def backtest():
    # Fetch data
    print("Fetching 15m data...")
    df15 = fetch_intraday_fx(FROM_SYMBOL, TO_SYMBOL, "15min", API_KEY)
    systime.sleep(15)  # be kind to API limits
    df5  = fetch_intraday_fx(FROM_SYMBOL, TO_SYMBOL, "5min", API_KEY)
    print("Fetching 5m data...")
    if df15.empty or df5.empty:
        print("❌ Could not fetch sufficient data. Exiting.")
        return

    # Date clip
    df15 = clip_date_range(df15, START_DATE, END_DATE)
    df5  = clip_date_range(df5,  START_DATE, END_DATE)

    if df15.empty or df5.empty:
        print("❌ No data in the requested date range.")
        return

    # Prep 15m indicators
    df15 = add_ema(df15, EMA_PERIOD)
    df15 = add_swings(df15, SWING_LOOKBACK)

    # Build results
    results = []

    # Iterate across 15m bars (start where EMA and body avg are defined)
    for i in range(max(EMA_PERIOD, BODY_AVG_WINDOW) + 2, len(df15)):
        # Trend by EMA50 at current bar
        trend = current_trend(df15, i)
        if trend is None:
            continue

        # Look for valid 3-candle breakout aligned with trend + after 12:00 UTC
        bo = three_candle_breakout(df15, i, trend)
        if not bo:
            continue

        # Compute FVG per rule
        fvg = compute_fvg(bo, trend)
        if not fvg:
            continue
        fvg_low, fvg_high = fvg
        entry = (fvg_low + fvg_high) / 2.0  # 50% of FVG

        # SL: below/above the breakout candle (1st of the 3)
        c1 = bo["c1"]
        if trend == "LONG":
            sl = c1["Low"]
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + RR * risk
        else:
            sl = c1["High"]
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - RR * risk

        # Simulate on 5m starting from the close time of the 3rd candle
        c3_time = df15.iloc[i]["Datetime"]
        sim = simulate_trade_on_5m(df5, entry, sl, tp, c3_time, trend)
        if not sim:
            continue  # either not filled or never resolved → ignore per your rule

        entry_time, exit_time, result, pip_risk, pip_outcome = sim
        # Plot and save chart enter the plotted c hart here

        # Log only executed/resolved trades
        results.append({
            "PAIR TRADED": f"{FROM_SYMBOL}{TO_SYMBOL}=X".replace("=X",""),  # label
            "DATE OPEN": entry_time.strftime("%Y-%m-%d %H:%M"),
            "DATE CLOSED": exit_time.strftime("%Y-%m-%d %H:%M"),
            "DAY OF WEEK": entry_time.strftime("%A"),
            "TIME": entry_time.strftime("%H:%M"),
            "PIP RISKED": round(pip_risk, 1),
            "DIRECTION(LONG OR SHORT)": "LONG" if trend == "LONG" else "SHORT",
            "RESULT": result,
            "PIP RISK": round(pip_outcome, 1)
        })

    # Save to Excel (append if exists)
    if results:
        df_out = pd.DataFrame(results)
        if os.path.exists(OUTPUT_FILE):
            try:
                existing = pd.read_excel(OUTPUT_FILE)
                df_out = pd.concat([existing, df_out], ignore_index=True)
            except Exception:
                pass
        df_out.to_excel(OUTPUT_FILE, index=False)
        print(f"✅ Saved {len(results)} trades to { OUTPUT_FILE }")
    else:
        print("⚠️ No executed trades found for the specified period with given rules.")

if __name__ == "__main__":
    try:
        backtest()
    except Exception as e:
        print(f"An error occurred: {e}")
