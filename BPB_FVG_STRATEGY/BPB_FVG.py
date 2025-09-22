import os
import numpy as np
import pandas as pd
from datetime import time as dtime


# ------------------ CONFIG ------------------
PAIR = "EUR/USD"
START_DATE = "2020-01-01 00:00:00"
END_DATE   = "2024-12-31 23:59:59"
OUTPUT_FILE = "./EXCEL_RESULT_BACKTEST/backtest_results_XAUUSD_excel_2020_2024_3_time10UTC_530UTC.xlsx"
EXCEL_FILE = "./INPUT_DATA_EXCEL/XAUUSD_2020_2024_15m.xlsx"  # <-- Your Excel file

EMA_PERIOD = 50
SWING_LOOKBACK = 3
BODY_AVG_WINDOW = 10
RR = 2.1
# --------------------------------------------

# ---------- Helpers ----------
def pip_size_for_pair(pair: str) -> float:
    """Auto-detect pip size (supports JPY pairs)."""
    return 0.01 if "JPY" in pair.replace("/", "").upper() else 0.0001

PIP_SIZE = pip_size_for_pair(PAIR)

def calculate_pip_values(entry: float, sl: float, exit_price: float, direction: str, rr: float):
    """Return pip_risk, pip_outcome based on entry, stop, and exit."""
    if direction == "LONG":
        pip_risk = abs(entry - sl) / PIP_SIZE
        pip_outcome = (exit_price - entry) / PIP_SIZE
    else:  # SHORT
        pip_risk = abs(sl - entry) / PIP_SIZE
        pip_outcome = (entry - exit_price) / PIP_SIZE
    return round(pip_risk, 2), round(pip_outcome, 2)

# ---------- Data Loading ----------
def load_15m_data_from_excel(filename: str) -> pd.DataFrame:
    df = pd.read_excel(filename)

    # Remove unnamed columns (if any)
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, regex=True)]

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Expected columns: time, open, high, low, close
    if not {"time", "open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError(f"Unexpected columns found: {df.columns}")

    # Rename for consistency
    df = df.rename(columns={
        "time": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    })

    # Convert types
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)  # keep in UTC
    df = df.dropna(subset=["Datetime"]).copy()

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df = df.sort_values("Datetime").reset_index(drop=True)

    return df

def clip_date_range(df: pd.DataFrame, start_ts: str, end_ts: str) -> pd.DataFrame:
    st = pd.to_datetime(start_ts, utc=True)   # force UTC
    en = pd.to_datetime(end_ts, utc=True)     # force UTC
    return df[(df["Datetime"] >= st) & (df["Datetime"] <= en)].copy()

def add_ema(df: pd.DataFrame, period: int = EMA_PERIOD) -> pd.DataFrame:
    df = df.copy()
    df["EMA50"] = df["Close"].ewm(span=period, adjust=False).mean()
    return df

def add_swings(df: pd.DataFrame, lb: int = SWING_LOOKBACK) -> pd.DataFrame:
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
    return abs(float(s["Close"]) - float(s["Open"]))

def three_candle_breakout(df15: pd.DataFrame, idx: int, trend_dir: str):
    if idx < BODY_AVG_WINDOW + 2:
        return None
    ts = df15.iloc[idx]["Datetime"]
    if not (dtime(10, 0) <= ts.time() <= dtime(18, 0)):
     return None
    c1 = df15.iloc[idx-2]
    c2 = df15.iloc[idx-1]
    c3 = df15.iloc[idx]
    bavg = df15.iloc[idx-BODY_AVG_WINDOW:idx]["Close"].sub(
        df15.iloc[idx-BODY_AVG_WINDOW:idx]["Open"]
    ).abs().mean()
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
    prior = df15.iloc[:idx-1]
    last_swing_high = prior["Swing_High"].dropna().iloc[-1] if prior["Swing_High"].dropna().size else np.nan
    last_swing_low  = prior["Swing_Low"].dropna().iloc[-1]  if prior["Swing_Low"].dropna().size  else np.nan
    if trend_dir == "LONG":
        if np.isnan(last_swing_high) or c3["Close"] <= last_swing_high:
            return None
    else:
        if np.isnan(last_swing_low) or c3["Close"] >= last_swing_low:
            return None
    return {"c1": c1, "c2": c2, "c3": c3, "last_swing_high": last_swing_high, "last_swing_low": last_swing_low}

def compute_fvg(bounds: dict, direction: str):
    c1 = bounds["c1"]
    c3 = bounds["c3"]
    if direction == "LONG":
        high1 = float(c1["High"])
        low3  = float(c3["Low"])
        if high1 < low3:
            return (high1, low3)
    else:
        low1  = float(c1["Low"])
        high3 = float(c3["High"])
        if low1 > high3:
            return (high3, low1)
    return None

def current_trend(df15: pd.DataFrame, idx: int):
    row = df15.iloc[idx]
    if np.isnan(row["EMA50"]):
        return None
    return "LONG" if float(row["Close"]) > float(row["EMA50"]) else "SHORT"

# ---------------- Trade Simulation on 15m ----------------
def simulate_trade_on_15m(df15, start_idx, entry, sl, tp, direction):
    entry_time = df15.iloc[start_idx]["Datetime"]
    entry = float(entry); sl = float(sl); tp = float(tp)

    for j in range(start_idx + 1, len(df15)):
        row = df15.iloc[j]
        high, low, ts = float(row["High"]), float(row["Low"]), row["Datetime"]

        if direction == "LONG":
            if low <= sl:  # SL hit
                pip_risk, pip_outcome = calculate_pip_values(entry, sl, sl, direction, RR)
                return entry_time, ts, "LOSS", pip_risk, pip_outcome
            if high >= tp:  # TP hit
                pip_risk, pip_outcome = calculate_pip_values(entry, sl, tp, direction, RR)
                return entry_time, ts, "WIN", pip_risk, pip_outcome

        else:  # SHORT
            if high >= sl:  # SL hit
                pip_risk, pip_outcome = calculate_pip_values(entry, sl, sl, direction, RR)
                return entry_time, ts, "LOSS", pip_risk, pip_outcome
            if low <= tp:  # TP hit
                pip_risk, pip_outcome = calculate_pip_values(entry, sl, tp, direction, RR)
                return entry_time, ts, "WIN", pip_risk, pip_outcome

    return None

# ---------------- Backtest ----------------
def backtest():
    print("Loading 15m data from Excel...")
    df15 = load_15m_data_from_excel(EXCEL_FILE)
    df15 = clip_date_range(df15, START_DATE, END_DATE)
    if df15.empty:
        print("❌ No data in the requested date range.")
        return
    df15 = add_ema(df15, EMA_PERIOD)
    df15 = add_swings(df15, SWING_LOOKBACK)

    results = []
    start_i = max(EMA_PERIOD, BODY_AVG_WINDOW) + 2
    for i in range(start_i, len(df15)):
        trend = current_trend(df15, i)
        if not trend:
            continue
        bo = three_candle_breakout(df15, i, trend)
        if not bo:
            continue
        fvg = compute_fvg(bo, trend)
        if not fvg:
            continue
        fvg_low, fvg_high = fvg
        entry = float((fvg_low + fvg_high) / 2.0)
        c1 = bo["c1"]

        if trend == "LONG":
            sl = float(c1["Low"])
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + RR * risk
        else:
            sl = float(c1["High"])
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - RR * risk

        sim = simulate_trade_on_15m(df15, i, entry, sl, tp, trend)
        if not sim:
            continue

        entry_time, exit_time, result, pip_risk, pip_outcome = sim
        results.append({
            "PAIR TRADED": PAIR,
            "DATE OPEN": entry_time.strftime("%Y-%m-%d %H:%M"),
            "DATE CLOSED": exit_time.strftime("%Y-%m-%d %H:%M"),
            "DAY OF WEEK": entry_time.strftime("%A"),
            "TIME": entry_time.strftime("%H:%M"),
            "PIP RISKED": pip_risk,
            "DIRECTION(LONG OR SHORT)": trend,
            "RESULT": result,
            "PIP OUTCOME": pip_outcome
        })

    if results:
        df_out = pd.DataFrame(results)
        try:
            if os.path.exists(OUTPUT_FILE):
                existing = pd.read_excel(OUTPUT_FILE)
                df_out = pd.concat([existing, df_out], ignore_index=True)
            df_out.to_excel(OUTPUT_FILE, index=False)
            print(f"✅ Saved {len(results)} trades to {OUTPUT_FILE}")
        except PermissionError:
            alt = OUTPUT_FILE.replace(".xlsx", "_new.xlsx")
            df_out.to_excel(alt, index=False)
            print(f"⚠️ {OUTPUT_FILE} was open. Saved to {alt} instead.")
    else:
        print("⚠️ No executed trades found for the specified period with given rules.")

if __name__ == "__main__":
    try:
        backtest()
    except Exception as e:
        print(f"An error occurred: {e}")
