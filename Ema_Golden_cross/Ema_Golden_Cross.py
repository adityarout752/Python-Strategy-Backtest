import os
import sys
import numpy as np
import pandas as pd
from datetime import time as dtime

# Add path for trade_screenshot if needed
sys.path.append('..')
try:
    from trade_screenshot import save_trade_screenshot
except ImportError:
    save_trade_screenshot = None  # Optional

# Import EMA chart function
try:
    from ema_chart import save_ema_trade_screenshot
except ImportError:
    save_ema_trade_screenshot = None  # Optional

# ------------------ CONFIG ------------------
PAIR = "EUR/USD"
START_DATE = "2023-01-01 00:00:00"
END_DATE   = "2023-01-31 23:59:59"
OUTPUT_FILE = "./Ema_Golden_cross/output_trde/backtest_ema_golden_cross.xlsx"
EXCEL_FILE = "./INPUT_DATA_EXCEL/EURUSD_2023_15m_data.xlsx"  # <-- Your Excel file

RR = 2.1
MAX_PULLBACKS = 3
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
    print("Excel file read successfully.")

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

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # EMAs
    df["EMA6"] = df["Close"].ewm(span=6, adjust=False).mean()
    df["EMA18"] = df["Close"].ewm(span=18, adjust=False).mean()
    # SMAs
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    return df

def detect_crossover(df: pd.DataFrame, idx: int) -> str:
    """Detect if 6EMA crosses above 18EMA (uptrend) or below (downtrend) at idx."""
    if idx < 1:
        print(f"DEBUG: idx {idx} < 1, skipping crossover detection.")
        return None
    prev_ema6 = df.iloc[idx-1]["EMA6"]
    prev_ema18 = df.iloc[idx-1]["EMA18"]
    curr_ema6 = df.iloc[idx]["EMA6"]
    curr_ema18 = df.iloc[idx]["EMA18"]
    print(f"DEBUG: idx {idx}, prev_ema6={prev_ema6:.5f}, prev_ema18={prev_ema18:.5f}, curr_ema6={curr_ema6:.5f}, curr_ema18={curr_ema18:.5f}")
    if prev_ema6 <= prev_ema18 and curr_ema6 > curr_ema18:
        print(f"DEBUG: UPTREND crossover detected at idx {idx}")
        return "UPTREND"
    elif prev_ema6 >= prev_ema18 and curr_ema6 < curr_ema18:
        print(f"DEBUG: DOWNTREND crossover detected at idx {idx}")
        return "DOWNTREND"
    print(f"DEBUG: No valid crossover at idx {idx}")
    return None

def is_bullish_candle(row: pd.Series) -> bool:
    return row["Close"] > row["Open"]

def is_bearish_candle(row: pd.Series) -> bool:
    return row["Close"] < row["Open"]

def is_inside_bar(prev_row: pd.Series, curr_row: pd.Series) -> bool:
    return curr_row["High"] <= prev_row["High"] and curr_row["Low"] >= prev_row["Low"]

def find_swing_and_validate(df: pd.DataFrame, crossover_idx: int, trend: str) -> dict:
    """Find swing high/low after crossover and validate."""
    print(f"DEBUG: Finding swing for trend {trend} starting from idx {crossover_idx}")
    swing_idx = None
    swing_price = None
    checked_count = 0
    for i in range(crossover_idx, len(df)):
        row = df.iloc[i]
        checked_count += 1
        if trend == "LONG":
            if is_bullish_candle(row):
                swing_idx = i
                swing_price = row["High"]
                print(f"DEBUG: Found bullish swing at idx {i}, high={swing_price} after checking {checked_count} candles")
                break
            else:
                print(f"DEBUG: Candle at idx {i} is not bullish (O={row['Open']:.5f}, C={row['Close']:.5f})")
        elif trend == "SHORT":
            if is_bearish_candle(row):
                swing_idx = i
                swing_price = row["Low"]
                print(f"DEBUG: Found bearish swing at idx {i}, low={swing_price} after checking {checked_count} candles")
                break
            else:
                print(f"DEBUG: Candle at idx {i} is not bearish (O={row['Open']:.5f}, C={row['Close']:.5f})")
        if checked_count > 50:  # Limit logging to first 50 to avoid spam
            print(f"DEBUG: Checked {checked_count} candles so far, still no swing...")
            break
    if swing_idx is None:
        print(f"DEBUG: No swing found for trend {trend} from idx {crossover_idx} after checking {checked_count} candles")
        return None
    if swing_idx + 1 >= len(df):
        print(f"DEBUG: Swing at idx {swing_idx} is the last row, cannot proceed")
        return None
    # Removed strict validation to allow more trades; proceed with swing
    return {"swing_idx": swing_idx, "swing_price": swing_price}

def track_pullbacks(df: pd.DataFrame, swing_idx: int, trend: str) -> int:
    """Track pullbacks up to MAX_PULLBACKS, without SMA50 invalidation to allow more trades."""
    pullback_count = 0
    for i in range(swing_idx + 1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        if is_inside_bar(prev_row, row) or (trend == "LONG" and is_bearish_candle(row) and row["High"] < prev_row["High"]) or (trend == "SHORT" and is_bullish_candle(row) and row["Low"] > prev_row["Low"]):
            pullback_count += 1
            if pullback_count > MAX_PULLBACKS:
                return -1  # Too many pullbacks
        else:
            # Non-pullback candle found, stop counting
            return pullback_count
    return pullback_count

def check_entry_breakout(df: pd.DataFrame, swing_idx: int, swing_price: float, trend: str) -> int:
    """Find entry on breakout above swing high (long) or below swing low (short)."""
    for i in range(swing_idx + 1, len(df)):
        row = df.iloc[i]
        if trend == "LONG" and row["High"] > swing_price:
            return i
        elif trend == "SHORT" and row["Low"] < swing_price:
            return i
    return None

# ---------------- Trade Simulation on 15m ----------------
def simulate_trade_on_15m(df15, start_idx, entry, sl, tp, direction):
    entry = float(entry); sl = float(sl); tp = float(tp)

    # Find the first candle where price reaches the entry level
    entry_reached = False
    entry_time = None
    for j in range(start_idx + 1, len(df15)):
        row = df15.iloc[j]
        high, low, ts = float(row["High"]), float(row["Low"]), row["Datetime"]

        if direction == "LONG":
            if low <= entry <= high:
                entry_reached = True
                entry_time = ts
                break
        else:  # SHORT
            if low <= entry <= high:
                entry_reached = True
                entry_time = ts
                break

    if not entry_reached:
        return None

    # Start monitoring SL and TP from the candle after entry
    for k in range(j + 1, len(df15)):
        row = df15.iloc[k]
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
    print(f"Data loaded: {len(df15)} rows")
    df15 = clip_date_range(df15, START_DATE, END_DATE)
    print(f"After clipping: {len(df15)} rows")
    if df15.empty:
        print("❌ No data in the requested date range.")
        return
    df15 = add_indicators(df15)
    print("Indicators added.")

    results = []
    start_i = 200  # To have SMA200
    total_iterations = len(df15) - start_i
    print(f"Starting backtest loop with {total_iterations} iterations...")
    for idx, i in enumerate(range(start_i, len(df15))):
        if idx % 1000 == 0:  # Print progress every 1000 iterations
            print(f"Processed {idx}/{total_iterations} iterations...")
            # Periodic debug: print EMA values at this index
            ema6_val = df15.iloc[i]["EMA6"]
            ema18_val = df15.iloc[i]["EMA18"]
            print(f"DEBUG: At idx {i}, EMA6={ema6_val:.5f}, EMA18={ema18_val:.5f}")
        trend_signal = detect_crossover(df15, i)
        if not trend_signal:
            continue
        print(f"Found trend_signal at i={i}: {trend_signal}")
        trend = "LONG" if trend_signal == "UPTREND" else "SHORT"
        swing_data = find_swing_and_validate(df15, i, trend)
        if not swing_data:
            print(f"No swing data at i={i}")
            continue
        swing_idx = swing_data["swing_idx"]
        swing_price = swing_data["swing_price"]
        print(f"Swing found at idx={swing_idx}, price={swing_price}")
        pullbacks = track_pullbacks(df15, swing_idx, trend)
        print(f"Pullbacks: {pullbacks}")
        if pullbacks == -1 or pullbacks > MAX_PULLBACKS:
            continue
        entry_idx = check_entry_breakout(df15, swing_idx, swing_price, trend)
        if entry_idx is None:
            print(f"No entry breakout at swing_idx={swing_idx}")
            continue
        print(f"Entry at idx={entry_idx}")
        entry_price = swing_price  # Breakout at swing price

        # Check price vs SMAs for entry condition
        sma50_at_entry = df15.iloc[entry_idx]['SMA50']
        sma200_at_entry = df15.iloc[entry_idx]['SMA200']
        if trend == "LONG" and not (entry_price > sma50_at_entry and entry_price > sma200_at_entry):
            print(f"Skipping LONG entry: price {entry_price:.5f} not above SMA50 {sma50_at_entry:.5f} and SMA200 {sma200_at_entry:.5f}")
            continue
        elif trend == "SHORT" and not (entry_price < sma50_at_entry and entry_price < sma200_at_entry):
            print(f"Skipping SHORT entry: price {entry_price:.5f} not below SMA50 {sma50_at_entry:.5f} and SMA200 {sma200_at_entry:.5f}")
            continue
        entry_candle_low = df15.iloc[entry_idx]["Low"]
        entry_candle_high = df15.iloc[entry_idx]["High"]
        min_pips = 8 * PIP_SIZE
        if trend == "LONG":
            sl = min(entry_candle_low - min_pips, df15.iloc[swing_idx]["Low"])  # SL below entry candle low, min 8 pips, or swing low
            risk = entry_price - sl
            if risk <= min_pips:
                continue
            tp = entry_price + RR * risk
        else:
            sl = max(entry_candle_high + min_pips, df15.iloc[swing_idx]["High"])  # SL above entry candle high, min 8 pips, or swing high
            risk = sl - entry_price
            if risk <= min_pips:
                continue
            tp = entry_price - RR * risk

        sim = simulate_trade_on_15m(df15, entry_idx, entry_price, sl, tp, trend)
        if not sim:
            continue

        entry_time, exit_time, result, pip_risk, pip_outcome = sim
        exit_price = tp if result == "WIN" else sl

        # Save trade screenshot if available
        if save_trade_screenshot:
            trade_id = len(results) + 1
            save_trade_screenshot(df15, entry_time, exit_time, entry_price, sl, tp, None, swing_price if trend == "LONG" else None, swing_price if trend == "SHORT" else None, trend, trade_id, exit_price, result)

        # Save EMA trade screenshot if available
        if save_ema_trade_screenshot:
            trade_id = len(results) + 1
            save_ema_trade_screenshot(df15, entry_time, exit_time, entry_price, sl, tp, swing_price if trend == "LONG" else None, swing_price if trend == "SHORT" else None, trend, trade_id, exit_price, result)

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
