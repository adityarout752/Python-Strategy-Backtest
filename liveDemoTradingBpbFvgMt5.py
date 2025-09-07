import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta

# =========================
#  not tested it yet
# =========================
ACCOUNT    = 1234567             # <-- your MT5 demo account number
PASSWORD   = "your_password"     # <-- your MT5 demo password
SERVER     = "Broker-Demo"       # <-- broker server name from MT5 login window
SYMBOL     = "EURUSD"            # pair to trade
LOT_SIZE   = 0.1                 # adjust for risk
RR         = 2.1                 # risk-reward ratio
TIMEFRAME  = mt5.TIMEFRAME_M15   # 15m chart
EMA_PERIOD = 50
SWING_LOOKBACK = 3
BODY_AVG_WINDOW = 10
LOG_FILE = "live_trades.xlsx"

# =========================
# MT5 CONNECTION
# =========================
def connect():
    if not mt5.initialize():
        print("initialize() failed", mt5.last_error())
        quit()
    authorized = mt5.login(ACCOUNT, password=PASSWORD, server=SERVER)
    if authorized:
        print("Connected to MT5 account:", ACCOUNT)
    else:
        print("Login failed:", mt5.last_error())
        quit()

# =========================
# FETCH DATA
# =========================
def get_data(symbol=SYMBOL, n=300):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'time':'Datetime','open':'Open','high':'High','low':'Low','close':'Close'}, inplace=True)
    return df

# =========================
# STRATEGY HELPERS
# =========================
def add_ema(df, period=EMA_PERIOD):
    df["EMA50"] = df["Close"].ewm(span=period, adjust=False).mean()
    return df

def add_swings(df, lb=SWING_LOOKBACK):
    highs, lows = df["High"].values, df["Low"].values
    swing_high, swing_low = [np.nan]*len(df), [np.nan]*len(df)
    for i in range(lb, len(df)-lb):
        if highs[i] > max(highs[i-lb:i]) and highs[i] > max(highs[i+1:i+1+lb]):
            swing_high[i] = highs[i]
        if lows[i] < min(lows[i-lb:i]) and lows[i] < min(lows[i+1:i+1+lb]):
            swing_low[i] = lows[i]
    df["Swing_High"] = swing_high
    df["Swing_Low"]  = swing_low
    return df

def body_size(candle):
    return abs(candle["Close"] - candle["Open"])

def three_candle_breakout(df, idx, trend_dir):
    if idx < BODY_AVG_WINDOW + 2:
        return None
    ts = df.iloc[idx]["Datetime"]
    if ts.time().hour < 12:  # only trade after 12:00
        return None

    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    bavg = df.iloc[idx-BODY_AVG_WINDOW:idx]["Close"].sub(
        df.iloc[idx-BODY_AVG_WINDOW:idx]["Open"]
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

    prior = df.iloc[:idx-1]
    last_swing_high = prior["Swing_High"].dropna().iloc[-1] if prior["Swing_High"].dropna().size else np.nan
    last_swing_low  = prior["Swing_Low"].dropna().iloc[-1]  if prior["Swing_Low"].dropna().size  else np.nan

    if trend_dir == "LONG":
        if np.isnan(last_swing_high) or c3["Close"] <= last_swing_high:
            return None
    else:
        if np.isnan(last_swing_low) or c3["Close"] >= last_swing_low:
            return None

    return {"c1": c1, "c2": c2, "c3": c3, "last_swing_high": last_swing_high, "last_swing_low": last_swing_low}

def compute_fvg(bounds, direction):
    c1, c3 = bounds["c1"], bounds["c3"]
    if direction == "LONG":
        if c1["High"] < c3["Low"]:
            return (c1["High"], c3["Low"])
    else:
        if c1["Low"] > c3["High"]:
            return (c3["High"], c1["Low"])
    return None

def current_trend(df, idx):
    row = df.iloc[idx]
    if np.isnan(row["EMA50"]):
        return None
    return "LONG" if row["Close"] > row["EMA50"] else "SHORT"

# =========================
# SIGNAL GENERATOR
# =========================
def signal_generator(df):
    df = add_ema(df)
    df = add_swings(df)

    idx = len(df) - 1
    trend = current_trend(df, idx)
    if not trend:
        return None

    bo = three_candle_breakout(df, idx, trend)
    if not bo:
        return None

    fvg = compute_fvg(bo, trend)
    if not fvg:
        return None

    fvg_low, fvg_high = fvg
    entry = (fvg_low + fvg_high) / 2.0
    c1 = bo["c1"]

    if trend == "LONG":
        sl = float(c1["Low"])
        risk = entry - sl
        if risk <= 0: return None
        tp = entry + RR * risk
    else:
        sl = float(c1["High"])
        risk = sl - entry
        if risk <= 0: return None
        tp = entry - RR * risk

    return (trend.lower(), entry, sl, tp)

# =========================
# ORDER EXECUTION
# =========================
def place_order(symbol, lot, side, sl, tp):
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if side == "buy" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": "Python MT5 Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print("Order result:", result)
    return result

# =========================
# LOGGING
# =========================
def log_trade(symbol, side, entry, sl, tp, result):
    trade_data = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Symbol": symbol,
        "Side": side,
        "Entry": entry,
        "SL": sl,
        "TP": tp,
        "OrderResult": str(result)
    }
    df = pd.DataFrame([trade_data])

    if os.path.exists(LOG_FILE):
        old = pd.read_excel(LOG_FILE)
        df = pd.concat([old, df], ignore_index=True)

    df.to_excel(LOG_FILE, index=False)
    print(f"✅ Trade logged to {LOG_FILE}")

# =========================
# MAIN LOOP
# =========================
def run():
    connect()
    while True:
        df = get_data(SYMBOL)
        signal = signal_generator(df)

        if signal:
            side, entry, sl, tp = signal
            print(f"📈 Signal: {side.upper()} @ {entry:.5f} | SL={sl:.5f}, TP={tp:.5f}")
            result = place_order(SYMBOL, LOT_SIZE, side, sl, tp)
            log_trade(SYMBOL, side, entry, sl, tp, result)
        else:
            print("No signal this candle.")

        print("⏳ Waiting for next 15m candle...")
        time.sleep(60*15)

# =========================
# START
# =========================
if __name__ == "__main__":
    run()
