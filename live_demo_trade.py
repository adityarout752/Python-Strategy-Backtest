import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
import logging

# =========================
# MT4 LIVE TRADING - BPB FVG STRATEGY
# =========================

# MT4 Configuration
ACCOUNT    = ""                  # <-- your MT4 demo account number
PASSWORD   = ""                  # <-- your MT4 demo password
SERVER     = ""                  # <-- broker server name from MT4 login window
SYMBOL     = "EURUSD"            # pair to trade
LOT_SIZE   = 0.1                 # adjust for risk
RR         = 2.1                 # risk-reward ratio
TIMEFRAME  = "M15"               # 15m chart
EMA_PERIOD = 50
SWING_LOOKBACK = 3
BODY_AVG_WINDOW = 10
LOG_FILE = "live_trades_mt4.xlsx"

# MT4 API Setup (you'll need to install the MT4 Python library)
# pip install MetaTrader4

try:
    import MetaTrader4 as mt4
    MT4_AVAILABLE = True
except ImportError:
    print("⚠️ MetaTrader4 library not installed. Please install with: pip install MetaTrader4")
    MT4_AVAILABLE = False

# =========================
# LOGGING SETUP
# =========================
def setup_logging():
    """Set up logging for the trading system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('mt4_trading.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =========================
# MT4 CONNECTION
# =========================
def connect_mt4():
    """Connect to MT4 terminal."""
    if not MT4_AVAILABLE:
        logger.error("MetaTrader4 library not available")
        return False

    if not mt4.initialize():
        logger.error(f"MT4 initialize() failed: {mt4.last_error()}")
        return False

    # Login to account
    authorized = mt4.login(ACCOUNT, password=PASSWORD, server=SERVER)
    if authorized:
        logger.info(f"✅ Connected to MT4 account: {ACCOUNT}")
        account_info = mt4.account_info()
        if account_info:
            logger.info(f"Account Balance: ${account_info.balance:.2f}")
            logger.info(f"Account Equity: ${account_info.equity:.2f}")
        return True
    else:
        logger.error(f"MT4 Login failed: {mt4.last_error()}")
        return False

# =========================
# DATA FETCHING
# =========================
def get_mt4_data(symbol=SYMBOL, n=300):
    """Fetch historical data from MT4."""
    if not MT4_AVAILABLE:
        logger.error("MT4 not available for data fetching")
        return None

    try:
        # Get rates from MT4
        rates = mt4.copy_rates_from_pos(symbol, mt4.TIMEFRAME_M15, 0, n)

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get rates for {symbol}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={
            'time': 'Datetime',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)

        return df

    except Exception as e:
        logger.error(f"Error fetching MT4 data: {e}")
        return None

# =========================
# STRATEGY HELPERS (Same as backtesting)
# =========================
def add_ema(df, period=EMA_PERIOD):
    """Add EMA to DataFrame."""
    df = df.copy()
    df["EMA50"] = df["Close"].ewm(span=period, adjust=False).mean()
    return df

def add_swings(df, lb=SWING_LOOKBACK):
    """Add swing highs and lows to DataFrame."""
    df = df.copy()
    highs = df["High"].values
    lows  = df["Low"].values
    swing_high = [np.nan] * len(df)
    swing_low  = [np.nan] * len(df)

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

def body_size(candle):
    """Calculate candle body size."""
    return abs(candle["Close"] - candle["Open"])

def three_candle_breakout(df, idx, trend_dir):
    """Detect 3-candle breakout pattern."""
    if idx < BODY_AVG_WINDOW + 2:
        return None

    ts = df.iloc[idx]["Datetime"]
    if ts.time().hour < 12:  # only trade after 12:00
        return None

    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]

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
    """Compute Fair Value Gap."""
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

def current_trend(df, idx):
    """Determine current trend."""
    row = df.iloc[idx]
    if np.isnan(row["EMA50"]):
        return None
    return "LONG" if float(row["Close"]) > float(row["EMA50"]) else "SHORT"

# =========================
# SIGNAL GENERATOR
# =========================
def signal_generator(df):
    """Generate trading signals based on BPB FVG strategy."""
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
        if risk <= 0:
            return None
        tp = entry + RR * risk
    else:
        sl = float(c1["High"])
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - RR * risk

    return (trend.lower(), entry, sl, tp)

# =========================
# ORDER EXECUTION
# =========================
def place_mt4_order(symbol, lot, side, sl, tp):
    """Place order through MT4."""
    if not MT4_AVAILABLE:
        logger.error("MT4 not available for order execution")
        return None

    try:
        # Get current price
        tick = mt4.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick info for {symbol}")
            return None

        price = tick.ask if side == "buy" else tick.bid

        # Determine order type
        order_type = mt4.ORDER_TYPE_BUY if side == "buy" else mt4.ORDER_TYPE_SELL

        # Create order request
        request = {
            "action": mt4.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "BPB FVG Strategy",
            "type_time": mt4.ORDER_TIME_GTC,
            "type_filling": mt4.ORDER_FILLING_IOC,
        }

        # Send order
        result = mt4.order_send(request)
        if result is not None:
            if result.retcode == mt4.TRADE_RETCODE_DONE:
                logger.info(f"✅ Order executed: {result.order}")
                return result
            else:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return result
        else:
            logger.error("Order send returned None")
            return None

    except Exception as e:
        logger.error(f"Error placing MT4 order: {e}")
        return None

# =========================
# LOGGING
# =========================
def log_trade(symbol, side, entry, sl, tp, result):
    """Log trade details to Excel."""
    try:
        trade_data = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Symbol": symbol,
            "Side": side,
            "Entry": entry,
            "SL": sl,
            "TP": tp,
            "OrderResult": str(result) if result else "FAILED"
        }

        df = pd.DataFrame([trade_data])

        if os.path.exists(LOG_FILE):
            old = pd.read_excel(LOG_FILE)
            df = pd.concat([old, df], ignore_index=True)

        df.to_excel(LOG_FILE, index=False)
        logger.info(f"✅ Trade logged to {LOG_FILE}")

    except Exception as e:
        logger.error(f"Error logging trade: {e}")

# =========================
# RISK MANAGEMENT
# =========================
def calculate_position_size(account_balance, risk_percent=1.0):
    """Calculate position size based on account risk."""
    # This is a basic implementation - you may want to make it more sophisticated
    return LOT_SIZE  # For now, use fixed lot size

# =========================
# MAIN TRADING LOOP
# =========================
def run_trading():
    """Main trading loop."""
    logger.info("🚀 Starting BPB FVG MT4 Live Trading...")

    if not connect_mt4():
        logger.error("Failed to connect to MT4. Exiting.")
        return

    consecutive_failures = 0
    max_failures = 5

    while True:
        try:
            # Get data
            df = get_mt4_data(SYMBOL)
            if df is None or len(df) < 100:
                logger.warning("Insufficient data, waiting...")
                time.sleep(60)
                continue

            # Generate signal
            signal = signal_generator(df)

            if signal:
                side, entry, sl, tp = signal
                logger.info(f"📈 Signal: {side.upper()} @ {entry".5f"} | SL={sl".5f"}, TP={tp".5f"}")

                # Calculate position size
                account_info = mt4.account_info()
                if account_info:
                    lot_size = calculate_position_size(account_info.balance)
                else:
                    lot_size = LOT_SIZE

                # Place order
                result = place_mt4_order(SYMBOL, lot_size, side, sl, tp)

                # Log trade
                log_trade(SYMBOL, side, entry, sl, tp, result)

                consecutive_failures = 0  # Reset failure counter

            else:
                logger.info("⏳ No signal this candle. Waiting...")

            # Wait for next candle (15 minutes)
            logger.info("⏳ Waiting for next 15m candle...")
            time.sleep(60 * 15)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            consecutive_failures += 1

            if consecutive_failures >= max_failures:
                logger.error("Too many consecutive failures. Attempting to reconnect...")
                mt4.shutdown()
                time.sleep(10)
                if connect_mt4():
                    consecutive_failures = 0
                    logger.info("✅ Reconnected to MT4")
                else:
                    logger.error("Failed to reconnect. Waiting before retry...")
                    time.sleep(60)

            time.sleep(60)

# =========================
# START TRADING
# =========================
if __name__ == "__main__":
    if not MT4_AVAILABLE:
        print("❌ MetaTrader4 library not installed!")
        print("Please install it with: pip install MetaTrader4")
        print("Then update your account credentials in the script.")
    else:
        print("✅ MT4 library available")
        print("📝 Please update your account credentials before running:")
        print("   - ACCOUNT number")
        print("   - PASSWORD")
        print("   - SERVER name")
        print("\n🚀 Starting live trading...")

        try:
            run_trading()
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
            if MT4_AVAILABLE:
                mt4.shutdown()
        except Exception as e:
            print(f"💥 Fatal error: {e}")
            if MT4_AVAILABLE:
                mt4.shutdown()
