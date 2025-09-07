📊 Forex Backtesting Script

This project provides a Python-based backtesting engine for the EUR/USD (or any forex pair).
It simulates trades based on a three-candle breakout strategy with Fair Value Gap (FVG) confirmation on 15-minute charts.

⚙️ Features

Loads 15-minute OHLC data from Excel

Adds EMA(50) and swing high/low detection

Detects 3 consecutive same-direction candles after 12:00

Confirms breakout vs. last swing high/low

Identifies Fair Value Gap (FVG) between candles 1 and 3

Places trades with stop loss (SL) and take profit (TP) based on Risk/Reward ratio

Simulates trades bar by bar until SL/TP is hit

Saves results to Excel output file

📂 Project Structure
backtest.py                 # Main backtesting script
EURUSD_2024_15m_data.xlsx   # Input OHLCV data (15-minute timeframe)
backtest_results_excel.xlsx # Output file with trade results
chart.py                    # (Optional) Trade visualization utilities
README.md                   # Documentation

🔧 Requirements

Python 3.9+

Libraries:

pip install pandas numpy openpyxl


Input Excel file with columns:

Time | Open | High | Low | Close

🛠️ How It Works

Load Data
Reads 15-minute OHLC data from Excel and cleans it.

Indicators & Swings

Adds EMA(50) for trend detection

Marks swing highs and swing lows

Strategy Rules

Check for 3 bullish/bearish candles after 12:00

Each body size ≥ ⅓ of recent average body size

Last candle must close beyond last swing high/low

Confirm a Fair Value Gap (FVG) between candle 1 and 3

Trade Execution

Entry at midpoint of FVG

Stop Loss at candle 1 low/high

Take Profit = SL distance × RR (default RR = 2.1)

Simulated bar by bar until SL/TP is hit

Results
Each trade is logged with:

Pair

Entry/Exit time

Day & Hour

Direction

Risk (pips)

Outcome (WIN/LOSS)

Result (pips gained/lost)

📑 Example Output

Output Excel (backtest_results_excel.xlsx) will contain:

PAIR TRADED	DATE OPEN	DATE CLOSED	DAY OF WEEK	TIME	PIP RISKED	DIRECTION	RESULT	PIP OUTCOME
EUR/USD	2024-03-15 14:15	2024-03-15 17:30	Friday	14:15	20	LONG	WIN	42
EUR/USD	2024-04-02 13:45	2024-04-02 15:00	Tuesday	13:45	18	SHORT	LOSS	-18
▶️ Usage

Run the backtest:

python backtest.py


Results will be saved to:

backtest_results_excel.xlsx (or _new.xlsx if file is open)

⚡ Configuration

Inside backtest.py you can adjust:

PAIR = "EUR/USD"
START_DATE = "2024-01-01 00:00:00"
END_DATE   = "2024-12-31 23:59:59"
EMA_PERIOD = 50
SWING_LOOKBACK = 3
BODY_AVG_WINDOW = 10
RR = 2.1

📈 Next Steps

Add charts for entry/exit visualization

Expand to multiple pairs

Add equity curve & performance metrics (win rate, drawdown, etc.)
