# BPB FVG Trading Strategy Documentation

## Overview
This document explains the BPB FVG (Breakout Pullback Fair Value Gap) trading strategy implemented in the `BPB_FVG.py` script. The strategy is designed for EUR/USD on 15-minute charts and uses technical indicators to identify high-probability trade setups.

## Code Structure Explanation

### Main Components
1. **Configuration**: Sets trading pair, date range, file paths, and parameters like EMA period, swing lookback, body average window, and risk-reward ratio.

2. **Helpers**:
   - `pip_size_for_pair()`: Determines pip size based on currency pair (0.0001 for most pairs, 0.01 for JPY pairs).
   - `calculate_pip_values()`: Calculates pip risk and outcome for trades.

3. **Data Loading**:
   - `load_15m_data_from_excel()`: Loads 15-minute OHLC data from Excel, cleans and formats it.
   - `clip_date_range()`: Filters data to specified date range.
   - `add_ema()`: Adds Exponential Moving Average (EMA50).
   - `add_swings()`: Identifies swing highs and lows using lookback period.

4. **Trade Logic**:
   - `three_candle_breakout()`: Detects 3-candle breakouts in trend direction.
   - `compute_fvg()`: Calculates Fair Value Gap from breakout candles.
   - `current_trend()`: Determines trend based on close vs EMA50.

5. **Simulation**:
   - `simulate_trade_on_15m()`: Simulates trade execution on 15m data, checking for SL/TP hits.
   - `backtest()`: Main function that runs the backtest, identifies setups, simulates trades, and saves results.

### How the Code Works
- Loads and preprocesses data.
- Iterates through each 15m candle.
- Checks for trend, breakout, and FVG conditions.
- If all conditions met, enters trade and simulates until exit.
- Records trade results and saves to Excel.

## Trading Strategy Rules

The BPB FVG strategy combines trend following, breakout detection, and gap analysis. Here's the step-by-step rules:

### 1. Trend Determination
- **LONG Trend**: Current close price > EMA50
- **SHORT Trend**: Current close price < EMA50
- Only trade in the direction of the prevailing trend.

### 2. Three-Candle Breakout Setup
- Look for 3 consecutive candles of the same color:
  - **LONG**: 3 green candles (close > open)
  - **SHORT**: 3 red candles (close < open)
- Each candle's body size must be ≥ average body size of last 10 candles / 3
- Setup must occur after 12:00 (noon) to avoid early session volatility
- The 3rd candle must break:
  - **LONG**: Above the most recent swing high
  - **SHORT**: Below the most recent swing low

### 3. Fair Value Gap (FVG) Calculation
- **LONG FVG**: If high of 1st candle < low of 3rd candle, FVG = (high1, low3)
- **SHORT FVG**: If low of 1st candle > high of 3rd candle, FVG = (high3, low1)
- FVG represents an "unfair" price gap that the market should fill.

### 4. Entry Rules
- Enter at the midpoint of the FVG: (FVG_low + FVG_high) / 2
- Only enter if FVG exists and conditions are met.

### 5. Stop Loss (SL)
- **LONG**: SL at low of the 1st candle in the breakout
- **SHORT**: SL at high of the 1st candle in the breakout

### 6. Take Profit (TP)
- Risk-Reward Ratio = 2.1:1
- **LONG**: TP = Entry + (RR × Risk)
- **SHORT**: TP = Entry - (RR × Risk)
- Where Risk = |Entry - SL|

### 7. Trade Execution
- Enter immediately after the 3rd candle closes.
- Monitor on 15-minute chart for SL/TP hits.
- Exit on first SL or TP hit.

## Simple Diagram (Even a Child Can Understand)

Imagine you're playing a game where you follow arrows on a treasure map. Here's how the trading strategy works:

```
START: Look at the price chart

↓

Is the price ABOVE the smooth line (EMA50)?
   YES → LONG TREASURE HUNT
   NO  → SHORT TREASURE HUNT

↓ (For LONG)

Find 3 GREEN ARROWS (candles) in a row
Each arrow must be BIG enough (body ≥ average/3)
Time must be AFTER NOON (12:00)

↓

The last GREEN ARROW must go ABOVE the highest mountain peak (swing high)

↓

Check for MAGIC GAP:
If the top of 1st arrow < bottom of 3rd arrow → GAP FOUND!

↓

ENTER at the MIDDLE of the MAGIC GAP

↓

STOP LOSS: At the bottom of the 1st arrow
TAKE PROFIT: 2.1 times higher than entry

↓

Wait for price to hit STOP or PROFIT → EXIT

For SHORT: Same but with RED ARROWS, below valleys, etc.
```

**Visual Analogy:**
- Candles = Arrows pointing up (green) or down (red)
- EMA50 = Smooth path you follow
- Swing High/Low = Mountain peaks/valleys
- FVG = Magic gap in the path that needs filling
- Entry = Jump into the middle of the gap
- SL/TP = Safety net and treasure chest

## Example Trade Scenario
1. Price is above EMA50 (LONG trend)
2. See 3 green candles after 12:00, each with decent size
3. 3rd candle breaks above recent swing high
4. FVG: High of 1st candle = 1.0500, Low of 3rd candle = 1.0520
5. Entry: (1.0500 + 1.0520) / 2 = 1.0510
6. SL: Low of 1st candle = 1.0490
7. Risk: 1.0510 - 1.0490 = 0.0020
8. TP: 1.0510 + (2.1 × 0.0020) = 1.0552
9. Wait for price to reach 1.0490 (loss) or 1.0552 (win)

## Risk Management
- Fixed Risk-Reward ratio of 2.1:1
- SL based on recent swing structure
- Only trade after 12:00 to avoid news/events
- Position sizing should consider account risk (not implemented in code)

## Backtesting Results
Run the script to generate Excel file with trade results. Analyze win rate, average profit/loss, and drawdowns to evaluate strategy performance.

## Notes
- This is a simplified explanation. Always paper trade first.
- Market conditions change; monitor performance regularly.
- The code uses 15m data but simulates on same timeframe.
- Adjust parameters based on backtesting results.
