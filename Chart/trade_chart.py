import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import mplfinance as mpf
import matplotlib.pyplot as plt

def plot_trade_chart(df, entry_time, exit_time, entry_price, stop_loss, take_profit, fvg_zone, swing_high, swing_low, direction, trade_id):
    """
    Plot a candlestick chart for a specific trade with entry, stop-loss, take-profit, FVG, swing levels, and EMA50.
    Mark entry and SL with arrows.
    """
    df_trade = df[(df["Datetime"] >= entry_time) & (df["Datetime"] <= exit_time)].copy()
    if df_trade.empty:
        print(f"No data for trade {trade_id}")
        return

    df_trade.set_index("Datetime", inplace=True)

    # Ensure EMA50 is present
    if 'EMA50' not in df_trade.columns:
        df_trade['EMA50'] = df_trade['Close'].ewm(span=50, adjust=False).mean()

    # Custom style
    tradingview_style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',
            edge='#131722',
            wick={'up': '#26a69a', 'down': '#ef5350'},
            volume='in',
            ohlc='white'
        ),
        gridcolor="#131722",
        facecolor="#131722",
        figcolor="#131722",
        rc={
            'font.size': 14,
            'font.family': 'sans-serif',
            'axes.labelcolor': 'white',
            'axes.edgecolor': 'white',
            'axes.titlecolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white'
        }
    )

    # Add plots
    ap = [
        mpf.make_addplot(df_trade['EMA50'], color='blue', width=1),
        mpf.make_addplot([fvg_zone[0]] * len(df_trade), color="orange", alpha=0.1, width=4),
        mpf.make_addplot([fvg_zone[1]] * len(df_trade), color="orange", alpha=0.1, width=4),
        mpf.make_addplot([take_profit] * len(df_trade), color="#00e676", linestyle="-", width=1),
        mpf.make_addplot([swing_high] * len(df_trade), color="#ab47bc", linestyle=":", width=2),
        mpf.make_addplot([swing_low] * len(df_trade), color="#fbc02d", linestyle=":", width=2),
    ]

    title = f"Trade {trade_id} | {direction} | Entry: {entry_price} SL: {stop_loss} TP: {take_profit}"
    fig, axlist = mpf.plot(
        df_trade,
        type="candle",
        style=tradingview_style,
        addplot=ap,
        title=title,
        ylabel="Price",
        volume=False,
        returnfig=True,
        tight_layout=True,
        figratio=(16,9),
        figscale=2,
        update_width_config=dict(candle_linewidth=3, candle_width=0.8)
    )

    # Add arrows
    idx_entry = 0
    idx_exit = len(df_trade) - 1

    # Direction arrow at entry
    if direction == "LONG":
        axlist[0].annotate('', xy=(idx_entry, entry_price), xytext=(idx_entry, entry_price + 0.002), arrowprops=dict(arrowstyle='->', color='cyan'), fontsize=10)
    else:
        axlist[0].annotate('', xy=(idx_entry, entry_price), xytext=(idx_entry, entry_price - 0.002), arrowprops=dict(arrowstyle='->', color='magenta'), fontsize=10)

    # Entry arrow
    axlist[0].annotate('Entry', xy=(idx_entry, entry_price), xytext=(idx_entry + 5, entry_price + 0.001), arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')

    # SL arrow at exit
    axlist[0].annotate('SL', xy=(idx_exit, stop_loss), xytext=(idx_exit - 5, stop_loss - 0.001), arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')

    return fig

def plot_overall_chart(df_ohlc, fvg_zones, df_trades):
    """
    Plot overall candlestick chart with EMA50, FVG zones, and trade directions.
    """
    df = df_ohlc.copy()
    df.set_index("Datetime", inplace=True)

    # Ensure EMA50 is present
    if 'EMA50' not in df.columns:
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # Custom style
    tradingview_style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',
            edge='#131722',
            wick={'up': '#26a69a', 'down': '#ef5350'},
            volume='in',
            ohlc='white'
        ),
        gridcolor="#131722",
        facecolor="#131722",
        figcolor="#131722",
        rc={
            'font.size': 14,
            'font.family': 'sans-serif',
            'axes.labelcolor': 'white',
            'axes.edgecolor': 'white',
            'axes.titlecolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white'
        }
    )

    # Add plots
    ap = [
        mpf.make_addplot(df['EMA50'], color='blue', width=1),
    ]

    # Add FVG zones as horizontal lines
    for zone in fvg_zones:
        idx = zone["idx"]
        fvg_low, fvg_high = zone["fvg"]
        direction = zone["direction"]
        color = 'orange' if direction == "LONG" else 'purple'
        ap.append(mpf.make_addplot([fvg_low] * len(df), color=color, alpha=0.3, width=2, linestyle='--'))
        ap.append(mpf.make_addplot([fvg_high] * len(df), color=color, alpha=0.3, width=2, linestyle='--'))

    title = "Candlestick Chart with 50 EMA and FVG Zones"
    fig, axlist = mpf.plot(
        df,
        type="candle",
        style=tradingview_style,
        addplot=ap,
        title=title,
        ylabel="Price",
        volume=False,
        returnfig=True,
        tight_layout=True,
        figratio=(16,9),
        figscale=2,
        update_width_config=dict(candle_linewidth=3, candle_width=0.8)
    )

    # Add trade direction markers
    for _, trade in df_trades.iterrows():
        entry_time = trade["date_open"]
        direction = trade["direction"]
        entry_price = trade.get("entry_price", df.loc[df.index == entry_time, "Close"].iloc[0] if entry_time in df.index else None)
        if entry_price is None:
            continue
        idx_entry = df.index.get_loc(entry_time) if entry_time in df.index else None
        if idx_entry is None:
            continue
        if direction == "LONG":
            axlist[0].annotate('', xy=(idx_entry, entry_price), xytext=(idx_entry, entry_price + 0.002), arrowprops=dict(arrowstyle='->', color='green'), fontsize=10)
        else:
            axlist[0].annotate('', xy=(idx_entry, entry_price), xytext=(idx_entry, entry_price - 0.002), arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)

    return fig

if __name__ == "__main__":
    # Create sample data of 20 candlesticks
    times = pd.date_range(start="2023-01-01 12:00", periods=20, freq="15min", tz="UTC")
    np.random.seed(42)
    base_price = 1.1000
    prices = []
    for i in range(len(times)):
        change = np.random.normal(0, 0.001)
        base_price += change
        o = base_price
        h = o + abs(np.random.normal(0, 0.0005))
        l = o - abs(np.random.normal(0, 0.0005))
        c = np.random.uniform(l, h)
        prices.append([o, h, l, c])
    df = pd.DataFrame(prices, columns=["Open", "High", "Low", "Close"])
    df["Datetime"] = times
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # Mock trade parameters
    entry_time = times[5]
    exit_time = times[15]
    entry_price = 1.105
    stop_loss = 1.100
    take_profit = 1.115
    fvg_zone = (1.102, 1.108)
    swing_high = 1.110
    swing_low = 1.100
    direction = "LONG"
    trade_id = 1

    plot_trade_chart(df, entry_time, exit_time, entry_price, stop_loss, take_profit, fvg_zone, swing_high, swing_low, direction, trade_id)
