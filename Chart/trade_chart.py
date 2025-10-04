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
    # Find indices
    entry_idx = df[df["Datetime"] == entry_time].index
    if entry_idx.empty:
        print(f"Entry time not found for trade {trade_id}")
        return
    entry_idx = entry_idx[0]

    exit_idx = df[df["Datetime"] == exit_time].index
    if exit_idx.empty:
        print(f"Exit time not found for trade {trade_id}")
        return
    exit_idx = exit_idx[0]

    # Include 10 candles before entry and 10 after exit
    start_idx = max(0, entry_idx - 10)
    end_idx = min(len(df), exit_idx + 11)
    df_trade = df.iloc[start_idx:end_idx].copy()
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
        mpf.make_addplot(df_trade['EMA50'], color='blue', width=3),
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

    # Add FVG zone as shaded area
    axlist[0].fill_between(range(len(df_trade)), fvg_zone[0], fvg_zone[1], color='lightpink', alpha=0.3, label='FVG Zone')

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

def plot_position_tool(df, entry_price, stop_loss, take_profit, direction):
    """
    Plot a visual similar to TradingView's Long/Short Position tool overlaid on candlestick chart.
    Shows candlesticks, EMA50, entry line, TP zone (green shaded), SL zone (red shaded), and RRR.
    """
    entry_price = float(entry_price)
    stop_loss = float(stop_loss)
    take_profit = float(take_profit)
    direction = direction.upper()

    df_plot = df.copy()
    if 'EMA50' not in df_plot.columns:
        df_plot['EMA50'] = df_plot['Close'].ewm(span=50, adjust=False).mean()
    df_plot.set_index("Datetime", inplace=True)

    # Calculate RRR
    if direction == "LONG":
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:  # SHORT
        risk = stop_loss - entry_price
        reward = entry_price - take_profit

    if risk <= 0:
        print("Invalid SL: risk must be positive.")
        return

    rrr = reward / risk

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

    # Add plots for lines
    ap = [
        mpf.make_addplot(df_plot['EMA50'], color='blue', width=2),
        mpf.make_addplot([entry_price] * len(df_plot), color='black', linewidth=2),
        mpf.make_addplot([take_profit] * len(df_plot), color='green', linestyle='-', width=1),
        mpf.make_addplot([stop_loss] * len(df_plot), color='red', linestyle='-', width=1),
    ]

    title = f'{direction} Position - Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit}, RRR: {rrr:.2f}'
    fig, axlist = mpf.plot(
        df_plot,
        type="candle",
        style=tradingview_style,
        addplot=ap,
        title=title,
        ylabel="Price",
        volume=False,
        returnfig=True,
        tight_layout=True,
        figratio=(16,9),
        figscale=1.5,
        update_width_config=dict(candle_linewidth=2, candle_width=0.6)
    )

    # Add shaded zones post-plot
    x = range(len(df_plot))
    if direction == "LONG":
        # Green TP zone
        axlist[0].fill_between(x, entry_price, take_profit, color='green', alpha=0.2)
        # Red SL zone
        axlist[0].fill_between(x, stop_loss, entry_price, color='red', alpha=0.2)
    else:
        # Green TP zone
        axlist[0].fill_between(x, take_profit, entry_price, color='green', alpha=0.2)
        # Red SL zone
        axlist[0].fill_between(x, entry_price, stop_loss, color='red', alpha=0.2)

    plt.show()
    return fig

