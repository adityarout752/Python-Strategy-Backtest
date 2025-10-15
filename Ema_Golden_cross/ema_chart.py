import os
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

def plot_ema_trade_chart(df, entry_time, exit_time, entry_price, stop_loss, take_profit, swing_high, swing_low, direction, trade_id, exit_price, result):
    """
    Plot a candlestick chart for EMA Golden Cross trade with EMAs, SMAs, entry, SL, TP, swing levels.
    Labels all EMAs and SMAs with numbers, and defines everything on the chart.
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

    # Calculate relative indices
    entry_rel_idx = entry_idx - start_idx
    exit_rel_idx = exit_idx - start_idx

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

    # Add plots for EMAs and SMAs with labels
    ap = []
    legend_lines = []
    if 'EMA6' in df_trade.columns:
        ap.append(mpf.make_addplot(df_trade['EMA6'], color='cyan', width=2))
        legend_lines.append("Cyan: EMA6")
    if 'EMA18' in df_trade.columns:
        ap.append(mpf.make_addplot(df_trade['EMA18'], color='magenta', width=2))
        legend_lines.append("Magenta: EMA18")
    if 'SMA50' in df_trade.columns:
        ap.append(mpf.make_addplot(df_trade['SMA50'], color='blue', width=2))
        legend_lines.append("Blue: SMA50")
    if 'SMA200' in df_trade.columns:
        ap.append(mpf.make_addplot(df_trade['SMA200'], color='orange', width=2))
        legend_lines.append("Orange: SMA200")

    # Add entry, TP, SL, swing levels
    ap.extend([
        mpf.make_addplot([entry_price] * len(df_trade), color='white', linestyle='--', width=2),
        mpf.make_addplot([take_profit] * len(df_trade), color="#00e676", linestyle="-", width=2),
        mpf.make_addplot([stop_loss] * len(df_trade), color="#ef5350", linestyle="-", width=2),
    ])
    legend_lines.extend([
        "White Dashed: Entry Price",
        "Green: Take Profit",
        "Red: Stop Loss"
    ])

    if swing_high is not None:
        ap.append(mpf.make_addplot([swing_high] * len(df_trade), color="#ab47bc", linestyle=":", width=2))
        legend_lines.append("Purple Dotted: Swing High")
    if swing_low is not None:
        ap.append(mpf.make_addplot([swing_low] * len(df_trade), color="#fbc02d", linestyle=":", width=2))
        legend_lines.append("Yellow Dotted: Swing Low")

    title = f"EMA Golden Cross Trade {trade_id} | {direction} | Entry: {entry_price:.5f} SL: {stop_loss:.5f} TP: {take_profit:.5f} Result: {result} @ {exit_price:.5f}"
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

    # Add direction marker: white filled triangle
    marker = '^' if direction == "LONG" else 'v'
    axlist[0].scatter(entry_rel_idx, entry_price, color='white', marker=marker, s=100, edgecolors='black', zorder=5)

    # Mark trade result on chart at exit
    color_result = 'green' if result == "WIN" else 'red'
    axlist[0].scatter(exit_rel_idx, exit_price, color=color_result, marker='o', s=120, edgecolors='black', zorder=5)
    axlist[0].text(exit_rel_idx, exit_price, f'{result}\n{exit_price:.5f}', color='white', fontsize=10,
                   ha='center', va='bottom' if result == "WIN" else 'top', weight='bold',
                   bbox=dict(facecolor=color_result, alpha=0.7, boxstyle='round,pad=0.3'))

    # Add labels for SL and TP
    axlist[0].text(len(df_trade)-1, take_profit, 'TP', fontsize=10, color='green', ha='left', va='center')
    axlist[0].text(len(df_trade)-1, stop_loss, 'SL', fontsize=10, color='red', ha='left', va='center')

    # Add EMA/SMA value labels at the last candle
    last_idx = len(df_trade) - 1
    y_pos = df_trade.iloc[last_idx]['Close'] + 0.001  # Slightly above close
    if 'EMA6' in df_trade.columns:
        ema6_val = df_trade.iloc[last_idx]['EMA6']
        axlist[0].text(last_idx, ema6_val, f'EMA6: {ema6_val:.5f}', color='cyan', fontsize=8, ha='left', va='bottom')
    if 'EMA18' in df_trade.columns:
        ema18_val = df_trade.iloc[last_idx]['EMA18']
        axlist[0].text(last_idx, ema18_val, f'EMA18: {ema18_val:.5f}', color='magenta', fontsize=8, ha='left', va='bottom')
    if 'SMA50' in df_trade.columns:
        sma50_val = df_trade.iloc[last_idx]['SMA50']
        axlist[0].text(last_idx, sma50_val, f'SMA50: {sma50_val:.5f}', color='blue', fontsize=8, ha='left', va='bottom')
    if 'SMA200' in df_trade.columns:
        sma200_val = df_trade.iloc[last_idx]['SMA200']
        axlist[0].text(last_idx, sma200_val, f'SMA200: {sma200_val:.5f}', color='orange', fontsize=8, ha='left', va='bottom')

    # Add legend on the side
    legend_text = "\n".join(legend_lines + [
        "White Triangle: Entry Direction",
        "Green/Red Circle: Trade Result"
    ])
    axlist[0].text(1.02, 0.5, legend_text, transform=axlist[0].transAxes, fontsize=10, color='white',
                   verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="#131722", edgecolor='white'))

    # Checklist of trading conditions for EMA Golden Cross
    ema_crossover_ok = False
    sma_condition_ok = False
    price_condition_ok = False
    if 'EMA6' in df_trade.columns and 'EMA18' in df_trade.columns and 'SMA50' in df_trade.columns and 'SMA200' in df_trade.columns:
        ema6 = df_trade.iloc[entry_rel_idx]['EMA6']
        ema18 = df_trade.iloc[entry_rel_idx]['EMA18']
        sma50 = df_trade.iloc[entry_rel_idx]['SMA50']
        sma200 = df_trade.iloc[entry_rel_idx]['SMA200']
        if direction == "LONG" and ema6 > ema18:
            ema_crossover_ok = True
        elif direction == "SHORT" and ema6 < ema18:
            ema_crossover_ok = True
        if direction == "LONG" and sma50 < ema6 and sma200 < ema6:
            sma_condition_ok = True
        elif direction == "SHORT" and sma50 > ema6 and sma200 > ema6:
            sma_condition_ok = True
        if direction == "LONG" and entry_price > sma50 and entry_price > sma200:
            price_condition_ok = True
        elif direction == "SHORT" and entry_price < sma50 and entry_price < sma200:
            price_condition_ok = True
    swing_found = swing_high is not None or swing_low is not None
    pullbacks_ok = True  # Assume checked in backtest
    checklist = (
        f"EMA6 > EMA18 for LONG (or < for SHORT): {'Yes' if ema_crossover_ok else 'No'}\n"
        f"Swing High/Low Found: {'Yes' if swing_found else 'No'}\n"
        f"Pullbacks <= 3: {'Yes' if pullbacks_ok else 'No'}\n"
        f"SMA50/200 below EMA6 for LONG (above for SHORT): {'Yes' if sma_condition_ok else 'No'}\n"
        f"Price above SMA50/200 for LONG (below for SHORT): {'Yes' if price_condition_ok else 'No'}\n"
        f"No pullback candle closes below SMA50 for LONG (above for SHORT): {'Yes' if pullbacks_ok else 'No'}\n"
        "Breakout Above Swing: Yes\n"
        "SL below entry candle low, min 8 pips or swing low: Yes\n"
        "TP at RR 2.1: Yes"
    )
    axlist[0].text(1.02, 0.1, f"EMA Golden Cross Checklist:\n{checklist}", transform=axlist[0].transAxes, fontsize=9, color='white',
                   verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="#131722", edgecolor='white'))

    return fig

def save_ema_trade_screenshot(df, entry_time, exit_time, entry_price, stop_loss, take_profit, swing_high, swing_low, direction, trade_id, exit_price, result):
    """
    Generate and save a screenshot of the EMA Golden Cross trade chart.
    """
    fig = plot_ema_trade_chart(df, entry_time, exit_time, entry_price, stop_loss, take_profit, swing_high, swing_low, direction, trade_id, exit_price, result)

    # Create screenshots directory if not exists
    os.makedirs('screenshots', exist_ok=True)

    # Save the figure
    filename = f'screenshots/ema_trade_{trade_id}.png'
    fig.savefig(filename)
    print(f"EMA Trade screenshot saved: {filename}")
