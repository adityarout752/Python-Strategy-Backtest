import os
import pandas as pd
import mplfinance as mpf
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from datetime import datetime, timedelta

def plot_trade_chart(df, entry_price, stop_loss, take_profit, fvg_zone, swing_high, swing_low, trade_date, direction, trade_id):
    """
    Plot a TradingView-style candlestick chart for a specific trade with entry, stop-loss, take-profit, FVG, and swing levels.
    Save the chart as an image.
    """
    df_trade = df[df["Datetime"].dt.date == trade_date].copy()
    if df_trade.empty:
        print(f"No data available for trade date: {trade_date}")
        return None

    df_trade.set_index("Datetime", inplace=True)
    df_trade = df_trade[["Open", "High", "Low", "Close"]]

    # Custom TradingView-like style (no grid, thick candles/wicks, dark background)
    tradingview_style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',
            edge='#26a69a',  # candle border color (up and down)
            wick='white',
            volume='in',
            ohlc='white'
        ),
        gridcolor="#131722",  # same as background, so grid is invisible
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

    # Add FVG zone as a shaded region
    fvg_addplot = mpf.make_addplot([fvg_zone[0]] * len(df_trade), color="orange", alpha=0.3, linewidth=4)
    fvg_addplot_low = mpf.make_addplot([fvg_zone[1]] * len(df_trade), color="orange", alpha=0.3, linewidth=4)

    # Add entry, stop-loss, take-profit, swing high, and swing low levels
    ap = [
        mpf.make_addplot([entry_price] * len(df_trade), color="#00bfff", linestyle="-", linewidth=1),
        mpf.make_addplot([stop_loss] * len(df_trade), color="#ff1744", linestyle="-", linewidth=1),
        mpf.make_addplot([take_profit] * len(df_trade), color="#00e676", linestyle="-", linewidth=1),
        mpf.make_addplot([swing_high] * len(df_trade), color="#ab47bc", linestyle=":", linewidth=2),
        mpf.make_addplot([swing_low] * len(df_trade), color="#fbc02d", linestyle=":", linewidth=2),
        fvg_addplot,
        fvg_addplot_low,
    ]

    title = f"Trade {trade_id} | {direction} | Entry: {entry_price} SL: {stop_loss} TP: {take_profit}"
    chart_filename = f"trade_{trade_id}.png"
    mpf.plot(
        df_trade,
        type="candle",
        style=tradingview_style,
        addplot=ap,
        title=title,
        ylabel="Price",
        volume=False,
        savefig=chart_filename,
        tight_layout=True,
        figratio=(16,9),
        figscale=2,
        update_width_config=dict(candle_linewidth=3, candle_width=0.8, wick_linewidth=2)
    )
    print(f"Chart saved as {chart_filename}")
    return chart_filename

def save_trade_to_excel(trades, filename="backtest_results.xlsx"):
    """
    Save trade details to an Excel file and embed candlestick chart images.
    """
    trades_df = pd.DataFrame(trades)
    trades_df.to_excel(filename, index=False, engine="openpyxl")

    wb = load_workbook(filename)
    ws = wb.active

    for idx, trade in trades_df.iterrows():
        chart_filename = trade["Chart"]
        if os.path.exists(chart_filename):
            img = Image(chart_filename)
            img.width, img.height = 300, 200
            ws.add_image(img, f"H{idx + 2}")

    wb.save(filename)
    print(f"Trades saved to {filename} with charts embedded.")

if __name__ == "__main__":
    # Create mock OHLC data for one day with a visible uptrend and swings
    times = [datetime(2025, 1, 1, 9, 0) + timedelta(minutes=15*i) for i in range(20)]
    open_prices  = [1.100, 1.105, 1.110, 1.115, 1.120, 1.125, 1.130, 1.128, 1.126, 1.124, 1.122, 1.120, 1.123, 1.126, 1.129, 1.132, 1.135, 1.138, 1.140, 1.142]
    close_prices = [1.105, 1.110, 1.115, 1.120, 1.125, 1.130, 1.128, 1.126, 1.124, 1.122, 1.120, 1.123, 1.126, 1.129, 1.132, 1.135, 1.138, 1.140, 1.142, 1.145]
    high_prices  = [max(o, c)+0.003 for o, c in zip(open_prices, close_prices)]
    low_prices   = [min(o, c)-0.003 for o, c in zip(open_prices, close_prices)]

    data = {
        "Datetime": times,
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": close_prices,
    }
    df = pd.DataFrame(data)

    # Mock trade parameters (these will be visible on the chart)
    entry_price = 1.126
    stop_loss = 1.120
    take_profit = 1.135
    fvg_zone = (1.123, 1.129)
    swing_high = 1.132
    swing_low = 1.120
    trade_date = datetime(2025, 1, 1).date()
    direction = "LONG"
    trade_id = 1

    plot_trade_chart(
        df, entry_price, stop_loss, take_profit, fvg_zone, swing_high, swing_low, trade_date, direction, trade_id
    )