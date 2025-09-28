import os
import sys
import os.path

# Add parent directory to sys.path for Chart import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Chart.trade_chart import plot_trade_chart

def save_trade_screenshot(df, entry_time, exit_time, entry_price, stop_loss, take_profit, fvg_zone, swing_high, swing_low, direction, trade_id):
    """
    Generate and save a screenshot of the trade chart using matplotlib.
    """
    fig = plot_trade_chart(df, entry_time, exit_time, entry_price, stop_loss, take_profit, fvg_zone, swing_high, swing_low, direction, trade_id)

    # Create screenshots directory if not exists
    os.makedirs('screenshots', exist_ok=True)

    # Save the figure
    filename = f'screenshots/trade_{trade_id}.png'
    fig.savefig(filename)
    print(f"Screenshot saved: {filename}")
