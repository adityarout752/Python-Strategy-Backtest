# trading_dashboard_app.py
# Streamlit-based trading journal/dashboard similar to TradeZella
# - Works with your backtest output file: backtest_results_excel.xlsx
# - Or upload a CSV/Excel with the following columns (case-insensitive):
#   PAIR TRADED, DATE OPEN, DATE CLOSED, DAY OF WEEK, TIME, PIP RISKED,
#   DIRECTION(LONG OR SHORT), RESULT, PIP OUTCOME
#
# Run:  streamlit run trading_dashboard_app.py

import os
import sys
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import calendar
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Chart.trade_chart import plot_trade_chart

st.set_page_config(
    page_title="Trading Journal Dashboard",
    page_icon="📈",
    layout="wide",
)

# ----------------------------- UTILITIES -----------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load trades and OHLC data from uploaded bytes or local file."""
    df_trades = pd.DataFrame()
    df_ohlc = pd.DataFrame()

    if file_bytes is None and filename and os.path.exists(filename):
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".xlsx", ".xls"]:
            # Load trades from first sheet
            df_trades = pd.read_excel(filename, sheet_name=0)
            # Try to load OHLC from second sheet if it exists
            try:
                df_ohlc = pd.read_excel(filename, sheet_name=1)
            except:
                pass
        elif ext == ".csv":
            df_trades = pd.read_csv(filename)
        else:
            raise ValueError("Unsupported file type. Use .xlsx, .xls, or .csv")
    else:
        if filename.lower().endswith((".xlsx", ".xls")):
            # Load trades from first sheet
            df_trades = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
            # Try to load OHLC from second sheet if it exists
            try:
                df_ohlc = pd.read_excel(io.BytesIO(file_bytes), sheet_name=1)
            except:
                pass
        elif filename.lower().endswith(".csv"):
            df_trades = pd.read_csv(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported uploaded file type.")

    # Process trades data
    if not df_trades.empty:
        # Normalize columns
        df_trades.columns = [c.strip().lower() for c in df_trades.columns]

        rename_map = {
            "pair traded": "pair",
            "pair": "pair",
            "date open": "date_open",
            "open date": "date_open",
            "date closed": "date_closed",
            "close date": "date_closed",
            "day of week": "day_of_week",
            "time": "time_open",
            "pip risked": "pip_risked",
            "direction(long or short)": "direction",
            "direction (long or short)": "direction",
            "result": "result",
            "pip outcome": "pip_outcome",
            "entry time": "entry_time",
            "exit time": "exit_time",
            "entry": "entry_price",
            "sl": "stop_loss",
            "tp": "take_profit",
            "fvg": "fvg_zone",
            "swing high": "swing_high",
            "swing low": "swing_low",
            "trend": "direction",
            "chart": "chart",
        }
        df_trades = df_trades.rename(columns={k: v for k, v in rename_map.items() if k in df_trades.columns})

        required = {"pair", "date_open", "date_closed", "pip_risked", "direction", "result", "pip_outcome"}
        missing = required - set(df_trades.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(list(missing))}")

        # Parse datetimes
        df_trades["date_open"] = pd.to_datetime(df_trades["date_open"], errors="coerce", utc=True)
        df_trades["date_closed"] = pd.to_datetime(df_trades["date_closed"], errors="coerce", utc=True)
        df_trades = df_trades.dropna(subset=["date_open", "date_closed"]).copy()

        # Numbers
        for c in ["pip_risked", "pip_outcome"]:
            df_trades[c] = pd.to_numeric(df_trades[c], errors="coerce")
        df_trades = df_trades.dropna(subset=["pip_risked", "pip_outcome"]).copy()

        # Clean strings
        if "direction" in df_trades:
            df_trades["direction"] = df_trades["direction"].str.upper().str.strip()
        if "result" in df_trades:
            df_trades["result"] = df_trades["result"].str.upper().str.strip()

        # Derived columns
        df_trades["R_outcome"] = np.where(df_trades["pip_risked"] > 0, df_trades["pip_outcome"] / df_trades["pip_risked"], np.nan)
        df_trades["is_win"] = np.where(df_trades["R_outcome"] > 0, 1, 0)

    # Process OHLC data
    if not df_ohlc.empty:
        df_ohlc.columns = [c.strip().lower() for c in df_ohlc.columns]
        if "datetime" in df_ohlc.columns:
            df_ohlc["Datetime"] = pd.to_datetime(df_ohlc["datetime"], errors="coerce", utc=True)
        elif "date" in df_ohlc.columns:
            df_ohlc["Datetime"] = pd.to_datetime(df_ohlc["date"], errors="coerce", utc=True)

    return df_trades.reset_index(drop=True), df_ohlc


def longest_streak(series: pd.Series, value: int) -> int:
    best = cur = 0
    for v in series.astype(int).tolist():
        if v == value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def get_streaks(series: pd.Series, value: int) -> list[int]:
    """Return list of streak lengths for the given value."""
    streaks = []
    cur = 0
    for v in series.astype(int).tolist():
        if v == value:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    return streaks


def compute_monthly_streaks_gt3(df: pd.DataFrame) -> pd.DataFrame:
    """Compute number of win/lose streaks >3 per month."""
    monthly_data = []
    for month, group in df.groupby("month"):
        win_streaks = get_streaks(group["is_win"], 1)
        lose_streaks = get_streaks(group["is_win"], 0)
        num_win_gt3 = sum(1 for s in win_streaks if s > 3)
        num_lose_gt3 = sum(1 for s in lose_streaks if s > 3)
        monthly_data.append({
            "month": month,
            "win_streaks_gt3": num_win_gt3,
            "lose_streaks_gt3": num_lose_gt3
        })
    return pd.DataFrame(monthly_data)


def equity_from_r(df: pd.DataFrame, starting_balance: float, risk_pct: float) -> pd.Series:
    risk_amount = starting_balance * (risk_pct / 100.0)
    pnl = df["R_outcome"].fillna(0.0) * risk_amount
    equity = starting_balance + pnl.cumsum()
    return equity


def equity_from_pips(df: pd.DataFrame, starting_balance: float, pip_value: float) -> pd.Series:
    pnl = df["pip_outcome"].fillna(0.0) * pip_value
    equity = starting_balance + pnl.cumsum()
    return equity


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min()) if len(dd) else 0.0


def build_calendar_view(df: pd.DataFrame):
    """Render calendar view showing P&L, trades, wins/losses."""
    if df.empty:
        st.info("No trades in this period.")
        return

    df_daily = df.groupby("date").agg(
        pnl=("pip_outcome", "sum"),
        trades=("R_outcome", "size"),
        winners=("is_win", "sum"),
    )
    df_daily["losers"] = df_daily["trades"] - df_daily["winners"]

    months = sorted(set(zip(df["date"].dt.year, df["date"].dt.month)))

    for year, month in months:
        st.subheader(f"📅 {calendar.month_name[month]} {year}")
        cal = calendar.monthcalendar(year, month)

        table = []
        for week in cal:
            row = []
            for day in week:
                if day == 0:
                    row.append("")
                else:
                    d = pd.Timestamp(year=year, month=month, day=day)
                    if d in df_daily.index:
                        r = df_daily.loc[d]
                        color = "#2ECC71" if r["pnl"] >= 0 else "#E74C3C"
                        text = (
                            f"**{day}**\n"
                            f"PnL: {r['pnl']:.1f}\n"
                            f"Trades: {r['trades']} | W:{int(r['winners'])}/L:{int(r['losers'])}"
                        )
                        row.append(f"<div style='background:{color};padding:6px;border-radius:8px'>{text}</div>")
                    else:
                        row.append(str(day))
            table.append(row)

        st.markdown(
            f"<table style='width:100%;text-align:center'>{''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in table)}</table>",
            unsafe_allow_html=True,
        )


# ----------------------------- SIDEBAR -----------------------------
st.sidebar.header("⚙️ Settings")

use_local_file = st.sidebar.checkbox("Use local file (backtest_results_excel.xlsx)", value=True)

uploaded = None
if use_local_file:
    default_path = st.sidebar.text_input("Local file path", value="backtest_results_excel.xlsx")
else:
    uploaded = st.sidebar.file_uploader("Upload trades with OHLC data (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
    default_path = None

use_ist = st.sidebar.checkbox("Display times in IST (UTC+5:30)", value=True)

col1, col2 = st.sidebar.columns(2)
with col1:
    starting_balance = st.number_input("Starting Balance", min_value=0.0, value=10000.0, step=100.0)
with col2:
    risk_pct = st.number_input("Risk % per trade (for R mode)", min_value=0.0, value=1.0, step=0.25)

mode = st.sidebar.radio("Equity Mode", ["R multiple", "Pips"], index=0)
pip_value = st.sidebar.number_input("Pip value (Pips mode)", min_value=0.0, value=1.0, step=0.5)

st.sidebar.markdown("---")
filter_start = st.sidebar.date_input("Filter start date")
filter_end = st.sidebar.date_input("Filter end date")

# ----------------------------- LOAD DATA -----------------------------
df = pd.DataFrame()
df_ohlc = pd.DataFrame()
if use_local_file:
    try:
        df, df_ohlc = load_data(None, default_path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
        raise SystemExit()
else:
    if uploaded is None:
        st.info("Upload a file or toggle 'Use local file'.")
        st.stop()
        raise SystemExit()
    try:
        df, df_ohlc = load_data(uploaded.getvalue(), uploaded.name)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
        raise SystemExit()

if df.empty:
    st.error("No trades data loaded.")
    st.stop()
    raise SystemExit()

# Date filters
df = df.sort_values("date_open").reset_index(drop=True)
min_dt = df["date_open"].min().date()
max_dt = df["date_open"].max().date()

if not filter_start:
    filter_start = min_dt
if not filter_end:
    filter_end = max_dt

mask = (df["date_open"].dt.date >= pd.to_datetime(filter_start).date()) & (df["date_open"].dt.date <= pd.to_datetime(filter_end).date())
Df = df.loc[mask].copy()

# Timezone & derived cols
if use_ist:
    Df["date_open_local"] = Df["date_open"].dt.tz_convert("Asia/Kolkata")
    Df["date_closed_local"] = Df["date_closed"].dt.tz_convert("Asia/Kolkata")
else:
    Df["date_open_local"] = Df["date_open"]
    Df["date_closed_local"] = Df["date_closed"]

# ✅ FIX: keep datetime64 for .dt access
Df["date"] = pd.to_datetime(Df["date_open_local"].dt.date)
Df["month"] = Df["date_open_local"].dt.to_period("M").astype(str)
Df["week"] = Df["date_open_local"].dt.to_period("W").astype(str)
Df["hour"] = Df["date_open_local"].dt.hour
Df["hour_utc"] = Df["date_open"].dt.hour
Df["weekday"] = Df["date_open_local"].dt.day_name()

# ----------------------------- KPIs -----------------------------
TOTAL = int(len(Df))
WINS = int(Df["is_win"].sum())
LOSSES = int(TOTAL - WINS)
winrate = (WINS / TOTAL * 100.0) if TOTAL else 0.0
avg_R = float(Df["R_outcome"].mean()) if TOTAL else 0.0
exp_R = float(Df["R_outcome"].mean())

win_streak = longest_streak(Df["is_win"], 1)
lose_streak = longest_streak(1 - Df["is_win"], 1)

# Overall streak counts
num_win_streaks = len(get_streaks(Df["is_win"], 1))
num_lose_streaks = len(get_streaks(Df["is_win"], 0))

# Overall streaks >3 counts
win_streaks_gt3 = sum(1 for s in get_streaks(Df["is_win"], 1) if s > 3)
lose_streaks_gt3 = sum(1 for s in get_streaks(Df["is_win"], 0) if s > 3)

if mode == "R multiple":
    equity = equity_from_r(Df, starting_balance, risk_pct)
else:
    equity = equity_from_pips(Df, starting_balance, pip_value)

mdd = max_drawdown(equity)

# Profit factor
gross_win = Df.loc[Df["R_outcome"] > 0, "R_outcome"].sum()
gross_loss = abs(Df.loc[Df["R_outcome"] < 0, "R_outcome"].sum())
profit_factor = (gross_win / gross_loss) if gross_loss else np.nan

# Monthly streaks >3
monthly_streaks_df = compute_monthly_streaks_gt3(Df)

# ----------------------------- HEADER -----------------------------
st.title("📊 Trading Journal Dashboard")
st.caption("Analyze performance — inspired by TradeZella style.")

k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12 = st.columns(12)
k1.metric("Total Trades", f"{TOTAL}")
k2.metric("Win Rate", f"{winrate:.1f}%")

# Calculate Risk:Reward ratio as average winning R_outcome / average losing R_outcome absolute value
avg_win_r = Df.loc[Df["R_outcome"] > 0, "R_outcome"].mean()
avg_loss_r = abs(Df.loc[Df["R_outcome"] < 0, "R_outcome"].mean())
risk_reward_ratio = avg_win_r / avg_loss_r if avg_loss_r != 0 else float('nan')
k3.metric("Risk:Reward Ratio", f"{risk_reward_ratio:.2f}" if not math.isnan(risk_reward_ratio) else "N/A")

k4.metric("Avg R", f"{avg_R:.2f}")
k5.metric("Longest Win Streak", f"{win_streak}")
k6.metric("Longest Lose Streak", f"{lose_streak}")
k7.metric("Max Drawdown", f"{mdd*100:.1f}%")
k8.metric("Profit Factor", f"{profit_factor:.2f}" if not math.isnan(profit_factor) else "N/A")
k9.metric("No. of Win Streaks", f"{num_win_streaks}")
k10.metric("No. of Lose Streaks", f"{num_lose_streaks}")
k11.metric("Win Streaks >3", f"{win_streaks_gt3}")
k12.metric("Lose Streaks >3", f"{lose_streaks_gt3}")

st.markdown("---")

# ----------------------------- CHARTS -----------------------------
# Equity curve
fig_e = go.Figure()
fig_e.add_trace(go.Scatter(x=Df["date_open_local"], y=equity, mode="lines", name="Equity"))
fig_e.update_layout(title="Equity Curve", xaxis_title="Time", yaxis_title="Balance")
st.plotly_chart(fig_e, use_container_width=True)

# Monthly summary
mgrp = Df.groupby("month").agg(
    trades=("R_outcome", "size"),
    win_rate=("is_win", lambda s: s.mean() * 100.0),
    avg_R=("R_outcome", "mean"),
    sum_pips=("pip_outcome", "sum"),
).reset_index()

st.subheader("📅 Monthly Summary")
st.dataframe(mgrp, use_container_width=True)

# Weekly summary
wgrp = Df.groupby("week").agg(
    trades=("R_outcome", "size"),
    win_rate=("is_win", lambda s: s.mean() * 100.0),
    avg_R=("R_outcome", "mean"),
    sum_pips=("pip_outcome", "sum"),
).reset_index()

st.subheader("📆 Weekly Summary")
st.dataframe(wgrp, use_container_width=True)

# Best time to trade
hgrp = Df.groupby("hour_utc").agg(
    trades=("R_outcome", "size"),
    win_rate=("is_win", lambda s: s.mean() * 100.0),
    avg_R=("R_outcome", "mean"),
    sum_pips=("pip_outcome", "sum"),
).reset_index().sort_values("hour_utc")
hgrp["time_window"] = hgrp["hour_utc"].apply(lambda h: f"{h:02d}:00-{(h+1)%24:02d}:00 UTC")

cc1, cc2 = st.columns(2)
with cc1:
    fig_hr = px.bar(hgrp, x="hour_utc", y="avg_R", title="Avg R by Hour (UTC)")
    st.plotly_chart(fig_hr, use_container_width=True)
with cc2:
    fig_hw = px.bar(hgrp, x="hour_utc", y="win_rate", title="Win Rate by Hour (UTC)")
    st.plotly_chart(fig_hw, use_container_width=True)

# ----------------------------- HOURLY PERFORMANCE DETAILS -----------------------------
st.markdown("---")
st.subheader("⏰ Hourly Performance Details (UTC)")
st.dataframe(hgrp[["time_window", "trades", "win_rate"]].rename(columns={"time_window": "Time Window (UTC)", "win_rate": "Win Rate (%)", "trades": "Number of Trades"}))

# ----------------------------- CALENDAR VIEW -----------------------------
st.markdown("---")
st.subheader("📅 Calendar View")
build_calendar_view(Df)

# ----------------------------- TABLE & EXPORT -----------------------------
with st.expander("Trades (filtered)", expanded=False):
    show_cols = [
        "pair", "date_open_local", "date_closed_local", "direction", "result",
        "pip_risked", "pip_outcome", "R_outcome", "hour", "weekday"
    ]
    show_cols = [c for c in show_cols if c in Df.columns]
    st.dataframe(Df[show_cols], use_container_width=True)

summary = {
    "total_trades": TOTAL,
    "win_rate": winrate,
    "avg_R": avg_R,
    "longest_win_streak": win_streak,
    "longest_lose_streak": lose_streak,
    "num_win_streaks": num_win_streaks,
    "num_lose_streaks": num_lose_streaks,
    "max_drawdown_pct": mdd * 100.0,
    "profit_factor": profit_factor,
}

b = io.BytesIO()
Df_export = Df.copy()
datetime_cols = ["date_open", "date_closed", "date_open_local", "date_closed_local", "date"]
for col in datetime_cols:
    if col in Df_export.columns and pd.api.types.is_datetime64tz_dtype(Df_export[col]):
        Df_export[col] = Df_export[col].dt.tz_localize(None)

with pd.ExcelWriter(b, engine="xlsxwriter") as writer:
    pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
    mgrp.to_excel(writer, sheet_name="ByMonth", index=False)
    monthly_streaks_df.to_excel(writer, sheet_name="StreaksByMonth", index=False)
    wgrp.to_excel(writer, sheet_name="ByWeek", index=False)
    hgrp.to_excel(writer, sheet_name="ByHour", index=False)
    Df_export.to_excel(writer, sheet_name="Trades", index=False)

st.download_button(
    label="📥 Download Summary Workbook",
    data=b.getvalue(),
    file_name="trading_dashboard_summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("---")

st.subheader("📈 Trade Charts")

if not df_ohlc.empty:
    for idx, trade in Df.iterrows():
        trade_id = idx + 1
        with st.expander(f"Trade {trade_id} - {trade['direction']} - {'Win' if trade['is_win'] else 'Loss'}", expanded=False):
            fig = plot_trade_chart(df_ohlc, trade['entry_time'], trade['exit_time'], trade['entry_price'], trade['stop_loss'], trade['take_profit'], trade['fvg_zone'], trade['swing_high'], trade['swing_low'], trade['direction'], trade_id)
            st.pyplot(fig)
            st.write(f"**Pair:** {trade['pair']}")
            st.write(f"**Open Date:** {trade['date_open_local']}")
            st.write(f"**Close Date:** {trade['date_closed_local']}")
            st.write(f"**Direction:** {trade['direction']}")
            st.write(f"**Result:** {trade['result']}")
            st.write(f"**Pip Risked:** {trade['pip_risked']}")
            st.write(f"**Pip Outcome:** {trade['pip_outcome']}")
            st.write(f"**R Outcome:** {trade['R_outcome']:.2f}")
else:
    st.info("No OHLC data found in the uploaded file. Ensure your Excel file has a second sheet with OHLC data.")

st.caption("Built with ❤️ using Streamlit & Plotly. Inspired by TradeZella-style metrics.")
