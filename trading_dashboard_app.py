# trading_dashboard_app.py
# Streamlit-based trading journal/dashboard similar to TradeZella
# - Works with your backtest output file: backtest_results_excel.xlsx
# - Or upload a CSV/Excel with the following columns (case-insensitive):
#   PAIR TRADED, DATE OPEN, DATE CLOSED, DAY OF WEEK, TIME, PIP RISKED, DIRECTION(LONG OR SHORT), RESULT, PIP OUTCOME
#
# Run:  streamlit run trading_dashboard_app.py

import os
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Trading Journal Dashboard",
    page_icon="📈",
    layout="wide",
)

# ----------------------------- UTILITIES -----------------------------
@st.cache_data(show_spinner=False)
def load_trades(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load trades from uploaded bytes or local file.
    Supports .xlsx, .xls, .csv
    """
    if file_bytes is None and filename and os.path.exists(filename):
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(filename)
        elif ext == ".csv":
            df = pd.read_csv(filename)
        else:
            raise ValueError("Unsupported file type. Use .xlsx, .xls, or .csv")
    else:
        # uploaded
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        elif filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported uploaded file type.")

    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename common variants
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
        "result": "result",
        "pip outcome": "pip_outcome",
        # Sometimes exported with spaces/cases
        "direction (long or short)": "direction",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"pair", "date_open", "date_closed", "pip_risked", "direction", "result", "pip_outcome"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}")

    # Parse datetimes: allow timezone-naive; treat as UTC then convert later if user wants
    df["date_open"] = pd.to_datetime(df["date_open"], errors="coerce", utc=True)
    df["date_closed"] = pd.to_datetime(df["date_closed"], errors="coerce", utc=True)
    df = df.dropna(subset=["date_open", "date_closed"]).copy()

    # Numbers
    for c in ["pip_risked", "pip_outcome"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["pip_risked", "pip_outcome"]).copy()

    # Clean strings
    if "direction" in df:
        df["direction"] = df["direction"].str.upper().str.strip()
    if "result" in df:
        df["result"] = df["result"].str.upper().str.strip()

    # Derived columns
    # Handle potential division by zero for R
    df["R_outcome"] = np.where(df["pip_risked"] > 0, df["pip_outcome"] / df["pip_risked"], np.nan)
    df["is_win"] = np.where(df["R_outcome"] > 0, 1, 0)

    return df.reset_index(drop=True)


def longest_streak(series: pd.Series, value: int) -> int:
    """Longest consecutive streak where series equals 'value' (1 for win, 0 for loss)."""
    best = cur = 0
    for v in series.astype(int).tolist():
        if v == value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


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
    return float(dd.min())  # negative number


# ----------------------------- SIDEBAR -----------------------------
st.sidebar.header("⚙️ Settings")

use_local_file = st.sidebar.checkbox("Use local file (backtest_results_excel.xlsx)", value=True)

uploaded = None
if use_local_file:
    default_path = st.sidebar.text_input("Local file path", value="backtest_results_excel.xlsx")
else:
    uploaded = st.sidebar.file_uploader("Upload trades (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
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
try:
    if use_local_file:
        df = load_trades(None, default_path)
    else:
        if uploaded is None:
            st.info("Upload a file or toggle 'Use local file'.")
            st.stop()
        df = load_trades(uploaded.getvalue(), uploaded.name)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Date filters
df = df.sort_values("date_open").reset_index(drop=True)
min_dt = df["date_open"].min().date()
max_dt = df["date_open"].max().date()

# Apply filter defaults if empty
if not filter_start:
    filter_start = min_dt
if not filter_end:
    filter_end = max_dt

mask = (df["date_open"].dt.date >= pd.to_datetime(filter_start).date()) & (df["date_open"].dt.date <= pd.to_datetime(filter_end).date())
Df = df.loc[mask].copy()

# Timezone display
if use_ist:
    Df["date_open_local"] = Df["date_open"].dt.tz_convert("Asia/Kolkata")
    Df["date_closed_local"] = Df["date_closed"].dt.tz_convert("Asia/Kolkata")
else:
    Df["date_open_local"] = Df["date_open"]
    Df["date_closed_local"] = Df["date_closed"]

Df["date"] = Df["date_open_local"].dt.date
Df["month"] = Df["date_open_local"].dt.to_period("M").astype(str)
Df["hour"] = Df["date_open_local"].dt.hour
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

if mode == "R multiple":
    equity = equity_from_r(Df, starting_balance, risk_pct)
else:
    equity = equity_from_pips(Df, starting_balance, pip_value)

mdd = max_drawdown(equity) if len(equity) else 0.0

# ----------------------------- HEADER -----------------------------
st.title("📊 Trading Journal Dashboard")
st.caption("Analyze win rate, streaks, sessions, equity curve, and more — TradeZella style.")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Trades", f"{TOTAL}")
k2.metric("Win Rate", f"{winrate:.1f}%")
k3.metric("Avg R", f"{avg_R:.2f}")
k4.metric("Longest Win Streak", f"{win_streak}")
k5.metric("Longest Lose Streak", f"{lose_streak}")
k6.metric("Max Drawdown", f"{mdd*100:.1f}%")

st.markdown("---")

# ----------------------------- CHARTS -----------------------------
# Equity curve
fig_e = go.Figure()
fig_e.add_trace(go.Scatter(x=Df["date_open_local"], y=equity, mode="lines", name="Equity"))
fig_e.update_layout(title="Equity Curve", xaxis_title="Time", yaxis_title="Balance")
st.plotly_chart(fig_e, use_container_width=True)

# Win rate & pips by month
mgrp = Df.groupby("month", sort=True).agg(
    trades=("R_outcome", "size"),
    win_rate=("is_win", lambda s: s.mean() * 100.0),
    avg_R=("R_outcome", "mean"),
    sum_pips=("pip_outcome", "sum"),
    avg_pips=("pip_outcome", "mean"),
).reset_index()

cc1, cc2 = st.columns(2)
with cc1:
    fig_wr = px.bar(mgrp, x="month", y="win_rate", title="Win Rate by Month (%)")
    st.plotly_chart(fig_wr, use_container_width=True)
with cc2:
    fig_p = px.bar(mgrp, x="month", y="sum_pips", title="Total Pips by Month")
    st.plotly_chart(fig_p, use_container_width=True)

# Best time to trade (hourly performance)
hgrp = Df.groupby("hour").agg(
    trades=("R_outcome", "size"),
    win_rate=("is_win", lambda s: s.mean() * 100.0),
    avg_R=("R_outcome", "mean"),
    sum_pips=("pip_outcome", "sum"),
).reset_index().sort_values("hour")

cc3, cc4 = st.columns(2)
with cc3:
    fig_hr = px.bar(hgrp, x="hour", y="avg_R", title="Avg R by Hour (Best Session)")
    st.plotly_chart(fig_hr, use_container_width=True)
with cc4:
    fig_hw = px.bar(hgrp, x="hour", y="win_rate", title="Win Rate by Hour (%)")
    st.plotly_chart(fig_hw, use_container_width=True)

# Day x Hour heatmap of Avg R
pivot = Df.pivot_table(index="weekday", columns="hour", values="R_outcome", aggfunc="mean")
fig_heat = px.imshow(pivot, aspect="auto", title="Performance Heatmap (Avg R)")
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ----------------------------- TABLE & EXPORT -----------------------------
with st.expander("Trades (filtered)", expanded=True):
    show_cols = [
        "pair", "date_open_local", "date_closed_local", "direction", "result",
        "pip_risked", "pip_outcome", "R_outcome", "hour", "weekday"
    ]
    show_cols = [c for c in show_cols if c in Df.columns]
    st.dataframe(Df[show_cols], use_container_width=True)

# Export summary
summary = {
    "total_trades": TOTAL,
    "win_rate": winrate,
    "avg_R": avg_R,
    "longest_win_streak": win_streak,
    "longest_lose_streak": lose_streak,
    "max_drawdown_pct": mdd * 100.0,
}

b = io.BytesIO()
with pd.ExcelWriter(b, engine="xlsxwriter") as writer:
    pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
    mgrp.to_excel(writer, sheet_name="ByMonth", index=False)
    hgrp.to_excel(writer, sheet_name="ByHour", index=False)
    Df.to_excel(writer, sheet_name="Trades", index=False)

st.download_button(
    label="📥 Download Summary Workbook",
    data=b.getvalue(),
    file_name="trading_dashboard_summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption("Built with ❤️ using Streamlit & Plotly. Inspired by TradeZella-style metrics.")
