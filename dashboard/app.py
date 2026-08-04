from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VAL_PREDICTIONS_PATH = DATA_DIR / "val_predictions.parquet"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "feature_importance.parquet"

ACCENT = "#5B8DEF"
SECONDARY = "#8B5CF6"
POSITIVE = "#22C55E"
NEGATIVE = "#EF4444"


def _require(path):
    if not path.exists():
        st.error(f"No precomputed data found at {path}. Run `python src/train.py` first.")
        st.stop()


@st.cache_data
def load_val_predictions():
    _require(VAL_PREDICTIONS_PATH)
    return pd.read_parquet(VAL_PREDICTIONS_PATH)


@st.cache_data
def load_feature_importance():
    _require(FEATURE_IMPORTANCE_PATH)
    return pd.read_parquet(FEATURE_IMPORTANCE_PATH)


def rmspe(actual, predicted):
    """Root Mean Square Percentage Error, excluding rows where actual == 0."""
    actual = actual.to_numpy(dtype=float)
    predicted = predicted.to_numpy(dtype=float)
    mask = actual != 0
    pct_error = (actual[mask] - predicted[mask]) / actual[mask]
    return float(np.sqrt(np.mean(pct_error**2)))


def flagged_count(df, date, threshold):
    day = df[(df["Date"] == date) & (df["Open"] == 1)].copy()
    day["GapPct"] = (day["Sales"] - day["Predicted"]) / day["Predicted"] * 100
    return int((day["GapPct"] <= -threshold).sum())


st.set_page_config(page_title="Rossmann Sales Forecast", layout="wide")

val_fe = load_val_predictions()
feature_importance = load_feature_importance()
holdout_start, holdout_end = val_fe["Date"].min().date(), val_fe["Date"].max().date()

with st.sidebar:
    st.header("Controls")
    store_ids = sorted(val_fe["Store"].unique())
    selected_store = st.selectbox("Store", store_ids)
    threshold = st.slider("Underperforming threshold (%)", 5, 50, 15)
    st.caption(f"Validation holdout: {holdout_start} to {holdout_end}")

st.title("Rossmann Sales Forecast")
st.caption(f"Forecasted vs. actual sales over the validation holdout ({holdout_start} to {holdout_end}).")

# --- KPI row ---
latest_date = val_fe["Date"].max()
prior_week_date = latest_date - pd.Timedelta(days=7)
prior_day_date = latest_date - pd.Timedelta(days=1)

open_latest = val_fe[(val_fe["Date"] == latest_date) & (val_fe["Open"] == 1)]
open_prior_week = val_fe[(val_fe["Date"] == prior_week_date) & (val_fe["Open"] == 1)]

avg_sales_today = open_latest["Sales"].mean()
avg_sales_last_week = open_prior_week["Sales"].mean()
sales_pct_change = (avg_sales_today - avg_sales_last_week) / avg_sales_last_week * 100

accuracy_overall = 100 - rmspe(val_fe["Sales"], val_fe["Predicted"]) * 100
accuracy_today = 100 - rmspe(open_latest["Sales"], open_latest["Predicted"]) * 100
accuracy_delta = accuracy_today - accuracy_overall

flagged_today = flagged_count(val_fe, latest_date, threshold)
flagged_yesterday = flagged_count(val_fe, prior_day_date, threshold)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Avg. Daily Sales",
        f"{avg_sales_today:,.0f}",
        f"{sales_pct_change:+.1f}% vs last week",
        border=True,
        help=f"Average actual sales per open store on {latest_date.date()}, vs. the same weekday one week earlier.",
    )
with col2:
    st.metric(
        "Model Accuracy",
        f"{accuracy_overall:.1f}%",
        f"{accuracy_delta:+.1f} pts today vs. holdout avg",
        border=True,
        help="100% minus RMSPE (the Rossmann competition metric) over the full validation holdout, "
        "compared against accuracy on just the most recent day.",
    )
with col3:
    st.metric(
        "Underperforming Stores",
        f"{flagged_today}",
        f"{flagged_today - flagged_yesterday:+d} vs yesterday",
        delta_color="inverse",
        border=True,
        help=f"Open stores on {latest_date.date()} with actual sales at least {threshold}% below predicted.",
    )

# --- Tabs ---
tab_forecast, tab_importance, tab_underperforming = st.tabs(
    ["Forecast", "Feature Importance", "Underperforming Stores"]
)

with tab_forecast:
    store_data = val_fe[val_fe["Store"] == selected_store].sort_values("Date")

    st.subheader(f"Store {selected_store}: predicted vs. actual sales")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=store_data["Date"],
            y=store_data["Sales"],
            mode="lines",
            name="Actual",
            line=dict(color=ACCENT, width=2.5),
            hovertemplate="%{x|%b %d, %Y}<br>Actual: %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=store_data["Date"],
            y=store_data["Predicted"],
            mode="lines",
            name="Predicted",
            line=dict(color=SECONDARY, width=2.5, dash="dot"),
            hovertemplate="%{x|%b %d, %Y}<br>Predicted: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis=dict(title="Sales"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40, l=10, r=10, b=10),
    )
    st.plotly_chart(fig)

with tab_importance:
    st.subheader("What drives the forecast: feature importance")
    importance_pct = feature_importance.set_index("Feature")["Gain"]
    importance_pct = (importance_pct / importance_pct.sum() * 100).sort_values(ascending=True)

    fig2 = go.Figure(
        go.Bar(
            x=importance_pct.values,
            y=importance_pct.index,
            orientation="h",
            marker=dict(color=ACCENT),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig2.update_layout(
        xaxis=dict(title="Share of total gain (%)"),
        margin=dict(t=10, l=10, r=10, b=10),
    )
    st.plotly_chart(fig2)

with tab_underperforming:
    st.subheader("Underperforming stores")
    latest = val_fe[(val_fe["Date"] == latest_date) & (val_fe["Open"] == 1)].copy()
    latest["GapPct"] = (latest["Sales"] - latest["Predicted"]) / latest["Predicted"] * 100
    flagged = latest[latest["GapPct"] <= -threshold].sort_values("GapPct")

    st.caption(f"Stores on {latest_date.date()} where actual sales are at least {threshold}% below predicted.")
    st.dataframe(
        flagged[["Store", "Sales", "Predicted", "GapPct"]]
        .rename(columns={"Sales": "Actual"})
        .round({"Predicted": 0, "GapPct": 1})
        .reset_index(drop=True),
        width="stretch",
    )
