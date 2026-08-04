from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VAL_PREDICTIONS_PATH = DATA_DIR / "val_predictions.parquet"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "feature_importance.parquet"

BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "text.color": "#0b0b0b",
        "xtick.color": "#898781",
        "ytick.color": "#898781",
        "grid.color": "#e1e0d9",
        "font.family": "sans-serif",
    }
)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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


st.set_page_config(page_title="Rossmann Sales Forecast", layout="wide")
st.title("Rossmann Sales Forecast")

val_fe = load_val_predictions()
feature_importance = load_feature_importance()
holdout_start, holdout_end = val_fe["Date"].min().date(), val_fe["Date"].max().date()
st.caption(f"Forecasted vs. actual sales over the validation holdout ({holdout_start} to {holdout_end}).")

store_ids = sorted(val_fe["Store"].unique())
selected_store = st.selectbox("Store", store_ids)

store_data = val_fe[val_fe["Store"] == selected_store].sort_values("Date")

st.subheader(f"Store {selected_store}: predicted vs. actual sales")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(store_data["Date"], store_data["Sales"], color=BLUE, linewidth=2, label="Actual")
ax.plot(store_data["Date"], store_data["Predicted"], color=ORANGE, linewidth=2, label="Predicted")
ax.set_ylabel("Sales")
ax.grid(axis="y", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
_despine(ax)
plt.tight_layout()
st.pyplot(fig)

st.subheader("What drives the forecast: feature importance")
importance_pct = feature_importance.set_index("Feature")["Gain"]
importance_pct = (importance_pct / importance_pct.sum() * 100).sort_values(ascending=True)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.barh(importance_pct.index, importance_pct.values, color=BLUE)
ax2.set_xlabel("Share of total gain (%)")
ax2.grid(axis="x", linewidth=0.8)
ax2.set_axisbelow(True)
_despine(ax2)
plt.tight_layout()
st.pyplot(fig2)

st.subheader("Underperforming stores")
threshold = st.slider("Flag stores at least this far below predicted sales (%)", 5, 50, 15)

latest_date = val_fe["Date"].max()
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
