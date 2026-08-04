# Rossmann Store Sales Forecasting & Dashboard

## The Business Problem
Rossmann operates over 3,000 drugstores across 7 European countries. Store managers currently forecast their own daily sales up to six weeks out, with accuracy varying widely from manager to manager. A reliable, data-driven forecast lets staffing and inventory decisions be based on real patterns instead of guesswork.

**This project builds a machine learning model that predicts daily sales for 1,115 stores, and an interactive dashboard so a regional manager can see forecasted vs. actual sales and flag underperforming stores.**

## Data Source
- [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) — Kaggle competition dataset
- ~1M daily sales records across 1,115 stores (Jan 2013 – Jul 2015), plus store-level metadata (store type, assortment, competition distance, promo intervals, state/school holidays)

## Approach

**1. Data cleaning** (`src/data_prep.py`)
- `store.csv` nulls that are structural, not missing: stores with `Promo2 == 0` never opted into the continuity promo, so `Promo2SinceWeek`/`Promo2SinceYear`/`PromoInterval` are filled with `0`/`0`/`""` rather than treated as unknown.
- `CompetitionDistance` (3 stores with no competition data at all): imputed with the median distance (~2,325m), flagged via `CompetitionDistanceUnknown`.
- `CompetitionOpenSinceMonth`/`Year` (354 stores, 351 of which have a known `CompetitionDistance` but an unrecorded open date): left null rather than fabricating a date, flagged via `CompetitionOpenUnknown` — the "months since competition opened" figure is computed at feature-engineering time and stays null wherever the open date itself is unknown.
- `test.csv`'s 11 missing `Open` values, which all belong to a single store (Store 622), filled with `1`.

**2. Feature engineering** (`src/data_prep.py`)
- Calendar parts (`Year`/`Month`/`Day`/`DayOfWeek`) derived from `Date`.
- `UnusualOpenDay`: flags a store open on a day of week it's historically closed on (open rate below 50% for that store/day-of-week, fit on training data only).
- `IsStateHoliday` and `Promo2Active` (whether the continuity promo is actually running on a given date — accounting for the dataset spelling September as `"Sept"`, not `"Sep"`, in `PromoInterval`).
- `CompetitionOpenMonths`: months since a nearby competitor opened, as of each row's date.

**3. Validation** — a time-based split, no shuffling: the most recent 6 weeks held out as validation, mirroring the ~6.5-week horizon `test.csv` actually asks for.

**4. Modeling** (`src/model.py`, `src/train.py`)
- A naive baseline: each store's historical average sales for that day of week.
- LightGBM trained on the engineered features (excluding `Customers`, which isn't available in `test.csv` and would leak future information), early-stopped directly against validation RMSPE.

**5. Dashboard** (`dashboard/app.py`) — reads small precomputed prediction/importance files rather than the raw CSVs or the model itself at runtime, so it works in a deployed environment where the ~40MB training data isn't present.

## Results

- **Data quality**: closed stores reliably report zero sales (172,817 of 172,871 zero-sales rows). The remaining 54 rows are open-but-zero-sales anomalies spread across 41 stores with no shared pattern — likely data-entry quirks.
- **Day of week**: Sunday sales are near-zero in aggregate, since almost all stores are closed, but the minority of stores that do open on Sundays sell about as much per store as on Mondays, the best weekday.
- **Promotions**: average sales are about 39% higher on promo days than non-promo days.
- **Store type / assortment**: type `b` and assortment `b` are rare (17 and 9 stores respectively) but sell noticeably more per store than the other categories — a small high-volume group rather than a broad pattern.
- **Baseline vs. model**: the naive per-store/day-of-week baseline scores **0.236 RMSPE** on the 6-week holdout; LightGBM on the engineered features scores **0.149 RMSPE**, about 37% lower.
- **What drives the forecast**: `Store` identity dominates feature importance (~71% of total gain, reflecting a roughly 20× spread in average sales across stores), `Promo` is the strongest behavioral driver (~15%), and calendar features explain most of the rest. Three engineered features — `IsStateHoliday`, `CompetitionDistanceUnknown`, `UnusualOpenDay` — turned out too rare or too redundant with existing columns for the model to ever split on them.

## Dashboard
A Streamlit app for a regional manager to select a store, compare its forecasted vs. actual sales over the validation period, see what drives the model's predictions, and review a table of stores currently flagged as underperforming (actual sales below predicted by an adjustable threshold).

🔗 Live dashboard: [rossman-sales-forecasting.streamlit.app](https://rossman-sales-forecasting.streamlit.app/)

## Tech Stack
- Python — pandas, scikit-learn, LightGBM/XGBoost
- Streamlit for the interactive dashboard
- Deployed on Streamlit Community Cloud

## Project Structure
```
├── data/            # download train.csv and store.csv from Kaggle, place here
├── notebooks/       # EDA and modeling notebooks
├── src/             # data cleaning + training scripts
├── dashboard/        # Streamlit app (app.py)
└── models/           # saved trained model
```

## How to Run Locally
```bash
git clone https://github.com/india-ehrhardt/rossmann-sales-forecasting.git
cd rossmann-sales-forecasting
pip install -r requirements.txt
streamlit run dashboard/app.py
```
