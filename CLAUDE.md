# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

All the top-level directories from the intended structure now have content: `src/data_prep.py` (cleaning +
feature engineering, including `build_features()` as the single pipeline entry point), `src/model.py`
(baseline + LightGBM training/evaluation), `src/train.py` (trains and saves the model to
`models/lightgbm_model.pkl` via joblib), `notebooks/01_eda.ipynb`, and `dashboard/app.py` (Streamlit app —
store selector, predicted vs. actual chart, feature importance, underperforming-stores table). Check current
file contents before assuming behavior, since this is still an early, evolving codebase.

## Workflow: commit and push after meaningful work

After finishing any meaningful unit of work — a cleaning step, a set of EDA cells, a model change, a
dashboard change — proactively commit and push to GitHub. Don't wait to be asked.

- Write a clear, descriptive commit message (what changed and why, not just what file).
- Still show the diff/`git status` and the exact commands you're about to run, and wait for approval before
  executing `git commit`/`git push` — this rule authorizes doing the work unprompted, not skipping the
  confirmation step.
- If a notebook was touched, re-execute it (e.g. via `jupyter nbconvert --execute --inplace`) and confirm no
  cell errors before committing, so committed notebooks always reflect a clean run.

## The project

Rossmann Store Sales Forecasting: predicts daily sales for 1,115 Rossmann drugstores from the
[Kaggle Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) competition dataset, plus a
Streamlit dashboard for regional managers to compare forecasted vs. actual sales and flag underperforming
stores.

## Setup

```bash
pip install -r requirements.txt
```

On macOS, LightGBM also needs the OpenMP runtime, which is not a pip package: `brew install libomp`.
Without it, `import lightgbm` fails with an `OSError: ... Library not loaded: @rpath/libomp.dylib`.

## Running the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard reads only `data/val_predictions.parquet` and `data/feature_importance.parquet` — both
committed, both small. It does **not** load `train.csv`/`store.csv` or the model at runtime, since those raw
CSVs are gitignored (too large to commit) and would not exist in a deployed environment like Streamlit
Community Cloud. If those parquet files are missing (e.g. after changing the feature/model pipeline), run
`python src/train.py` to regenerate them alongside `models/lightgbm_model.pkl`.

Layout: store selector and underperforming-threshold slider live in the sidebar; the main area has a KPI
row (`st.metric` with deltas — avg. daily sales, model accuracy, underperforming store count) above three
tabs (Forecast, Feature Importance, Underperforming Stores). Charts are Plotly (`go.Figure`, not
matplotlib) so hover/zoom/range-slider work; `st.plotly_chart`'s default `theme="streamlit"` makes them
follow the active light/dark theme automatically — don't hardcode a paper/plot background color on the
figures, or that auto-theming breaks.

## Custom theme (`.streamlit/config.toml`)

Defines both `[theme.light]` and `[theme.dark]` (extending the base `[theme]`), which is what makes
Streamlit's own System/Light/Dark toggle (in the app's `⋮` menu) show real content instead of the default
theme in both states. Accent/positive/negative colors (`primaryColor`, `greenColor`, `redColor`) are shared
across both — `greenColor`/`redColor` are what `st.metric`'s delta arrows use, not just decorative
"palette" colors, so don't rename or drop them without checking the KPI row. Font is Inter, loaded via
`[[theme.fontFaces]]` pointing at pinned `@fontsource/inter` files on jsDelivr (not the simpler
`font = "Inter:<google fonts css url>"` shortcut) — that was a deliberate choice, keep it that way rather
than "simplifying" back to the CSS-URL form.

## Tech stack

- pandas / numpy / scikit-learn for data handling and modeling
- LightGBM for the forecasting model (per README; XGBoost mentioned as an alternative)
- matplotlib / seaborn for EDA plots in the notebook
- Streamlit for the dashboard, deployed on Streamlit Community Cloud
- Plotly for the dashboard's own charts (interactive hover/zoom) — the notebook still uses matplotlib
- joblib for model serialization
- pyarrow for the parquet files the dashboard reads

## Data (`data/`)

Raw Kaggle CSVs, not committed transformations:

- `train.csv` — ~1M daily records, Jan 2013–Jul 2015: `Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday`
- `test.csv` — same schema minus `Sales`/`Customers`, plus `Id` (used for submission)
- `store.csv` — one row per store (1,115 total): `Store, StoreType, Assortment, CompetitionDistance, CompetitionOpenSinceMonth/Year, Promo2, Promo2SinceWeek/Year, PromoInterval`
- `sample_submission.csv` — `Id, Sales` submission format

`Sales` is the prediction target; `train.csv` and `test.csv` join to `store.csv` on `Store`.

Also in `data/`, but committed (small, not gitignored): `val_predictions.parquet` (Store/Date/Sales/Open/
Predicted for the validation holdout) and `feature_importance.parquet` (Feature/Gain) — both written by
`src/train.py` and read by the dashboard instead of the raw CSVs.

## Intended structure (per README)

```
├── data/            # train.csv and store.csv from Kaggle
├── notebooks/        # EDA and modeling notebooks
├── src/              # data cleaning + training scripts
├── dashboard/         # Streamlit app (app.py)
└── models/            # saved trained model
```
