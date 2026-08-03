# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repo is an early-stage scaffold. `src/`, `notebooks/`, `dashboard/`, and `models/` currently contain no
files — only `data/` is populated (raw Kaggle CSVs). When implementing features, you are largely establishing
the code in these directories for the first time rather than following existing conventions, so check the
current directory contents before assuming any file (e.g. `dashboard/app.py`) already exists.

## The project

Rossmann Store Sales Forecasting: predicts daily sales for 1,115 Rossmann drugstores from the
[Kaggle Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) competition dataset, plus a
Streamlit dashboard for regional managers to compare forecasted vs. actual sales and flag underperforming
stores.

## Setup

```bash
pip install -r requirements.txt
```

## Running the dashboard

```bash
streamlit run dashboard/app.py
```

(This file does not exist yet — create it when building the dashboard.)

## Tech stack

- pandas / numpy / scikit-learn for data handling and modeling
- LightGBM for the forecasting model (per README; XGBoost mentioned as an alternative)
- matplotlib / seaborn for EDA plots
- Streamlit for the dashboard, deployed on Streamlit Community Cloud
- joblib for model serialization

## Data (`data/`)

Raw Kaggle CSVs, not committed transformations:

- `train.csv` — ~1M daily records, Jan 2013–Jul 2015: `Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday`
- `test.csv` — same schema minus `Sales`/`Customers`, plus `Id` (used for submission)
- `store.csv` — one row per store (1,115 total): `Store, StoreType, Assortment, CompetitionDistance, CompetitionOpenSinceMonth/Year, Promo2, Promo2SinceWeek/Year, PromoInterval`
- `sample_submission.csv` — `Id, Sales` submission format

`Sales` is the prediction target; `train.csv` and `test.csv` join to `store.csv` on `Store`.

## Intended structure (per README)

```
├── data/            # train.csv and store.csv from Kaggle
├── notebooks/        # EDA and modeling notebooks
├── src/              # data cleaning + training scripts
├── dashboard/         # Streamlit app (app.py)
└── models/            # saved trained model
```
