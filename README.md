# Rossmann Store Sales Forecasting & Dashboard

## The Business Problem
Rossmann operates over 3,000 drugstores across 7 European countries. Store managers currently forecast their own daily sales up to six weeks out, with accuracy varying widely from manager to manager. A reliable, data-driven forecast lets staffing and inventory decisions be based on real patterns instead of guesswork.

**This project builds a machine learning model that predicts daily sales for 1,115 stores, and an interactive dashboard so a regional manager can see forecasted vs. actual sales and flag underperforming stores.**

## Data Source
- [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) — Kaggle competition dataset
- ~1M daily sales records across 1,115 stores (Jan 2013 – Jul 2015), plus store-level metadata (store type, assortment, competition distance, promo intervals, state/school holidays)

## Approach
🚧 In progress — check back soon.

## Results
🚧 In progress — check back soon.

## Dashboard
🚧 In progress — check back soon.

🔗 Live dashboard: 🚧 coming soon

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
