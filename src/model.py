import lightgbm as lgb
import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["Store", "DayOfWeek", "StateHoliday", "StoreType", "Assortment"]
NUMERIC_FEATURES = [
    "Promo",
    "SchoolHoliday",
    "CompetitionDistance",
    "CompetitionDistanceUnknown",
    "CompetitionOpenUnknown",
    "CompetitionOpenMonths",
    "Promo2",
    "Promo2Active",
    "IsStateHoliday",
    "UnusualOpenDay",
    "Year",
    "Month",
    "Day",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def rmspe(y_true, y_pred):
    """Root Mean Square Percentage Error, the Rossmann competition metric.

    Rows where actual Sales == 0 are excluded, since percentage error is
    undefined there (closed-store days predict 0 by convention and would
    not contribute to the score either way).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    pct_error = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(pct_error**2)))


def dow_baseline_predict(train_fe, val_fe):
    """Naive baseline: predict each store's historical average Sales for
    that day of week, computed on open days in train_fe only. Falls back
    to the store's overall average, then the global average, for
    (Store, DayOfWeek) combinations unseen in training. Closed days
    (Open == 0) predict 0.
    """
    open_train = train_fe[train_fe["Open"] == 1]
    store_dow_avg = open_train.groupby(["Store", "DayOfWeek"])["Sales"].mean()
    store_avg = open_train.groupby("Store")["Sales"].mean()
    global_avg = open_train["Sales"].mean()

    key = pd.MultiIndex.from_frame(val_fe[["Store", "DayOfWeek"]])
    pred = pd.Series(key.map(store_dow_avg), index=val_fe.index)
    pred = pred.fillna(val_fe["Store"].map(store_avg))
    pred = pred.fillna(global_avg)
    pred = pred.where(val_fe["Open"] == 1, 0)
    return pred


def prep_features(df):
    """Select model features and cast categoricals to pandas 'category'
    dtype so LightGBM can split on them natively.

    Excludes Customers (not available in test.csv, so using it would
    leak future information) and raw columns superseded by engineered
    ones (Promo2SinceWeek/Year, PromoInterval -> Promo2Active).
    """
    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def _rmspe_lgb_eval(y_pred, dataset):
    y_true = dataset.get_label()
    return "RMSPE", rmspe(y_true, y_pred), False


def train_lightgbm(train_fe, val_fe, params=None, num_boost_round=2000, early_stopping_rounds=100):
    """Train on open-store rows only (closed days are a trivial 0 to
    predict and would just dilute training on the signal that matters).
    Early-stopped directly against RMSPE on the validation set.
    """
    train_open = train_fe[train_fe["Open"] == 1]
    val_open = val_fe[val_fe["Open"] == 1]

    X_train, y_train = prep_features(train_open), train_open["Sales"]
    X_val, y_val = prep_features(val_open), val_open["Sales"]

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=CATEGORICAL_FEATURES, reference=train_set)

    default_params = {
        "objective": "regression",
        "metric": "None",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.train(
        default_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        feval=_rmspe_lgb_eval,
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(period=0)],
    )
    return model


def predict_open_rows(model, df):
    """Predict Sales for every row in df, setting closed-store (Open ==
    0) predictions to 0 rather than running them through the model.
    """
    pred = pd.Series(0.0, index=df.index)
    open_mask = df["Open"] == 1
    X = prep_features(df[open_mask])
    pred.loc[open_mask] = model.predict(X, num_iteration=model.best_iteration)
    return pred
