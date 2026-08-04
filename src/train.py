from pathlib import Path

import joblib
import pandas as pd

from data_prep import build_features
from model import predict_open_rows, rmspe, train_lightgbm

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "lightgbm_model.pkl"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VAL_PREDICTIONS_PATH = DATA_DIR / "val_predictions.parquet"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "feature_importance.parquet"


def main():
    train_fe, val_fe = build_features()
    model = train_lightgbm(train_fe, val_fe)

    val_pred = predict_open_rows(model, val_fe)
    val_rmspe = rmspe(val_fe["Sales"], val_pred)
    print(f"Validation RMSPE: {val_rmspe:.4f} (best_iteration={model.best_iteration})")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Precompute what the dashboard needs, since train.csv/store.csv are
    # gitignored (too large to commit) and won't exist in a deployed
    # environment — the dashboard reads these small files instead of
    # calling build_features()/load_raw() at runtime.
    val_predictions = val_fe[["Store", "Date", "Sales", "Open"]].copy()
    val_predictions["Predicted"] = val_pred
    val_predictions.to_parquet(VAL_PREDICTIONS_PATH, index=False)
    print(f"Saved {len(val_predictions)} validation predictions to {VAL_PREDICTIONS_PATH}")

    importance = pd.Series(model.feature_importance(importance_type="gain"), index=model.feature_name())
    feature_importance = (
        pd.DataFrame({"Feature": importance.index, "Gain": importance.values})
        .sort_values("Gain", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_parquet(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"Saved feature importance to {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
