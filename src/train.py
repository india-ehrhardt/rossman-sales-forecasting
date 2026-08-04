from pathlib import Path

import joblib

from data_prep import build_features
from model import predict_open_rows, rmspe, train_lightgbm

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "lightgbm_model.pkl"


def main():
    train_fe, val_fe = build_features()
    model = train_lightgbm(train_fe, val_fe)

    val_pred = predict_open_rows(model, val_fe)
    val_rmspe = rmspe(val_fe["Sales"], val_pred)
    print(f"Validation RMSPE: {val_rmspe:.4f} (best_iteration={model.best_iteration})")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
