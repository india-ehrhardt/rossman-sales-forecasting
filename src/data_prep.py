from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw():
    train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["Date"], dtype={"StateHoliday": str})
    test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=["Date"], dtype={"StateHoliday": str})
    store = pd.read_csv(f"{DATA_DIR}/store.csv")
    return train, test, store


def clean_store(store):
    """Fix store.csv nulls that are structural, not missing data.

    Promo2 == 0 means a store never opted into the continuity promo, so
    Promo2SinceWeek/Promo2SinceYear/PromoInterval are inapplicable rather
    than unknown. CompetitionDistance/CompetitionOpenSinceMonth/Year are
    left untouched.
    """
    store = store.copy()
    no_promo2 = store["Promo2"] == 0
    store.loc[no_promo2, "Promo2SinceWeek"] = 0
    store.loc[no_promo2, "Promo2SinceYear"] = 0
    store.loc[no_promo2, "PromoInterval"] = ""
    return store


def fill_test_open(test):
    test = test.copy()
    test["Open"] = test["Open"].fillna(1)
    return test


def build_train_store(train, store):
    return train.merge(store, on="Store", how="left")


if __name__ == "__main__":
    train, test, store = load_raw()

    store_clean = clean_store(store)
    test_clean = fill_test_open(test)
    train_store = build_train_store(train, store_clean)

    print("train_store shape:", train_store.shape)
    print("test Open nulls remaining:", test_clean["Open"].isna().sum())
    print(
        "store Promo2-related nulls remaining:",
        store_clean.loc[store_clean["Promo2"] == 0, ["Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]]
        .isna()
        .sum()
        .to_dict(),
    )
