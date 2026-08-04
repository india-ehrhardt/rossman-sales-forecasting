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
    than unknown.

    CompetitionDistance is null for 3 stores with no competition data at
    all (also missing CompetitionOpenSinceMonth/Year) — imputed with the
    median distance, flagged via CompetitionDistanceUnknown.

    CompetitionOpenSinceMonth/Year is null for 354 stores (3 of which are
    the ones above); the other 351 have a known CompetitionDistance but an
    unrecorded open date. Left null here — flagged via CompetitionOpenUnknown
    — and deferred to feature engineering, since imputing a date now would
    bake in an unverified assumption about "months since competition opened".
    """
    store = store.copy()
    no_promo2 = store["Promo2"] == 0
    store.loc[no_promo2, "Promo2SinceWeek"] = 0
    store.loc[no_promo2, "Promo2SinceYear"] = 0
    store.loc[no_promo2, "PromoInterval"] = ""

    store["CompetitionDistanceUnknown"] = store["CompetitionDistance"].isna().astype(int)
    median_distance = store["CompetitionDistance"].median()
    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(median_distance)

    store["CompetitionOpenUnknown"] = store["CompetitionOpenSinceMonth"].isna().astype(int)

    return store


def fill_test_open(test):
    test = test.copy()
    test["Open"] = test["Open"].fillna(1)
    return test


def build_train_store(train, store):
    return train.merge(store, on="Store", how="left")


def add_date_features(df):
    """Derive Year/Month/Day/DayOfWeek from Date. DayOfWeek uses the ISO
    convention (1=Monday..7=Sunday) to match the raw Kaggle column."""
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.weekday + 1
    return df


_PROMO_INTERVAL_MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec",
}  # Sept, not Sep — matches the raw PromoInterval strings in store.csv


def add_promo_holiday_features(df):
    """Add IsStateHoliday (bool version of StateHoliday) and Promo2Active
    (whether the store's continuity promo is running as of this row's
    Date): Promo2 == 1, the date is on/after the promo's ISO-week start
    (Promo2SinceYear/Week), and the row's month is one of PromoInterval's
    months. Requires Year/Month from add_date_features.
    """
    df = df.copy()
    df["IsStateHoliday"] = (df["StateHoliday"] != "0").astype(int)

    iso = df["Date"].dt.isocalendar()
    started = (iso["year"] > df["Promo2SinceYear"]) | (
        (iso["year"] == df["Promo2SinceYear"]) & (iso["week"] >= df["Promo2SinceWeek"])
    )
    month_abbr = df["Month"].map(_PROMO_INTERVAL_MONTH_ABBR)
    in_promo_month = pd.Series(
        [
            abbr in interval.split(",") if interval else False
            for abbr, interval in zip(month_abbr, df["PromoInterval"])
        ],
        index=df.index,
    )
    df["Promo2Active"] = ((df["Promo2"] == 1) & started & in_promo_month).astype(int)
    return df


def add_competition_open_months(df):
    """Months since a nearby competitor opened, as of this row's Date.
    NaN when the open date itself is unknown (see CompetitionOpenUnknown
    in clean_store) rather than fabricating one; clipped at 0 when the
    competitor hasn't opened yet as of this date. Requires Year/Month
    from add_date_features.
    """
    df = df.copy()
    months_since = 12 * (df["Year"] - df["CompetitionOpenSinceYear"]) + (
        df["Month"] - df["CompetitionOpenSinceMonth"]
    )
    df["CompetitionOpenMonths"] = months_since.clip(lower=0)
    return df


def time_based_split(df, weeks_holdout=6):
    """Split a date-indexed dataframe into train/validation by time, with
    no shuffling: the most recent `weeks_holdout` weeks become validation.
    """
    cutoff = df["Date"].max() - pd.Timedelta(weeks=weeks_holdout)
    train_split = df[df["Date"] <= cutoff].copy()
    val_split = df[df["Date"] > cutoff].copy()
    return train_split, val_split


def compute_dow_open_rate(df):
    """Per-store, per-day-of-week fraction of days the store is open.
    Fit this on training data only, then apply to validation/test, so a
    store's usual-open pattern doesn't leak in from the holdout period.
    """
    return df.groupby(["Store", "DayOfWeek"])["Open"].mean()


def add_unusual_open_flag(df, dow_open_rate, threshold=0.5):
    """Flag rows where a store is open on a day of week it is usually
    closed on (its historical open rate for that store/day-of-week is
    below `threshold`) — e.g. a store that almost never opens on Sundays
    but does on this particular date.
    """
    df = df.copy()
    key = pd.MultiIndex.from_frame(df[["Store", "DayOfWeek"]])
    typical_rate = pd.Series(key.map(dow_open_rate), index=df.index)
    df["UnusualOpenDay"] = ((df["Open"] == 1) & (typical_rate.fillna(0) < threshold)).astype(int)
    return df


def build_features(weeks_holdout=6):
    """Run the full cleaning + feature engineering pipeline and return
    (train_fe, val_fe) split by time, with every engineered feature
    applied to both. The single entry point used by training scripts
    and the dashboard, so the pipeline is defined in one place.
    """
    train, test, store = load_raw()
    store_clean = clean_store(store)
    train_store = build_train_store(train, store_clean)
    train_store = add_date_features(train_store)
    train_store = add_promo_holiday_features(train_store)
    train_store = add_competition_open_months(train_store)

    train_fe, val_fe = time_based_split(train_store, weeks_holdout=weeks_holdout)
    dow_open_rate = compute_dow_open_rate(train_fe)
    train_fe = add_unusual_open_flag(train_fe, dow_open_rate)
    val_fe = add_unusual_open_flag(val_fe, dow_open_rate)
    return train_fe, val_fe


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
    print("store CompetitionDistance nulls remaining:", store_clean["CompetitionDistance"].isna().sum())
    print("store CompetitionDistanceUnknown flagged:", store_clean["CompetitionDistanceUnknown"].sum())
    print("store CompetitionOpenUnknown flagged:", store_clean["CompetitionOpenUnknown"].sum())
    print(
        "store CompetitionOpenSinceMonth/Year nulls remaining:",
        store_clean[["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]].isna().sum().to_dict(),
    )

    train_store = add_date_features(train_store)
    train_store = add_promo_holiday_features(train_store)
    train_store = add_competition_open_months(train_store)

    print("Promo2Active rate among Promo2==1 rows:", train_store.loc[train_store["Promo2"] == 1, "Promo2Active"].mean())
    print("Promo2Active among Promo2==0 rows (should be 0):", train_store.loc[train_store["Promo2"] == 0, "Promo2Active"].sum())
    print("IsStateHoliday rate:", train_store["IsStateHoliday"].mean())
    print(
        "CompetitionOpenMonths: nulls =",
        train_store["CompetitionOpenMonths"].isna().sum(),
        ", min =", train_store["CompetitionOpenMonths"].min(),
        ", max =", train_store["CompetitionOpenMonths"].max(),
    )

    train_fe, val_fe = time_based_split(train_store, weeks_holdout=6)
    dow_open_rate = compute_dow_open_rate(train_fe)
    train_fe = add_unusual_open_flag(train_fe, dow_open_rate)
    val_fe = add_unusual_open_flag(val_fe, dow_open_rate)

    print("train_fe date range:", train_fe["Date"].min(), "to", train_fe["Date"].max())
    print("val_fe date range:", val_fe["Date"].min(), "to", val_fe["Date"].max())
    print("train_fe/val_fe shapes:", train_fe.shape, val_fe.shape)
    print("train_fe UnusualOpenDay flagged:", train_fe["UnusualOpenDay"].sum())
    print("val_fe UnusualOpenDay flagged:", val_fe["UnusualOpenDay"].sum())
