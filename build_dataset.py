
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/processed")
OUTPUT = DATA / "ml"
OUTPUT.mkdir(exist_ok=True)

# Все временные части нужны для формирования
# истории и будущих целевых событий
df = pd.concat([
    pd.read_csv(DATA / "train.csv"),
    pd.read_csv(DATA / "val.csv"),
    pd.read_csv(DATA / "test.csv")
], ignore_index=True)

df["last_watch_dt"] = pd.to_datetime(df["last_watch_dt"])

items = pd.read_csv(
    "data/items.csv",
    usecols=["item_id", "genres", "release_year"]
).drop_duplicates("item_id")

items["genres"] = items["genres"].fillna("unknown")
items["release_year"] = pd.to_numeric(
    items["release_year"], errors="coerce"
).fillna(0)

# Для быстрого первого эксперимента
FIRST_DATE = pd.Timestamp("2021-06-14")

activity = (
    df[df["last_watch_dt"] < FIRST_DATE]
    .groupby("user_id")
    .size()
)

eligible = activity[activity >= 2].index.to_numpy()

rng = np.random.default_rng(42)
cohort = rng.choice(
    eligible,
    size=min(1500, len(eligible)),
    replace=False
)

def build_snapshot(date):
    cutoff = pd.Timestamp(date)
    end = cutoff + pd.Timedelta(days=14)

    past = df[df["last_watch_dt"] < cutoff]
    future = df[
        (df["last_watch_dt"] >= cutoff) &
        (df["last_watch_dt"] < end)
    ]

    # Кандидаты: популярные фильмы только из прошлого
    popular = (
        past.groupby("item_id")
        .size()
        .nlargest(300)
        .index
    )

    pairs = pd.MultiIndex.from_product(
        [cohort, popular],
        names=["user_id", "item_id"]
    ).to_frame(index=False)

    # Удаляем уже просмотренные фильмы
    seen = past[["user_id", "item_id"]].drop_duplicates()
    seen["seen"] = 1

    pairs = pairs.merge(
        seen, on=["user_id", "item_id"], how="left"
    )

    pairs = pairs[pairs["seen"].isna()].drop(
        columns="seen"
    )

    # Агрегированные признаки
    windows = [7, 30]

    groups = [
        (past, "total")
    ]

    for days in windows:
        part = past[
            past["last_watch_dt"] >=
            cutoff - pd.Timedelta(days=days)
        ]
        groups.append((part, f"{days}d"))

    for part, suffix in groups:
        user_count = (
            part.groupby("user_id")
            .size()
            .rename(f"user_views_{suffix}")
        )
        item_count = (
            part.groupby("item_id")
            .size()
            .rename(f"item_views_{suffix}")
        )

        pairs = pairs.join(user_count, on="user_id")
        pairs = pairs.join(item_count, on="item_id")

    # Характеристики фильмов
    pairs = pairs.merge(items, on="item_id", how="left")

    # Будущие взаимодействия: целевая переменная
    targets = future[
        ["user_id", "item_id"]
    ].drop_duplicates()

    targets["label"] = 1

    pairs = pairs.merge(
        targets,
        on=["user_id", "item_id"],
        how="left"
    )

    pairs["label"] = pairs["label"].fillna(0).astype(int)

    count_cols = [
        c for c in pairs.columns
        if c.startswith(("user_views_", "item_views_"))
    ]
    pairs[count_cols] = pairs[count_cols].fillna(0)

    pairs["genres"] = pairs["genres"].fillna("unknown")
    pairs["release_year"] = pairs["release_year"].fillna(0)

    return pairs


dates = {
    "train_1": "2021-06-14",
    "train_2": "2021-06-28",
    "train_3": "2021-07-12",
    "val": "2021-07-26",
    "test": "2021-08-09"
}

for name, date in dates.items():
    result = build_snapshot(date)
    result.to_csv(OUTPUT / f"{name}.csv", index=False)

    print(
        name,
        "строк:", len(result),
        "положительных:", result["label"].sum()
    )

print("Готово!")
