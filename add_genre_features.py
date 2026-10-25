
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("data/processed")
SOURCE = BASE / "ml"
OUTPUT = BASE / "ml_genre"
OUTPUT.mkdir(parents=True, exist_ok=True)

DATES = {
    "train_1": "2021-06-14",
    "train_2": "2021-06-28",
    "train_3": "2021-07-12",
    "val": "2021-07-26",
    "test": "2021-08-09",
}

# Загружаем только нужные столбцы
events = pd.concat([
    pd.read_csv(
        BASE / f"{name}.csv",
        usecols=["user_id", "item_id", "last_watch_dt"],
        parse_dates=["last_watch_dt"]
    )
    for name in ["train", "val", "test"]
], ignore_index=True)

items = pd.read_csv(
    "data/items.csv",
    usecols=["item_id", "genres"]
)

# Разбиваем строку жанров на отдельные значения
items["genre"] = (
    items["genres"]
    .fillna("")
    .astype(str)
    .str.split(",")
)

item_genres = items[
    ["item_id", "genre"]
].explode("genre")

item_genres["genre"] = (
    item_genres["genre"]
    .str.strip()
    .str.lower()
)

item_genres = item_genres[
    item_genres["genre"].notna() &
    (item_genres["genre"] != "")
].drop_duplicates(["item_id", "genre"])

for name, date in DATES.items():
    cutoff = pd.Timestamp(date)
    pairs = pd.read_csv(SOURCE / f"{name}.csv")

    past = events[
        events["last_watch_dt"] < cutoff
    ]

    # Сопоставляем фильмам-кандидатам их жанры
    expanded = (
        pairs[["user_id", "item_id"]]
        .reset_index(names="row_id")
        .merge(item_genres, on="item_id", how="left")
        .dropna(subset=["genre"])
    )

    for days in [None, 7, 30]:
        suffix = "total" if days is None else f"{days}d"

        history = past if days is None else past[
            past["last_watch_dt"] >=
            cutoff - pd.Timedelta(days=days)
        ]

        # Подсчитываем интерес пользователя к жанрам
        history_genres = history.merge(
            item_genres, on="item_id", how="inner"
        )

        counts = (
            history_genres
            .groupby(["user_id", "genre"])
            .size()
        )

        user_totals = history.groupby("user_id").size()

        keys = pd.MultiIndex.from_frame(
            expanded[["user_id", "genre"]]
        )

        genre_counts = counts.reindex(
            keys, fill_value=0
        ).to_numpy(dtype="float32")

        totals = (
            expanded["user_id"]
            .map(user_totals)
            .fillna(0)
            .to_numpy(dtype="float32")
        )

        expanded["share"] = (
            genre_counts / np.maximum(totals, 1)
        )

        # Для многожанрового фильма берём среднее
        values = expanded.groupby("row_id")["share"].mean()

        column = f"user_genre_share_{suffix}"

        pairs[column] = (
            values.reindex(pairs.index, fill_value=0)
            .to_numpy(dtype="float32")
        )

    pairs.to_csv(OUTPUT / f"{name}.csv", index=False)

    print(
        name,
        "строк:", len(pairs),
        "столбцов:", len(pairs.columns),
        "label=1:", int(pairs["label"].sum())
    )

print("Жанровые признаки созданы!")
