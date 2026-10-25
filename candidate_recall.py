
import pandas as pd
from pathlib import Path

DATA = Path("data/processed")
ML = DATA / "ml"

# Загружаем все взаимодействия выбранных пользователей
events = pd.concat([
    pd.read_csv(DATA / "train.csv"),
    pd.read_csv(DATA / "val.csv"),
    pd.read_csv(DATA / "test.csv")
], ignore_index=True)

events["last_watch_dt"] = pd.to_datetime(
    events["last_watch_dt"]
)

dates = {
    "train_1": "2021-06-14",
    "train_2": "2021-06-28",
    "train_3": "2021-07-12",
    "val": "2021-07-26",
    "test": "2021-08-09"
}

for name, date in dates.items():

    cutoff = pd.Timestamp(date)
    end = cutoff + pd.Timedelta(days=14)

    # Читаем уже созданную обучающую таблицу
    pairs = pd.read_csv(
        ML / f"{name}.csv",
        usecols=["user_id", "label"]
    )

    # Пользователи нашего эксперимента
    users = pairs["user_id"].unique()

    # История строго до даты прогноза
    past = events[
        events["last_watch_dt"] < cutoff
    ]

    # Реальные будущие взаимодействия
    future = events[
        (events["last_watch_dt"] >= cutoff) &
        (events["last_watch_dt"] < end) &
        (events["user_id"].isin(users))
    ]

    # Популярность фильмов только по истории
    popularity = past.groupby("item_id").size()

    popular_items = popularity.nlargest(500).index

    print(f"\n=== {name} ===")
    print("Будущих просмотров:", len(future))

    for k in [150, 300, 500]:

        candidates = popular_items[:k]

        hits = future["item_id"].isin(
            candidates
        ).sum()

        recall = (
            hits / len(future)
            if len(future) > 0 else 0
        )

        print(
            f"Candidate Recall@{k}: "
            f"{recall:.4f} ({recall * 100:.2f}%), "
            f"попаданий: {hits}"
        )

    print(
        "Положительных примеров в текущей таблице:",
        int(pairs["label"].sum())
    )
