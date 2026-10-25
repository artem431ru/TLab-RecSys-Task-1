
from pathlib import Path
import pandas as pd

DATA = Path("data/processed")
ML = DATA / "ml"
OUTPUT = DATA / "ml_auto"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Варианты временных окон
WINDOWS = [3, 7, 14, 30, 60]

# Сущности, по которым считаем статистики
ENTITIES = {
    "user": "user_id",
    "item": "item_id"
}

# Моменты формирования рекомендаций
DATES = {
    "train_1": "2021-06-14",
    "train_2": "2021-06-28",
    "train_3": "2021-07-12",
    "val": "2021-07-26",
    "test": "2021-08-09"
}

# Загружаем историю взаимодействий
events = pd.concat([
    pd.read_csv(DATA / "train.csv"),
    pd.read_csv(DATA / "val.csv"),
    pd.read_csv(DATA / "test.csv")
], ignore_index=True)

events["last_watch_dt"] = pd.to_datetime(
    events["last_watch_dt"]
)

for name, date in DATES.items():

    cutoff = pd.Timestamp(date)

    # Загружаем существующие пары user-item
    pairs = pd.read_csv(ML / f"{name}.csv")

    # Используем только прошлое
    past = events[
        events["last_watch_dt"] < cutoff
    ]

    # Автоматическая генерация
    for window in WINDOWS:

        start = cutoff - pd.Timedelta(days=window)

        recent = past[
            past["last_watch_dt"] >= start
        ]

        for entity, column in ENTITIES.items():

            feature_name = (
                f"{entity}_views_{window}d"
            )

            counts = recent.groupby(column).size()

            pairs[feature_name] = (
                pairs[column]
                .map(counts)
                .fillna(0)
                .astype("float32")
            )

    # Генерируем относительные признаки
    for entity in ENTITIES:

        for short, long in [
            (3, 7),
            (7, 14),
            (14, 30),
            (30, 60)
        ]:

            numerator = pairs[
                f"{entity}_views_{short}d"
            ]

            denominator = pairs[
                f"{entity}_views_{long}d"
            ]

            pairs[
                f"{entity}_ratio_{short}_{long}"
            ] = numerator / (denominator + 1)

    pairs.to_csv(
        OUTPUT / f"{name}.csv",
        index=False
    )

    print(
        f"{name}: {len(pairs)} строк, "
        f"{len(pairs.columns)} столбцов, "
        f"label=1: {int(pairs['label'].sum())}"
    )

print("\nАвтоматическая генерация завершена!")
