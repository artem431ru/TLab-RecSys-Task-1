
from pathlib import Path
import numpy as np
import pandas as pd

# Настройки эксперимента
SEED = 42
N_USERS = 10_000

TRAIN_END = pd.Timestamp("2021-07-26")
VAL_END = pd.Timestamp("2021-08-09")

# Загружаем взаимодействия
df = pd.read_csv(
    "data/interactions.csv",
    parse_dates=["last_watch_dt"]
)

# Для выбора пользователей используем
# ТОЛЬКО данные обучающего периода
history = df[df["last_watch_dt"] < TRAIN_END]

activity = history.groupby("user_id").size()
eligible_users = activity[activity >= 3].index.to_numpy()

print("Доступно пользователей:", len(eligible_users))

# Случайно выбираем пользователей
rng = np.random.default_rng(SEED)
selected_users = rng.choice(
    eligible_users,
    size=min(N_USERS, len(eligible_users)),
    replace=False
)

# Сохраняем ВСЮ доступную историю
# выбранных пользователей
df = df[df["user_id"].isin(selected_users)].copy()

# Временное разделение
train = df[df["last_watch_dt"] < TRAIN_END]

val = df[
    (df["last_watch_dt"] >= TRAIN_END) &
    (df["last_watch_dt"] < VAL_END)
]

test = df[df["last_watch_dt"] >= VAL_END]

# Сохраняем результаты
output = Path("data/processed")
output.mkdir(parents=True, exist_ok=True)

for name, part in [
    ("train", train),
    ("val", val),
    ("test", test),
]:
    part.to_csv(output / f"{name}.csv", index=False)

    print(
        f"{name}: {len(part)} взаимодействий, "
        f"{part['user_id'].nunique()} пользователей"
    )

print("\nВсего:", len(df), "взаимодействий")
print("Подготовка завершена!")
