
import pandas as pd

# 1. Загрузка датасета
print("Загружаем датасет KION...")

df = pd.read_csv(
    "data/interactions.csv",
    parse_dates=["last_watch_dt"]
)

# 2. Основные характеристики
print("\n=== ОБЩАЯ ИНФОРМАЦИЯ ===")

print("Количество взаимодействий:", len(df))
print("Количество пользователей:", df["user_id"].nunique())
print("Количество фильмов:", df["item_id"].nunique())

# 3. Временной период
print("\n=== ДАТЫ ===")

print("Первая дата:", df["last_watch_dt"].min())
print("Последняя дата:", df["last_watch_dt"].max())

print("\nКоличество взаимодействий по месяцам:")
print(
    df.groupby(
        df["last_watch_dt"].dt.to_period("M")
    ).size()
)

# 4. Активность пользователей
print("\n=== АКТИВНОСТЬ ПОЛЬЗОВАТЕЛЕЙ ===")

user_activity = df.groupby("user_id").size()

print("Среднее число взаимодействий:", user_activity.mean())
print("Медиана:", user_activity.median())
print("Максимум:", user_activity.max())
print("Пользователей с одним взаимодействием:",
      (user_activity == 1).sum())

# 5. Популярность фильмов
print("\n=== ПОПУЛЯРНОСТЬ ФИЛЬМОВ ===")

item_popularity = df.groupby("item_id").size()

print("Среднее число взаимодействий:", item_popularity.mean())
print("Самый популярный фильм:", item_popularity.idxmax())
print("Количество его взаимодействий:", item_popularity.max())

# 6. Проверка качества данных
print("\n=== КАЧЕСТВО ДАННЫХ ===")

print("Пропущенные значения:")
print(df.isna().sum())

print("Полные дубликаты:", df.duplicated().sum())
print(
    "Повторяющиеся пары user-item:",
    df.duplicated(subset=["user_id", "item_id"]).sum()
)

# 7. Статистика процента просмотра
print("\n=== ПРОЦЕНТ ПРОСМОТРА ===")
print(df["watched_pct"].describe())

print("\nАнализ завершён!")
