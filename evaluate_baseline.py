
import numpy as np
import pandas as pd

K = 10

# Предсказания CatBoost на Validation
df = pd.read_csv("results/validation_compare.csv")

# Реальные будущие просмотры
future = pd.read_csv(
    "data/processed/val.csv"
)

# Оцениваем только пользователей,
# для которых сформированы рекомендации
users = df["user_id"].unique()
future = future[
    future["user_id"].isin(users)
]

# Число реальных будущих просмотров
# для каждого пользователя
future_counts = future.groupby("user_id").size()

print("Пользователей:", len(users))
print("Будущих просмотров:", len(future))

# Сколько просмотров попало в кандидаты?
candidate_hits = int(df["label"].sum())
candidate_recall = candidate_hits / len(future)

print(
    f"Candidate Recall: {candidate_recall:.4f}"
)

def evaluate(score_column):
    # Сортируем кандидатов по предсказанному баллу
    ranked = df.sort_values(
        ["user_id", score_column, "item_id"],
        ascending=[True, False, True],
        kind="mergesort"
    )

    # Оставляем 10 рекомендаций каждому пользователю
    top = ranked.groupby("user_id").head(K).copy()

    # Определяем позиции рекомендаций
    top["rank"] = (
        top.groupby("user_id").cumcount() + 1
    )

    # Сколько релевантных фильмов найдено в топ-10
    hits = top.groupby("user_id")["label"].sum()

    # Рассчитываем Recall@10 по каждому пользователю,
    # у которого есть реальные будущие просмотры
    recall_per_user = (
        hits.reindex(future_counts.index, fill_value=0)
        / future_counts
    )

    # DCG: релевантные фильмы на высоких местах
    # дают больший вклад
    top["gain"] = (
        top["label"] / np.log2(top["rank"] + 1)
    )

    dcg = top.groupby("user_id")["gain"].sum()

    # Идеальный DCG, если все доступные позиции
    # занимают релевантные фильмы
    discounts = 1 / np.log2(np.arange(2, K + 2))
    ideal_dcg = np.r_[0, np.cumsum(discounts)]

    ideal_counts = np.minimum(
        future_counts.to_numpy(), K
    )

    idcg = ideal_dcg[ideal_counts]

    ndcg_per_user = (
        dcg.reindex(future_counts.index, fill_value=0)
        .to_numpy() / idcg
    )

    # Средние метрики по активным пользователям
    recall = recall_per_user.mean()
    ndcg = ndcg_per_user.mean()

    # Дополнительная метрика:
    # доля всех будущих событий, найденных в топ-10
    global_recall = top["label"].sum() / len(future)

    return recall, ndcg, global_recall


print("\n=== POPULARITY BASELINE ===")

pop_recall, pop_ndcg, pop_global = evaluate(
    "score_popularity"
)

print(f"Recall@10: {pop_recall:.4f}")
print(f"NDCG@10: {pop_ndcg:.4f}")
print(f"Global Recall@10: {pop_global:.4f}")


print("\n=== CATBOOST BASELINE ===")

cat_recall, cat_ndcg, cat_global = evaluate(
    "score_catboost"
)

print(f"Recall@10: {cat_recall:.4f}")
print(f"NDCG@10: {cat_ndcg:.4f}")
print(f"Global Recall@10: {cat_global:.4f}")


print("\n=== РАЗНИЦА ===")

print(f"Recall@10: {cat_recall - pop_recall:+.4f}")
print(f"NDCG@10: {cat_ndcg - pop_ndcg:+.4f}")

print("\n=== CATBOOST + AUTO FEATURES ===")

auto_recall, auto_ndcg, auto_global = evaluate(
    "score_auto"
)

print(f"Recall@10: {auto_recall:.4f}")
print(f"NDCG@10: {auto_ndcg:.4f}")
print(f"Global Recall@10: {auto_global:.4f}")

print("\n=== ПРИРОСТ К РУЧНЫМ ПРИЗНАКАМ ===")

print(f"Recall@10: {auto_recall - cat_recall:+.4f}")
print(f"NDCG@10: {auto_ndcg - cat_ndcg:+.4f}")