
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

GENRE = Path("data/processed/ml_genre")
AUTO = Path("data/processed/ml_auto")
RESULTS = Path("results")
K = 10

# Модели уже обучены и сохранены
model_files = {
    "CatBoost_8": "catboost_baseline.cbm",
    "Manual_11": "manual_11.cbm",
    "Auto_17": "manual_plus_counts.cbm",
}

models = {}

for name, filename in model_files.items():
    model = CatBoostClassifier()
    model.load_model(str(RESULTS / filename))
    models[name] = model

# Загружаем тестовые пары и ручные признаки
test = pd.read_csv(GENRE / "test.csv")

# Выясняем, какие автоматические признаки нужны
needed = sorted({
    col
    for model in models.values()
    for col in model.feature_names_
    if col not in test.columns
})

if needed:
    extra = pd.read_csv(
        AUTO / "test.csv",
        usecols=["user_id", "item_id", "label"] + needed
    )

    keys = ["user_id", "item_id", "label"]

    assert test[keys].equals(extra[keys]), (
        "Ошибка: тестовые пары не совпадают!"
    )

    for col in needed:
        test[col] = extra[col].to_numpy()

test["genres"] = (
    test["genres"].fillna("unknown").astype(str)
)

# Реальные просмотры в тестовом периоде
future = pd.read_csv("data/processed/test.csv")

future = future[
    future["user_id"].isin(test["user_id"])
]

future_counts = future.groupby("user_id").size()

print("Тестовых пар:", len(test))
print("Реальных будущих просмотров:", len(future))
print("Пользователей с будущими просмотрами:",
      len(future_counts))

print(
    "Candidate Recall:",
    round(test["label"].sum() / len(future), 4)
)

def evaluate(scores):
    ranked = test[
        ["user_id", "item_id", "label"]
    ].copy()

    ranked["score"] = scores

    ranked = ranked.sort_values(
        ["user_id", "score", "item_id"],
        ascending=[True, False, True],
        kind="mergesort"
    )

    top = ranked.groupby("user_id").head(K).copy()
    top["rank"] = top.groupby("user_id").cumcount() + 1

    hits = (
        top.groupby("user_id")["label"]
        .sum()
        .reindex(future_counts.index, fill_value=0)
    )

    recall = (hits / future_counts).mean()

    top["gain"] = (
        top["label"] / np.log2(top["rank"] + 1)
    )

    dcg = (
        top.groupby("user_id")["gain"]
        .sum()
        .reindex(future_counts.index, fill_value=0)
        .to_numpy()
    )

    discounts = 1 / np.log2(np.arange(2, K + 2))
    ideal = np.r_[0, np.cumsum(discounts)]

    counts = np.minimum(
        future_counts.to_numpy(), K
    )

    ndcg = np.mean(dcg / ideal[counts])
    global_recall = top["label"].sum() / len(future)

    return float(recall), float(ndcg), float(global_recall)


results = []

# Популярностная модель
scores = test["item_views_total"].to_numpy()
recall, ndcg, global_recall = evaluate(scores)

results.append({
    "model": "Popularity",
    "recall@10": recall,
    "ndcg@10": ndcg,
    "global_recall@10": global_recall
})

# Все CatBoost-модели
for name, model in models.items():

    scores = model.predict_proba(
        test[model.feature_names_]
    )[:, 1]

    recall, ndcg, global_recall = evaluate(scores)

    results.append({
        "model": name,
        "recall@10": recall,
        "ndcg@10": ndcg,
        "global_recall@10": global_recall
    })

summary = pd.DataFrame(results)

summary.to_csv(
    RESULTS / "test_metrics.csv",
    index=False
)

print("\n=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ===")
print(summary.to_string(index=False))
