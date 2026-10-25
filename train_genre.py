
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

DATA = Path("data/processed/ml_genre")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Загружаем данные
train = pd.concat([
    pd.read_csv(DATA / f"train_{i}.csv")
    for i in [1, 2, 3]
], ignore_index=True)

val = pd.read_csv(DATA / "val.csv")

manual_features = [
    "user_views_total",
    "user_views_7d",
    "user_views_30d",
    "item_views_total",
    "item_views_7d",
    "item_views_30d",
    "release_year",
    "genres"
]

genre_features = [
    "user_genre_share_total",
    "user_genre_share_7d",
    "user_genre_share_30d"
]

features = manual_features + genre_features

# Подготовка признаков
for df in [train, val]:
    df["genres"] = df["genres"].fillna("unknown").astype(str)

    for col in features:
        if col != "genres":
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            ).fillna(0)

print("Train:", train.shape)
print("Validation:", val.shape)

print("\nСтатистика новых признаков:")
print(val[genre_features].describe())

# Обучение
model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.08,
    loss_function="Logloss",
    auto_class_weights="SqrtBalanced",
    random_seed=42,
    verbose=100,
    thread_count=-1
)

model.fit(
    train[features],
    train["label"],
    cat_features=["genres"],
    eval_set=(val[features], val["label"]),
    early_stopping_rounds=40
)

val["score_genre"] = model.predict_proba(
    val[features]
)[:, 1]

model.save_model(str(RESULTS / "catboost_genre.cbm"))

# Загружаем предсказания исходной модели
previous = pd.read_csv(
    RESULTS / "validation_predictions.csv"
)

keys = ["user_id", "item_id", "label"]

assert previous[keys].equals(val[keys]), (
    "Наборы Validation не совпадают!"
)

val["score_catboost"] = previous["score_catboost"].to_numpy()
val["score_popularity"] = previous["score_popularity"].to_numpy()

# Подготовка метрик
future = pd.read_csv("data/processed/val.csv")

future_counts = (
    future[future["user_id"].isin(val["user_id"])]
    .groupby("user_id")
    .size()
)

def evaluate(score_col, k=10):
    ranked = val.sort_values(
        ["user_id", score_col, "item_id"],
        ascending=[True, False, True],
        kind="mergesort"
    )

    top = ranked.groupby("user_id").head(k).copy()
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

    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.r_[0, np.cumsum(discounts)]
    counts = np.minimum(future_counts.to_numpy(), k)

    ndcg = np.mean(dcg / ideal[counts])

    return recall, ndcg

print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===")

for name, score in [
    ("Popularity", "score_popularity"),
    ("CatBoost (8)", "score_catboost"),
    ("CatBoost + Genres (11)", "score_genre")
]:
    recall, ndcg = evaluate(score)

    print(
        f"{name}: "
        f"Recall@10={recall:.4f}, "
        f"NDCG@10={ndcg:.4f}"
    )

val.to_csv(
    RESULTS / "validation_genre.csv",
    index=False
)

print("\n=== ВАЖНОСТЬ ПРИЗНАКОВ ===")

importance = pd.Series(
    model.get_feature_importance(),
    index=features
).sort_values(ascending=False)

print(importance)

print("\nГотово!")
