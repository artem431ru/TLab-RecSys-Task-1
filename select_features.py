
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

DATA = Path("data/processed/ml_auto")
RESULTS = Path("results")
K = 10

manual = [
    "user_views_total",
    "user_views_7d",
    "user_views_30d",
    "item_views_total",
    "item_views_7d",
    "item_views_30d",
    "release_year",
    "genres",
]

# Загружаем обучающие данные
train = pd.concat([
    pd.read_csv(DATA / f"train_{i}.csv")
    for i in [1, 2, 3]
], ignore_index=True)

val = pd.read_csv(DATA / "val.csv")
future = pd.read_csv("data/processed/val.csv")

future_counts = (
    future[future["user_id"].isin(val["user_id"])]
    .groupby("user_id")
    .size()
)

# Берём важность из расширенной модели
previous_model = CatBoostClassifier()
previous_model.load_model(
    str(RESULTS / "catboost_auto.cbm")
)

importance = pd.Series(
    previous_model.get_feature_importance(),
    index=previous_model.feature_names_
)

auto_features = [
    col for col in importance.index
    if col not in manual
]

auto_features = sorted(
    auto_features,
    key=lambda col: importance[col],
    reverse=True
)

print("Рейтинг автоматических признаков:")
print(auto_features)

def evaluate(scores):
    ranked = val[
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

    hits = top.groupby("user_id")["label"].sum()

    recall = (
        hits.reindex(future_counts.index, fill_value=0)
        / future_counts
    ).mean()

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

    ideal_counts = np.minimum(
        future_counts.to_numpy(), K
    )

    ndcg = np.mean(dcg / ideal[ideal_counts])

    return float(recall), float(ndcg)


results = []
best_ndcg = -1

for n in [0, 3, 6, 10, len(auto_features)]:

    features = manual + auto_features[:n]

    for frame in [train, val]:
        frame["genres"] = (
            frame["genres"].fillna("unknown").astype(str)
        )

        for col in features:
            if col != "genres":
                frame[col] = pd.to_numeric(
                    frame[col], errors="coerce"
                ).fillna(0).astype("float32")

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        auto_class_weights="SqrtBalanced",
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )

    model.fit(
        train[features],
        train["label"],
        cat_features=["genres"],
        eval_set=(val[features], val["label"]),
        early_stopping_rounds=40,
    )

    scores = model.predict_proba(val[features])[:, 1]
    recall, ndcg = evaluate(scores)

    results.append({
        "auto_features": n,
        "recall@10": recall,
        "ndcg@10": ndcg,
    })

    print(
        f"Автопризнаков: {n:2d} | "
        f"Recall@10: {recall:.4f} | "
        f"NDCG@10: {ndcg:.4f}"
    )

    if ndcg > best_ndcg:
        best_ndcg = ndcg

        model.save_model(
            str(RESULTS / "catboost_selected.cbm")
        )

        with open(
            RESULTS / "selected_features.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(features, f, indent=2)

summary = pd.DataFrame(results)
summary.to_csv(
    RESULTS / "feature_selection_validation.csv",
    index=False
)

print("\n=== ИТОГИ ===")
print(summary.to_string(index=False))
print("\nЛучшая NDCG@10:", round(best_ndcg, 4))
