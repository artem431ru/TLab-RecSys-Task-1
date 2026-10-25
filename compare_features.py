
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

GENRE = Path("data/processed/ml_genre")
AUTO = Path("data/processed/ml_auto")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

manual = [
    "user_views_total",
    "user_views_7d",
    "user_views_30d",
    "item_views_total",
    "item_views_7d",
    "item_views_30d",
    "release_year",
    "genres",
    "user_genre_share_total",
    "user_genre_share_7d",
    "user_genre_share_30d",
]

auto_columns = pd.read_csv(
    AUTO / "val.csv", nrows=0
).columns.tolist()

auto_columns = [
    c for c in auto_columns
    if (
        c.startswith(("user_views_", "item_views_"))
        and c not in manual
    )
    or c.startswith(("user_ratio_", "item_ratio_"))
]

def load_data(name):
    df = pd.read_csv(GENRE / f"{name}.csv")

    extra = pd.read_csv(
        AUTO / f"{name}.csv",
        usecols=["user_id", "item_id", "label"] + auto_columns
    )

    keys = ["user_id", "item_id", "label"]
    assert df[keys].equals(extra[keys]), "Несовпадение строк!"

    for col in auto_columns:
        df[col] = extra[col].to_numpy(dtype="float32")

    df["genres"] = df["genres"].fillna("unknown").astype(str)

    return df


train = pd.concat(
    [load_data(f"train_{i}") for i in (1, 2, 3)],
    ignore_index=True
)

val = load_data("val")

future = pd.read_csv("data/processed/val.csv")
future = future[future["user_id"].isin(val["user_id"])]

future_counts = future.groupby("user_id").size()

def evaluate(scores, k=10):
    ranked = val[["user_id", "item_id", "label"]].copy()
    ranked["score"] = scores

    ranked = ranked.sort_values(
        ["user_id", "score", "item_id"],
        ascending=[True, False, True],
        kind="mergesort"
    )

    top = ranked.groupby("user_id").head(k).copy()
    top["rank"] = top.groupby("user_id").cumcount() + 1

    hits = top.groupby("user_id")["label"].sum()
    hits = hits.reindex(future_counts.index, fill_value=0)

    recall = (hits / future_counts).mean()

    top["gain"] = top["label"] / np.log2(top["rank"] + 1)

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
    return float(recall), float(ndcg)


counts = [c for c in auto_columns if "_views_" in c]
ratios = [c for c in auto_columns if "_ratio_" in c]

variants = {
    "manual_11": manual,
    "manual_plus_counts": manual + counts,
    "manual_plus_ratios": manual + ratios,
    "manual_plus_all": manual + auto_columns,
}

results = []

for name, features in variants.items():
    print(f"\n=== {name} ===")

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        auto_class_weights="SqrtBalanced",
        random_seed=42,
        verbose=False,
        thread_count=-1
    )

    model.fit(
        train[features], train["label"],
        cat_features=["genres"],
        eval_set=(val[features], val["label"]),
        early_stopping_rounds=40
    )

    scores = model.predict_proba(val[features])[:, 1]
    recall, ndcg = evaluate(scores)

    model.save_model(str(RESULTS / f"{name}.cbm"))

    print(f"Признаков: {len(features)}")
    print(f"Recall@10: {recall:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")

    results.append({
        "model": name,
        "features": len(features),
        "recall": recall,
        "ndcg": ndcg
    })

summary = pd.DataFrame(results)
summary.to_csv(
    RESULTS / "final_feature_comparison.csv",
    index=False
)

print("\n=== СРАВНЕНИЕ ===")
print(summary.to_string(index=False))
print("\nЛучшая по NDCG:")
print(summary.loc[summary["ndcg"].idxmax()])
