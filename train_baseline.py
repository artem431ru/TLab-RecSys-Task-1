
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

DATA = Path("data/processed/ml")

# 1. Объединяем обучающие срезы
train = pd.concat(
    [
        pd.read_csv(DATA / f"train_{i}.csv")
        for i in [1, 2, 3]
    ],
    ignore_index=True,
)

val = pd.read_csv(DATA / "val.csv")

# 2. Признаки для модели
features = [
    "user_views_total",
    "user_views_7d",
    "user_views_30d",
    "item_views_total",
    "item_views_7d",
    "item_views_30d",
    "release_year",
    "genres",
]

for df in [train, val]:
    df["genres"] = df["genres"].fillna("unknown").astype(str)
    for col in features:
        if col != "genres":
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            ).fillna(0)

print("Train:", train.shape)
print("Validation:", val.shape)
print("Положительных в Train:", train["label"].sum())

# 3. Обучаем CatBoost
model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.08,
    loss_function="Logloss",
    auto_class_weights="SqrtBalanced",
    random_seed=42,
    verbose=100,
    thread_count=-1,
)

model.fit(
    train[features],
    train["label"],
    cat_features=["genres"],
    eval_set=(val[features], val["label"]),
    early_stopping_rounds=40,
)

# 4. Получаем предсказания
val["score_catboost"] = model.predict_proba(
    val[features]
)[:, 1]

# Простой baseline по популярности
val["score_popularity"] = val["item_views_total"]

# 5. Сохраняем модель и предсказания
Path("results").mkdir(exist_ok=True)

model.save_model("results/catboost_baseline.cbm")
val.to_csv(
    "results/validation_predictions.csv",
    index=False,
)

# 6. Выводим важность признаков
importance = pd.Series(
    model.get_feature_importance(),
    index=features,
).sort_values(ascending=False)

print("\n=== ВАЖНОСТЬ ПРИЗНАКОВ ===")
print(importance)

print("\nОбучение завершено!")
