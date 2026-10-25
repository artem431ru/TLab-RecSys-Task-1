
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier

DATA = Path("data/processed/ml_auto")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Загружаем те же обучающие периоды
train = pd.concat(
    [
        pd.read_csv(DATA / f"train_{i}.csv")
        for i in [1, 2, 3]
    ],
    ignore_index=True
)

val = pd.read_csv(DATA / "val.csv")

# Ручные признаки первого CatBoost
manual_features = [
    "user_views_total",
    "user_views_7d",
    "user_views_30d",
    "item_views_total",
    "item_views_7d",
    "item_views_30d",
    "release_year",
    "genres",
]

# Автоматически сгенерированные признаки
auto_features = [
    c for c in train.columns
    if (
        c.startswith(("user_views_", "item_views_"))
        and c not in manual_features
    )
    or c.startswith(("user_ratio_", "item_ratio_"))
]

features = manual_features + auto_features

print("Всего признаков:", len(features))
print("Новых признаков:", len(auto_features))
print("Новые признаки:", auto_features)

for frame in [train, val]:
    frame["genres"] = (
        frame["genres"].fillna("unknown").astype(str)
    )

    for col in features:
        if col != "genres":
            frame[col] = pd.to_numeric(
                frame[col], errors="coerce"
            ).fillna(0).astype("float32")

# Те же параметры, что у первой модели
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

print("\nОбучаем модель с автоматическими признаками...")

model.fit(
    train[features],
    train["label"],
    cat_features=["genres"],
    eval_set=(val[features], val["label"]),
    early_stopping_rounds=40,
)

# Предсказания второй модели
val["score_auto"] = model.predict_proba(
    val[features]
)[:, 1]

model.save_model(str(RESULTS / "catboost_auto.cbm"))

# Сравниваем с сохранёнными результатами
# первого CatBoost на тех же парах user-item
previous = pd.read_csv(
    RESULTS / "validation_predictions.csv"
)

keys = ["user_id", "item_id", "label"]

assert previous[keys].equals(val[keys]), (
    "Строки Validation не совпадают!"
)

val["score_catboost"] = previous[
    "score_catboost"
].to_numpy()

val["score_popularity"] = previous[
    "score_popularity"
].to_numpy()

val.to_csv(
    RESULTS / "validation_compare.csv",
    index=False
)

# Важность признаков
importance = pd.Series(
    model.get_feature_importance(),
    index=features
).sort_values(ascending=False)

print("\n=== ВАЖНОСТЬ ПРИЗНАКОВ ===")
print(importance.head(20))

print("\nЛучшая итерация:", model.best_iteration_)
print("Готово!")
