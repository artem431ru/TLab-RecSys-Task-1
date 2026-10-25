"""Common KION ingestion, synthetic smoke dataset, reproducible time splits and metrics."""
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_kion(path: str | Path, users: int = 0, seed: int = 42) -> pd.DataFrame:
    """Read official interactions.csv or zip containing it; sample USERS, retain ALL their rows."""
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            matches = [x for x in z.namelist() if x.lower().endswith('interactions.csv')]
            if not matches:
                raise ValueError(f'No interactions.csv in {path}; found {z.namelist()[:8]}')
            with z.open(matches[0]) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)
    rename = {'user': 'user_id', 'item': 'item_id', 'timestamp': 'last_watch_dt'}
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    required = {'user_id', 'item_id', 'last_watch_dt'}
    if not required.issubset(df):
        raise ValueError(f'Missing {required - set(df)} in {list(df.columns)}')
    df['time'] = pd.to_datetime(df['last_watch_dt'], errors='coerce')
    df = df.dropna(subset=['time', 'user_id', 'item_id']).copy()
    if users and df['user_id'].nunique() > users:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(df.user_id.unique(), users, replace=False)
        df = df.loc[df.user_id.isin(chosen)].copy()
    # Restrict pct to [0,100], not for prediction features: used solely as observed reward label.
    if 'watched_pct' in df:
        df['reward'] = np.clip(pd.to_numeric(df['watched_pct'], errors='coerce') / 100.0, 0, 1)
    elif 'total_dur' in df:
        # Weak proxy when watched_pct absent; document this limitation.
        dur = pd.to_numeric(df['total_dur'], errors='coerce')
        df['reward'] = np.clip(dur / 3600.0, 0, 1)
    else:
        df['reward'] = np.nan
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df = df.sort_values(['time', 'user_id', 'item_id']).reset_index(drop=True)
    return df[['user_id', 'item_id', 'time', 'reward']].copy()


def synthetic_events(n_users: int = 120, n_items: int = 48, events_per_user: int = 14,
                     seed: int = 42) -> pd.DataFrame:
    """Small artificial dataset ONLY for testing functionality; never report its metrics as KION."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp('2025-01-01')
    for u in range(n_users):
        fav = int(rng.integers(0, 6))
        for t in range(events_per_user):
            group = fav if rng.random() < 0.70 else int(rng.integers(0, 6))
            idx = group * max(1, n_items // 6) + int(rng.integers(0, max(1, n_items // 6)))
            idx %= n_items
            dt = t0 + pd.Timedelta(days=float(t * 2 + rng.uniform(0, 2)))
            reward = float(np.clip(0.65 + (group == fav) * .25 + rng.normal(0, .12), 0, 1))
            rows.append((str(u), str(idx), dt, reward))
    return pd.DataFrame(rows, columns=['user_id', 'item_id', 'time', 'reward']).sort_values('time').reset_index(drop=True)


def cutoffs(df: pd.DataFrame):
    """Single global chronological intervals: history / train / val / test."""
    times = df['time'].sort_values().reset_index(drop=True)
    return [times.iloc[min(len(times) - 1, int(q * (len(times) - 1)))] for q in (.40, .70, .85)]


def ranking_metrics(labels, scores, groups, k: int = 10) -> dict:
    d = pd.DataFrame({'label': np.asarray(labels), 'score': np.asarray(scores), 'group': np.asarray(groups)})
    rec, ndcg = [], []
    for _, f in d.groupby('group'):
        positives = float(f.label.sum())
        if positives <= 0:
            continue
        top = f.sort_values('score', ascending=False).head(k)
        hits = top.label.to_numpy(dtype=float)
        rec.append(float(hits.sum() / positives))
        dcg = float(np.sum(hits / np.log2(np.arange(2, len(hits) + 2))))
        ideal = min(int(positives), k)
        idcg = float(np.sum(1 / np.log2(np.arange(2, ideal + 2))))
        ndcg.append(dcg / idcg)
    return {'users': len(ndcg), f'recall@{k}': float(np.mean(rec)) if rec else None,
            f'ndcg@{k}': float(np.mean(ndcg)) if ndcg else None}


def dump_results(path, metrics):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
