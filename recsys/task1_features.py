"""Task 1: CatBoost manual features vs automatically enumerated temporal aggregates."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .common import cutoffs, dump_results, load_kion, ranking_metrics, seed_all, synthetic_events

WINDOWS = (3, 7, 14, 30, 60)
BASE_FEATURES = ['user_count_all', 'item_count_all', 'pair_count_all',
                 'user_count_7d', 'user_count_30d', 'item_count_7d', 'item_count_30d',
                 'pair_count_7d', 'pair_count_30d', 'user_gap_days', 'item_gap_days',
                 'pair_gap_days', 'user_distinct_items', 'item_distinct_users',
                 'pair_share_user_30d']


def feature_frame(history: pd.DataFrame, candidates: pd.DataFrame, asof, windows=WINDOWS):
    """Build features using only history strictly BEFORE asof; no future-based aggregation."""
    out = candidates[['user_id', 'item_id']].copy()
    h = history.loc[history.time < asof]
    definitions = [('user', ['user_id']), ('item', ['item_id']), ('pair', ['user_id', 'item_id'])]
    for prefix, keys in definitions:
        counts = h.groupby(keys).size().rename(f'{prefix}_count_all').reset_index()
        out = out.merge(counts, how='left', on=keys)
    for days in windows:
        hw = h[h.time >= asof - pd.Timedelta(days=days)]
        for prefix, keys in definitions:
            counts = hw.groupby(keys).size().rename(f'{prefix}_count_{days}d').reset_index()
            out = out.merge(counts, how='left', on=keys)
    feature_names = [c for c in out if '_count_' in c]
    out[feature_names] = out[feature_names].fillna(0).astype('float32')
    for prefix, keys in definitions:
        last = h.groupby(keys)['time'].max().rename(f'{prefix}_last').reset_index()
        out = out.merge(last, how='left', on=keys)
        out[f'{prefix}_gap_days'] = ((asof - out[f'{prefix}_last']).dt.total_seconds() / 86400).fillna(9999).clip(lower=0).astype('float32')
        out = out.drop(columns=[f'{prefix}_last'])
    uc = h.groupby('user_id')['item_id'].nunique().rename('user_distinct_items').reset_index()
    ic = h.groupby('item_id')['user_id'].nunique().rename('item_distinct_users').reset_index()
    out = out.merge(uc,how='left',on='user_id').merge(ic,how='left',on='item_id')
    out[['user_distinct_items','item_distinct_users']] = out[['user_distinct_items','item_distinct_users']].fillna(0).astype('float32')
    # Automatically composed ratios in addition to automatically enumerated windows.
    for days in windows:
        out[f'pair_share_user_{days}d'] = (
            out[f'pair_count_{days}d'] / (1 + out[f'user_count_{days}d']))
        out[f'item_share_global_{days}d'] = (
            out[f'item_count_{days}d'] / (1 + len(h[h.time >= asof - pd.Timedelta(days=days)])))
    return out


def make_split(df, start, end, pool_size=100, max_users=2000, seed=42):
    """Time-snapshot candidate universe from PAST popularity only; no positive injection."""
    past = df[df.time < start]
    future = df[(df.time >= start) & (df.time < end)]
    if past.empty or future.empty:
        raise ValueError('Insufficient events for time split')
    catalog = past.groupby('item_id').size().sort_values(ascending=False).head(pool_size).index.to_numpy()
    present = past.user_id.unique()
    future_users = future.user_id.unique()
    valid_users = np.intersect1d(present, future_users)
    if len(valid_users) > max_users:
        valid_users = np.random.default_rng(seed).choice(valid_users, max_users, replace=False)
    # Candidate set does NOT depend on future labels; identical for all users.
    cand = pd.MultiIndex.from_product([valid_users, catalog], names=['user_id', 'item_id']).to_frame(index=False)
    positive = future.drop_duplicates(['user_id', 'item_id'])[['user_id', 'item_id']]
    positive['label'] = 1
    cand = cand.merge(positive, on=['user_id', 'item_id'], how='left')
    cand['label'] = cand['label'].fillna(0).astype('int8')
    # Keep users with a measurable positive among pre-existing candidates.
    keep = cand.groupby('user_id')['label'].transform('sum').gt(0)
    cand = cand[keep].reset_index(drop=True)
    if cand.empty:
        raise ValueError('No positive hits in past-only candidate pool; increase --pool-size / --users')
    features = feature_frame(past, cand, start)
    features['label'] = cand.label.to_numpy()
    return features, dict(candidate_count=len(catalog), users=cand.user_id.nunique(),
                          positive_rate=float(cand.label.mean()), snapshot=str(start))


def catboost_fit(train, val, cols, seed=42, iters=110):
    model = CatBoostClassifier(iterations=iters, depth=5, learning_rate=.08,
                               loss_function='Logloss', eval_metric='AUC',
                               random_seed=seed, thread_count=2, verbose=False,
                               l2_leaf_reg=4)
    model.fit(train[cols], train.label, eval_set=(val[cols], val.label),
              early_stopping_rounds=25, verbose=False)
    return model


def run(df, outdir, pool_size=100, max_users=2000, seed=42, iterations=110, source='kion'):
    seed_all(seed)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    t0, t1, t2 = cutoffs(df)
    splits = []
    for s, e in [(t0, t1), (t1, t2), (t2, df.time.max() + pd.Timedelta(days=1))]:
        split, meta = make_split(df, s, e, pool_size, max_users, seed)
        splits.append(split)
    train, val, test = splits
    allcols = [c for c in train if ('_count_' in c or '_share_' in c or '_gap_days' in c or '_distinct_' in c)]
    results = {'source': source, 'boundaries': [str(x) for x in (t0,t1,t2)],
               'train_users': int(train.user_id.nunique()), 'val_users': int(val.user_id.nunique()),
               'test_users': int(test.user_id.nunique()), 'candidate_policy': 'historical_popular_top_n',
               'sampled_metrics_caveat': 'Only historical top-N items; user subset with positive in pool',
               'experiments': {}}
    for name, cols in [('manual', BASE_FEATURES), ('automatic_grid', allcols)]:
        started = time.time()
        model = catboost_fit(train,val,cols,seed,iters=iterations)
        valpred = model.predict_proba(val[cols])[:,1]
        pred = model.predict_proba(test[cols])[:,1]
        scores = ranking_metrics(test.label, pred, test.user_id)
        scores['val'] = ranking_metrics(val.label,valpred,val.user_id)
        scores['features'] = len(cols)
        scores['wall_seconds_train_and_eval'] = round(time.time()-started,2)
        scores['feature_names'] = cols
        results['experiments'][name] = scores
        pd.DataFrame({'name':cols, 'importance':model.feature_importances_}).sort_values(
            'importance',ascending=False).to_csv(outdir/f'{name}_importance.csv',index=False)
    # Additional validation-only feature selection, without accessing test outcomes.
    importances = pd.read_csv(outdir/'automatic_grid_importance.csv')
    bestcols = importances.head(min(12,len(importances))).name.tolist()
    m = catboost_fit(train,val,bestcols,seed,iters=iterations)
    pred = m.predict_proba(test[bestcols])[:,1]
    results['experiments']['automatic_selected'] = {
        **ranking_metrics(test.label,pred,test.user_id), 'features': len(bestcols),
        'feature_names': bestcols,
        'val': ranking_metrics(val.label,m.predict_proba(val[bestcols])[:,1],val.user_id),
        'note': 'Feature selection uses model importances fitted on train, not test.'}
    dump_results(outdir/'task1_metrics.json',results)
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=str,help='KION interactions.csv(.zip), omit for synthetic smoke test')
    p.add_argument('--users',type=int,default=2000)
    p.add_argument('--pool-size',type=int,default=100)
    p.add_argument('--iterations',type=int,default=110)
    p.add_argument('--out',default='results/task1')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    df=load_kion(args.data,args.users,args.seed) if args.data else synthetic_events()
    r=run(df,args.out,args.pool_size,args.users,args.seed,args.iterations,
          source='kion' if args.data else 'synthetic_test')
    if not args.data:
        print('SYNTHETIC SMOKE TEST ONLY. These are NOT real-data results.')
    for k,v in r['experiments'].items():
        print(k, 'NDCG@10:', v['ndcg@10'], 'Recall@10:', v['recall@10'])

if __name__=='__main__': main()
