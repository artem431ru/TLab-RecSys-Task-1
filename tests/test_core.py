from recsys.common import synthetic_events,cutoffs,ranking_metrics
from recsys.task1_features import feature_frame,make_split
from recsys.task2_rl import build_examples
import pandas as pd


def test_no_future_leakage_features():
    df=synthetic_events(12,18,14)
    a,b,c=cutoffs(df)
    future=df[df.time>=b]
    base=df[df.time<b]
    cand=pd.DataFrame({'user_id':['0','1'],'item_id':['0','1']})
    x=feature_frame(df,cand,b)
    y=feature_frame(base,cand,b)
    assert x.equals(y)


def test_candidate_set_uses_only_history():
    df=synthetic_events(30,24,16)
    a,b,c=cutoffs(df)
    x,meta=make_split(df,a,b,pool_size=15)
    assert len(x)>0 and 0<x.label.mean()<1


def test_gpt_data_has_train_val_test():
    df=synthetic_events(30,24,16)
    rows,meta=build_examples(df,max_items=24)
    assert all(rows[k] for k in ('train','val','test'))
    assert all(len(r[0])==meta['seq_len'] for r in rows['train'])


def test_ndcg_is_one_for_ranked_top():
    m=ranking_metrics([1,0,0],[3,2,1],['u','u','u'])
    assert m['ndcg@10']==1 and m['recall@10']==1
