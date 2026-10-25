"""Task 2: GPT-2-style causal next-item model, reward model, REINFORCE + KL.

SYNTHETIC mode is a functionality check, NOT scientific results.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import cutoffs, dump_results, load_kion, seed_all, synthetic_events


class CausalGPT2Fallback(nn.Module):
    """Small pre-LN learned-position decoder; mirrors GPT-2 design at architecture level.

    Not a checkpoint-compatible exact implementation of GPT-2. For the literal
    HuggingFace GPT-2 implementation, install transformers and use backend=hf.
    """
    def __init__(self, vocab, seq_len, dim=64, layers=2, heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab,dim)
        self.pos = nn.Embedding(seq_len,dim)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim,heads,dim_feedforward=dim*4,
                                       activation='gelu',batch_first=True,norm_first=True,dropout=0.0),
            num_layers=layers,enable_nested_tensor=False)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim,vocab,bias=False)
        self.head.weight = self.tok.weight

    def forward(self, ids):
        b,t = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(t,device=ids.device)).unsqueeze(0)
        mask = torch.triu(torch.ones(t,t,device=ids.device,dtype=torch.bool),diagonal=1)
        # Left padding: last token always non-PAD; padding is masked as keys.
        x = self.layers(x,mask=mask,src_key_padding_mask=ids.eq(0))
        return self.head(self.ln(x))


class HuggingFaceGPT2(nn.Module):
    def __init__(self,vocab,seq_len,dim=64,layers=2,heads=4):
        super().__init__()
        from transformers import GPT2Config,GPT2LMHeadModel
        config=GPT2Config(vocab_size=vocab,n_positions=seq_len,n_ctx=seq_len,
                          n_embd=dim,n_layer=layers,n_head=heads,
                          resid_pdrop=0.0,embd_pdrop=0.0,attn_pdrop=0.0,
                          pad_token_id=0)
        self.inner=GPT2LMHeadModel(config)
    def forward(self, ids):
        return self.inner(input_ids=ids,attention_mask=ids.ne(0).long()).logits


class RewardModel(nn.Module):
    """Predict observed watched percentage in [0,1] for (past items, next item)."""
    def __init__(self,vocab,dim=48):
        super().__init__()
        self.embed=nn.Embedding(vocab,dim,padding_idx=0)
        self.mlp=nn.Sequential(nn.Linear(dim*2,dim),nn.GELU(),nn.Linear(dim,1),nn.Sigmoid())
    def forward(self,history,item):
        emb=self.embed(history)
        m=history.ne(0).unsqueeze(-1)
        hist=(emb*m).sum(1)/m.sum(1).clamp(min=1)
        target=self.embed(item)
        return self.mlp(torch.cat((hist,target),dim=-1)).squeeze(-1)


def build_examples(df, max_items=1000, seq_len=20, seed=42):
    """Fit vocabulary on *training PAST* only; global temporal split, no future item leakage."""
    t0,t1,t2=cutoffs(df)
    train_past=df[df.time<t1]
    catalog=train_past.item_id.value_counts().head(max_items).index.tolist()
    token={k:i+2 for i,k in enumerate(catalog)}
    # Include ALL chronologically preceding events in each user's prefix; OOV=1.
    rows={'train':[],'val':[],'test':[]}; first_seen={'val':set(),'test':set()}
    for user,g in df.sort_values('time').groupby('user_id'):
        history=[]
        for r in g.itertuples():
            target=token.get(r.item_id,1)
            split=('train' if t0<=r.time<t1 else
                   'val' if t1<=r.time<t2 else
                   'test' if r.time>=t2 else None)
            if split and history and target>=2:
                if split == 'train' or user not in first_seen[split]:
                    padded=([0]*max(0,seq_len-len(history)) + history[-seq_len:])
                    rows[split].append((padded,target,
                          float(np.nan_to_num(r.reward,nan=0.0)),str(user)))
                    if split != 'train': first_seen[split].add(user)
            history.append(target)
    if any(not rows[k] for k in rows):
        raise ValueError(f'Not enough in-vocabulary interactions across all splits: '
                         f'{[(k,len(v)) for k,v in rows.items()]}; '
                         f'raise --max-items or --users')
    return rows,{'vocab':len(token)+2,'catalog_size':len(token),
                 'cutoffs':[str(t0),str(t1),str(t2)],
                 'example_counts':{k:len(v) for k,v in rows.items()},
                 'max_items':max_items,'seq_len':seq_len}


def loader(rows,batch=64,shuffle=False):
    x=torch.tensor([r[0] for r in rows],dtype=torch.long)
    y=torch.tensor([r[1] for r in rows],dtype=torch.long)
    rew=torch.tensor([r[2] for r in rows],dtype=torch.float32)
    return DataLoader(TensorDataset(x,y,rew),batch_size=batch,shuffle=shuffle)


def train_lm(model,train_rows,epochs=2,batch=64,lr=3e-4,device='cpu'):
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    dl=loader(train_rows,batch,True)
    losses=[]
    for e in range(epochs):
        model.train(); vals=[]
        for x,y,_ in dl:
            x=x.to(device);y=y.to(device)
            logits=model(x)[:,-1,:]
            loss=nn.functional.cross_entropy(logits,y)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            vals.append(loss.item())
        losses.append(float(np.mean(vals)))
    return losses


def train_reward(model,rows,epochs=2,batch=64,lr=5e-4,device='cpu'):
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    dl=loader(rows,batch,True)
    losses=[]
    for _ in range(epochs):
        model.train();vals=[]
        for x,y,r in dl:
            x=x.to(device);y=y.to(device);r=r.to(device)
            loss=nn.functional.mse_loss(model(x,y),r)
            opt.zero_grad();loss.backward();opt.step();vals.append(loss.item())
        losses.append(float(np.mean(vals)))
    return losses


@torch.no_grad()
def evaluate(lm,reward_model,rows,k=10,device='cpu',batch=64,popularity=None):
    lm.eval();reward_model.eval()
    recalls=[]; ndcgs=[]; rewards=[]; mse=[]; pop_recs=[]; top_items=set(); total_recs=0
    for x,y,r in loader(rows,batch):
        x=x.to(device); y=y.to(device);r=r.to(device)
        logits=lm(x)[:,-1,:].clone(); logits[:,:2]=-1e9
        top=logits.topk(min(k,logits.size(1)-2),dim=1).indices
        good=top.eq(y[:,None]);found=good.any(1).float()
        recalls.extend(found.tolist())
        for a in good.cpu().numpy():
            loc=np.where(a)[0]
            ndcgs.append(float(1/np.log2(loc[0]+2)) if len(loc) else 0.)
        for j in range(top.shape[1]):
            rewards.extend(reward_model(x,top[:,j]).cpu().tolist())
        mse.extend((reward_model(x,y)-r).square().cpu().tolist())
        if popularity is not None:
            pop_recs.extend(popularity[top.cpu().flatten()].tolist())
        top_items.update(top.cpu().flatten().tolist());total_recs+=top.numel()
    return {'n_examples':len(rows),f'recall@{k}':float(np.mean(recalls)),
            f'ndcg@{k}':float(np.mean(ndcgs)),
            'predicted_reward_topk_mean':float(np.mean(rewards)),
            'observed_reward_mse':float(np.mean(mse)),
            'catalog_coverage_topk':len(top_items)/max(1,lm(torch.tensor([rows[0][0]],device=device))[:,-1].shape[-1]-2),
            'distinct_recommendations':len(top_items),
            'mean_normalized_item_popularity_topk':float(np.mean(pop_recs)) if pop_recs else None}


def rl_finetune(lm,reference,reward_model,train_rows,steps=40,batch=64,
                beta=.05,lr=1e-4,device='cpu',popularity=None,novelty_weight=0.):
    """One-step self-critical REINFORCE + exact token KL to frozen initial policy."""
    lm.train();reference.eval();reward_model.eval()
    for p in reference.parameters():p.requires_grad_(False)
    for p in reward_model.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW(lm.parameters(),lr=lr)
    rng=np.random.default_rng(42)
    traces=[]
    for step in range(steps):
        idx=rng.choice(len(train_rows),min(batch,len(train_rows)),replace=False)
        states=torch.tensor([train_rows[i][0] for i in idx],device=device,dtype=torch.long)
        logits=lm(states)[:,-1,:].clone();logits[:,:2]=-1e9
        with torch.no_grad():
            old=reference(states)[:,-1,:].clone();old[:,:2]=-1e9
        policy=torch.distributions.Categorical(logits=logits)
        action=policy.sample()
        with torch.no_grad():
            reward=reward_model(states,action)
        shaped=reward
        if novelty_weight and popularity is not None:
            # Penalize popularity estimated ONLY from training events.
            shaped=reward-novelty_weight*popularity[action].to(device)
        advantage=(shaped-shaped.mean()).detach()
        p_log=torch.log_softmax(logits,dim=1)
        q_log=torch.log_softmax(old,dim=1)
        kl=(p_log.exp()*(p_log-q_log)).sum(1).mean()
        loss=-(advantage*policy.log_prob(action)).mean() + beta*kl
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(lm.parameters(),1.0);opt.step()
        traces.append({'step':step+1,'policy_loss':float(loss.item()),
                       'pred_reward':float(reward.mean().item()),
                       'shaped_reward':float(shaped.mean().item()),'kl':float(kl.item())})
    return traces


def run(df,outdir='results/task2',max_items=500,seq_len=16,epochs=2,
        rl_steps=40,batch=64,dim=64,layers=2,heads=4,beta=.05,backend='auto',seed=42,
        source='kion',novelty_weight=0.):
    seed_all(seed)
    torch.set_num_threads(min(2,torch.get_num_threads()))
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    rows,meta=build_examples(df,max_items,seq_len,seed)
    if backend=='hf' or backend=='auto':
        try:
            import transformers
            cls=HuggingFaceGPT2
            backend_used='huggingface_gpt2'
        except ImportError:
            if backend=='hf':raise ImportError('Install transformers or select --backend fallback')
            cls=CausalGPT2Fallback
            backend_used='pytorch_gpt2_like_fallback'
    else:
        cls=CausalGPT2Fallback;backend_used='pytorch_gpt2_like_fallback'
    lm=cls(meta['vocab'],seq_len,dim,layers,heads).to(dev)
    rm=RewardModel(meta['vocab'],dim).to(dev)
    counts=np.bincount([r[1] for r in rows['train']],minlength=meta['vocab'])
    pop=np.log1p(counts.astype('float64'))
    pop=pop/max(pop.max(),1.)
    popularity=torch.tensor(pop,dtype=torch.float32,device=dev)
    ce=train_lm(lm,rows['train'],epochs,batch,device=dev)
    mse_train=train_reward(rm,rows['train'],epochs,batch,device=dev)
    base_val=evaluate(lm,rm,rows['val'],device=dev,popularity=popularity.cpu())
    base_test=evaluate(lm,rm,rows['test'],device=dev,popularity=popularity.cpu())
    ref=copy.deepcopy(lm).to(dev)
    traces=rl_finetune(lm,ref,rm,rows['train'],rl_steps,batch,beta,device=dev,
                       popularity=popularity,novelty_weight=novelty_weight)
    tuned_val=evaluate(lm,rm,rows['val'],device=dev,popularity=popularity.cpu())
    tuned_test=evaluate(lm,rm,rows['test'],device=dev,popularity=popularity.cpu())
    results={'source':source,**meta,'backend':backend_used,'device':dev,
             'novelty_weight':novelty_weight,'kl_beta':beta,
             'lm_train_ce':ce,'reward_train_mse':mse_train,
             'reference':{'validation':base_val,'test':base_test},
             'after_rl':{'validation':tuned_val,'test':tuned_test},
             'rl_last':traces[-1] if traces else None,
             'limitations':['Predicted reward is a model proxy, NOT user satisfaction',
               'The reward model is only trained on historically observed actions',
               'Offline policy evaluation is confounded by exposure and distribution shift',
               'GPT-2-like fallback is not checkpoint-compatible with canonical GPT-2']}
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    dump_results(outdir/'task2_metrics.json',results)
    dump_results(outdir/'rl_trace.json',traces)
    torch.save({'lm':lm.cpu().state_dict(),'reward_model':rm.cpu().state_dict(),
                'configuration':meta,'backend':backend_used},outdir/'checkpoint.pt')
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',help='KION interactions.csv(.zip); omitted = synthetic smoke test')
    p.add_argument('--users',type=int,default=2500)
    p.add_argument('--max-items',type=int,default=500)
    p.add_argument('--seq-len',type=int,default=16)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--rl-steps',type=int,default=40)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--dim',type=int,default=64)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--beta',type=float,default=.05)
    p.add_argument('--novelty-weight',type=float,default=0.,
                   help='Optional popularity penalty in RL reward; 0 means observed watch proxy only')
    p.add_argument('--backend',choices=['auto','hf','fallback'],default='auto')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',default='results/task2')
    a=p.parse_args()
    df=load_kion(a.data,a.users,a.seed) if a.data else synthetic_events(80,42,14)
    r=run(df,a.out,a.max_items,a.seq_len,a.epochs,a.rl_steps,a.batch,
          a.dim,a.layers,a.heads,a.beta,a.backend,a.seed,
          source='kion' if a.data else 'synthetic_test',novelty_weight=a.novelty_weight)
    if not a.data:print('SYNTHETIC SMOKE TEST ONLY. Metrics are NOT KION research results.')
    for key in ['reference','after_rl']:
        print(key,r[key]['test'])

if __name__=='__main__':main()
