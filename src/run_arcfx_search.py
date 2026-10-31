"""Bounded, resumable chronological ARC_FX exploration. See configs/arcfx_search.json.

All proposed losses use raw-return labels. Original files/results are never overwritten.
Report-period predictions are only generated after development selection is frozen.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from data_pipeline import prepare_data, set_seed
from arcfx_search_models import SearchModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/arcfx_search'
CCY = ['USD','EUR','JPY','GBP','CAD','AUD','KRW','CHF','NZD','SEK','NOK']
CONFIG = dict(epochs=20, patience=4, batch_size=256, lr=.0003, weight_decay=.0001,
              selection='validation sign-hit (zeros retained); tie: raw-label BCE',
              max_training_runs=60, threads=2, lookback=10, bootstrap_repeats=2000,
              development_fractions=[[.5,.6],[.6,.7]], final_train_fraction=.7,
              final_validation_end=.8, test_status='previously inspected; not untouched')


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    temporary.replace(path)


def event(**row):
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(time=time.time(),**row),default=str)+'\n')


def hit(z,y):
    # Retain the historical sign-hit definition; zero targets remain in denominator.
    if np.size(y)==0: return float('nan')
    return float((np.sign(z)==np.sign(y)).mean())


class Fold:
    def __init__(self,name,train_end,val_end,report_end=None):
        self.name=name; self.train_end=train_end; self.val_end=val_end
        # One purged target: last training forward return must finish before validation decision.
        self.scale_end=train_end-1
        self.p=prepare_data(str(ROOT/'data/processed/fx_panel_pit.csv'),
                            str(ROOT/'data/processed/nonfx_features_pit.csv'),CCY,10,
                            [self.scale_end/N,.1,.1],include_regime_onehot=True)
        p=self.p
        assert int(len(p.merged)*self.scale_end/N)==self.scale_end
        self.blocks={}
        for key in ['local','rate','equity','countrymacro','global']:
            arr=getattr(p,'x_'+key)
            # window[t] includes decision-date t; legacy uses window[t-1].
            windows=np.lib.stride_tricks.sliding_window_view(arr,10,axis=0)
            windows=np.moveaxis(windows,-1,1).copy()
            self.blocks['x_'+key]=torch.from_numpy(windows)
        self.y=torch.from_numpy(p.y_raw)
        self.yn=torch.from_numpy(p.y_norm)
        self.train=np.arange(10,self.scale_end)
        self.val=np.arange(train_end,val_end)
        self.report=np.arange(val_end,report_end) if report_end else np.array([],dtype=int)
        self.scalar=float(np.std(p.y_raw[self.train,1:]))
        self.threshold=float(np.quantile(np.abs(p.y_norm[self.train,1:]),.4))
        self.q80=float(np.quantile(np.abs(p.y_raw[self.train,1:]),.8))
        self.stress={}
        for name,values in [('vix',p.merged.Global_VIX_change.abs().values),
                            ('fx',np.abs(p.y_raw[:,1:]).mean(1)),
                            ('yield',p.merged.Global_US10Y_change.values)]:
            self.stress[name]=values>=np.nanquantile(values[self.train],.85)

    def batch(self,ix,legacy=False):
        w=ix-9-int(legacy)
        return {k:v[w] for k,v in self.blocks.items()}


def loss_fn(model,z,r,ix,fold,spec):
    y=fold.y[ix,1:]; zn=z[:,1:]
    kind=spec['loss']; terms={}
    if kind in ['legacy','active']:
        yn=fold.yn[ix,1:]; active=yn.abs()>=fold.threshold
        loss=F.softplus(-zn[active]*yn[active].sign()).mean()
        terms['active_fraction']=float(active.float().mean())
        if kind=='legacy':
            o=model.last
            terms['component']=sum(v.square().mean() for v in o['components'].values())*.0001
            for k,weight in [('smoothness',.001),('static_deviation',.001),('sparsity',.0001),('spectral',.001)]:
                terms[k]=o['regularization'][k]*weight
            loss=loss+sum(v for k,v in terms.items() if k!='active_fraction')
    else:
        loss=F.binary_cross_entropy_with_logits(zn,(y>0).float())
        terms['active_fraction']=1.
        if kind.startswith('dual'):
            # Single train-only scalar, no currency-dependent centering/scaling: preserves differences.
            aux=F.huber_loss(r[:,1:],y/fold.scalar,delta=1.)
            terms['return_aux']=float(kind.split('_')[1])*aux
            loss=loss+terms['return_aux']
    return loss,{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in terms.items()}


@torch.no_grad()
def predict(model,fold,ix,spec,ablate=False):
    model.eval(); zs=[]; rs=[]; diagnostics=[]; adj=[]
    for start in range(0,len(ix),512):
        batch=fold.batch(ix[start:start+512],spec.get('legacy_alignment',False))
        z,r=model(batch,zero_graph=ablate)
        zs.append(z[:,1:].numpy())
        if r is not None: rs.append(r[:,1:].numpy()*fold.scalar)
        o=model.last; a=o['adj'].numpy(); adj.append(a)
        g=o['graph_gate'].numpy(); e=o['edge_logits'].numpy(); e=e[np.isfinite(e)]
        diagnostics.append(dict(n=len(z),gate_mean=float(g.mean()),gate_min=float(g.min()),gate_max=float(g.max()),
                                gate_saturated=float(((g<.01)|(g>.99)).mean()),edge_active=float((a!=0).mean()),
                                edge_score_std=float(e.std()),message_rms=float(o.get('message',o['components']['rel']).square().mean().sqrt())))
    logits=np.concatenate(zs)
    pred=logits
    ret=np.concatenate(rs) if rs else None
    if spec['loss'] in ['legacy','active']:
        ret=logits*fold.p.y_std[1:]+fold.p.y_mean[1:]
        pred=ret
    y=fold.p.y_raw[ix,1:]
    bce=float(np.mean(np.logaddexp(0,logits)-logits*(y>0)))
    metrics=dict(mean_hit=hit(pred,y),bce=bce,pred_positive=float((pred>0).mean()),
                 actual_positive=float((y>0).mean()),zero_target=float((y==0).mean()),zero_pred=float((pred==0).mean()),
                 logit_mean=float(logits.mean()),logit_std=float(logits.std()),
                 extreme_hit=hit(pred[np.abs(y)>=fold.q80],y[np.abs(y)>=fold.q80]))
    for key in diagnostics[0]:
        if key!='n': metrics[key]=float(np.average([d[key] for d in diagnostics],weights=[d['n'] for d in diagnostics]))
    aa=np.concatenate(adj)!=0
    metrics['edge_turnover']=float(np.logical_xor(aa[1:],aa[:-1]).sum()/max(1,np.logical_or(aa[1:],aa[:-1]).sum()))
    stress=[]; extreme=[]
    for name,flags in fold.stress.items():
        mask=flags[ix]
        if mask.any():
            metrics['stress_'+name]=hit(pred[mask],y[mask]);stress.append(metrics['stress_'+name])
            ex=np.abs(y[mask])>=fold.q80
            extreme.append(hit(pred[mask][ex],y[mask][ex]))
    metrics['stress_mean_hit']=float(np.mean(stress));metrics['stress_extreme_hit']=float(np.mean(extreme))
    if ret is not None:
        metrics['return_rmse']=float(np.sqrt(np.mean((ret-y)**2)))
        metrics['return_head_hit']=hit(ret,y)
    return metrics,logits,pred,ret


def gradients(model):
    vals={'encoder':0.,'graph':0.,'head':0.}
    for name,p in model.named_parameters():
        if p.grad is None: continue
        group='encoder' if 'encoder' in name else 'graph' if any(k in name for k in ['Wq','Wk','Wv','edge','gate','q.','k.','value','correction','anchor','rank']) else 'head'
        vals[group]+=float(p.grad.square().sum())
    return {k+'_grad_norm':v**.5 for k,v in vals.items()}


def save_predictions(path,fold,ix,spec,seed,pack):
    metrics,z,pred,ret=pack
    frame=pd.DataFrame(dict(decision_date=np.repeat(fold.p.merged.Date.iloc[ix].astype(str).values,10),
                            target_date=np.repeat(fold.p.merged.Target_Date.iloc[ix].astype(str).values,10),
                            currency=np.tile(CCY[1:],len(ix)),target=fold.p.y_raw[ix,1:].ravel(),
                            logit=z.ravel(),pred=pred.ravel(),seed=seed))
    frame['label']=(frame.target>0).astype(int)
    if ret is not None: frame['return_pred']=ret.ravel()
    frame.to_parquet(path,index=False)


def run(spec,fold,seed,report=False):
    key=f"{spec['name']}__{fold.name}__s{seed}"; dest=OUT/'runs'/key
    if (dest/'result.json').exists(): return json.loads((dest/'result.json').read_text())
    dest.mkdir(parents=True,exist_ok=True)
    ledger=[json.loads(s) for s in (OUT/'ledger.jsonl').read_text().splitlines()] if (OUT/'ledger.jsonl').exists() else []
    if sum(r.get('status')=='started' for r in ledger)>=CONFIG['max_training_runs']: raise RuntimeError('training budget reached')
    event(run=key,status='started',spec=spec,seed=seed,fold=fold.name)
    set_seed(seed); model=SearchModel(fold.p.dims,CCY,spec)
    opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    best=(-float('inf'),-float('inf')); best_state=None; noimp=0; history=[]; start=time.perf_counter()
    checkpoint=dest/'resume.pt'; first=1
    if checkpoint.exists():
        ck=torch.load(checkpoint,weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['opt']);best=ck['best'];best_state=ck['best_state']
        noimp=ck['noimp'];history=ck['history'];first=ck['epoch']+1
        torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng'])
    try:
        for epoch in range(first,CONFIG['epochs']+1):
            model.train(); order=np.random.permutation(fold.train); losses=[]; previous=None
            for i in range(0,len(order),CONFIG['batch_size']):
                ix=order[i:i+CONFIG['batch_size']];batch=fold.batch(ix,spec.get('legacy_alignment',False))
                if previous is not None and spec['loss']=='legacy': batch['A_prev']=previous
                z,r=model(batch); loss,terms=loss_fn(model,z,r,ix,fold,spec)
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                opt.zero_grad();loss.backward();grad=gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                previous=model.last['adj'].detach();losses.append(float(loss.detach()))
            vm=predict(model,fold,fold.val,spec)[0]
            tm=predict(model,fold,fold.train,spec)[0]
            history.append(dict(epoch=epoch,train_loss=np.mean(losses),train_hit=tm['mean_hit'],
                                val_hit=vm['mean_hit'],val_bce=vm['bce'],**terms,**grad))
            score=(vm['mean_hit'],-vm['bce'])
            if score>best:
                best=score;best_state=copy.deepcopy(model.state_dict());noimp=0;best_epoch=epoch
            else:noimp+=1
            temporary=checkpoint.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),best=best,best_state=best_state,
                            noimp=noimp,history=history,epoch=epoch,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state()),temporary)
            temporary.replace(checkpoint)
            pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
            if noimp>=CONFIG['patience']: break
        model.load_state_dict(best_state);torch.save(best_state,dest/'best.pt')
        result=dict(name=spec['name'],spec=spec,seed=seed,fold=fold.name,epochs=len(history),
                    best_epoch=max(history,key=lambda x:(x['val_hit'],-x['val_bce']))['epoch'],
                    seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),
                    train_start=str(fold.p.merged.Date.iloc[fold.train[0]]),train_end=str(fold.p.merged.Date.iloc[fold.train[-1]]),
                    validation_start=str(fold.p.merged.Date.iloc[fold.val[0]]),validation_end=str(fold.p.merged.Date.iloc[fold.val[-1]]))
        for name,ix in [('train',fold.train),('validation',fold.val)]+([('report',fold.report)] if report else []):
            pack=predict(model,fold,ix,spec);result[name]=pack[0]
            save_predictions(dest/(name+'.parquet'),fold,ix,spec,seed,pack)
            if name!='train':
                ab=predict(model,fold,ix,spec,True)
                result[name]['zero_graph_hit']=ab[0]['mean_hit']
                result[name]['zero_graph_abs_logit_change']=float(np.abs(pack[1]-ab[1]).mean())
                result[name]['zero_graph_sign_disagreement']=float((np.sign(pack[2])!=np.sign(ab[2])).mean())
        dump(dest/'result.json',result);event(run=key,status='completed',seconds=result['seconds'])
        print(key,'val',round(result['validation']['mean_hit'],5),'seconds',round(result['seconds']),flush=True)
        return result
    except Exception:
        event(run=key,status='failed',traceback=traceback.format_exc());raise


def audit():
    p=prepare_data(str(ROOT/'data/processed/fx_panel_pit.csv'),str(ROOT/'data/processed/nonfx_features_pit.csv'),CCY,10,[.6,.2,.2],True)
    mismatch=((p.y_raw[:,1:]>0)!=(p.y_norm[:,1:]>0))
    rows=[]; paired=[]
    for seed in [21,42,456]:
        frames=[]
        for variant in ['pit_main','pit_arcfx_active_core']:
            path=ROOT/f'results/{variant}/runs/arc_fx/oursmain_seed{seed}/predictions/oursmain_predictions.parquet'
            f=pd.read_parquet(path);f=f[f.currency!='USD'].copy();f['variant']=variant
            f['year']=pd.to_datetime(f.target_date).dt.year
            for year,g in [('ALL',f)]+list(f.groupby('year')):
                rows.append(dict(variant=variant,seed=seed,year=year,n=len(g),hit=hit(g.pred.values,g.target.values),
                                 pred_positive=float((g.pred>0).mean()),actual_positive=float((g.target>0).mean()),pred_std=float(g.pred.std())))
            frames.append(f)
        a=frames[0].merge(frames[1],on=['target_date','currency'],suffixes=('_a','_b'),validate='one_to_one')
        ca=ROOT/f'results/pit_main/runs/arc_fx/oursmain_seed{seed}/checkpoints/oursmain_best.pt'
        cb=ROOT/f'results/pit_arcfx_active_core/runs/arc_fx/oursmain_seed{seed}/checkpoints/oursmain_best.pt'
        paired.append(dict(seed=seed,n=len(a),sign_disagreement=float((np.sign(a.pred_a)!=np.sign(a.pred_b)).mean()),
                           predictions_exactly_equal=bool(np.array_equal(a.pred_a,a.pred_b)),
                           checkpoint_hash_equal=hashlib.sha256(ca.read_bytes()).hexdigest()==hashlib.sha256(cb.read_bytes()).hexdigest()))
    pd.DataFrame(rows).to_csv(OUT/'historical_seed_year.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'historical_disagreement.csv',index=False)
    pd.DataFrame(dict(currency=CCY[1:],raw_positive=(p.y_raw[:,1:]>0).mean(0),
                      centered_positive=(p.y_norm[:,1:]>0).mean(0),label_disagreement=mismatch.mean(0),
                      train_target_mean=p.y_mean[1:],train_target_std=p.y_std[1:])).to_csv(OUT/'label_audit.csv',index=False)
    dump(OUT/'audit.json',dict(rows=len(p.merged),first=str(p.merged.Date.min()),last=str(p.merged.Date.max()),
         raw_centered_label_disagreement=float(mismatch.mean()),historical_comparison=paired,
         timing='Legacy window ends t-1, label is return t to t+1, stored target_date is t. New window ends t, target_date is Target_Date.',
         scaler='Existing scaler ends floor(N*.6); sequence split trains slightly further, so conservative, not future leakage.',
         smoothness='Legacy shuffled minibatch A_prev compares unrelated dates, not temporal neighbors.',
         pit_audit=pd.read_csv(ROOT/'data/processed/availability_audit.csv').filter(like='violations').sum().to_dict()))
    return len(p.merged)


def smoke(fold):
    path=OUT/'smoke.json'
    if path.exists():return
    spec=dict(name='smoke',encoder='gru',loss='bce',graph='topk')
    set_seed(0);m=SearchModel(fold.p.dims,CCY,spec);opt=torch.optim.Adam(m.parameters(),lr=.01)
    ix=fold.train[:16]; start=predict(m,fold,ix,spec)[0];initial=copy.deepcopy(start)
    for _ in range(250):
        m.train();z,r=m(fold.batch(ix));loss,_=loss_fn(m,z,r,ix,fold,spec)
        opt.zero_grad();loss.backward();opt.step()
    finish=predict(m,fold,ix,spec)[0]
    assert finish['mean_hit']>.9,finish
    b=fold.batch(ix);m.eval();z,_=m(b,zero_graph=True)
    assert float(m.last['components']['rel'].abs().max())==0.
    assert float(z[:,0].abs().max())==0.
    dump(path,dict(initial=initial,final=finish,steps=250,excluded_from_training_budget=True))


def rank(specs,folds,seeds=(0,)):
    rows=[]
    for spec in specs:
        rr=[run(spec,f,s) for f in folds for s in seeds]
        vals=[r['validation']['mean_hit'] for r in rr]
        rows.append((float(np.mean(vals)),float(np.min(vals)),spec))
    return [x[2] for x in sorted(rows,key=lambda x:(x[0],x[1]),reverse=True)]


def main():
    global N
    ap=argparse.ArgumentParser();ap.add_argument('--audit-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(CONFIG['threads'])
    dump(ROOT/'configs/arcfx_search.json',CONFIG)
    N=audit()
    if args.audit_only:return
    folds=[Fold('dev1',int(.5*N),int(.6*N)),Fold('dev2',int(.6*N),int(.7*N))]
    smoke(folds[0])
    legacy=dict(name='original_legacy',encoder='original',loss='legacy',legacy_alignment=True)
    losses=[legacy,dict(name='aligned_active',encoder='original',loss='active'),
            dict(name='all_bce',encoder='original',loss='bce'),
            dict(name='dual_003',encoder='original',loss='dual_0.03'),
            dict(name='dual_010',encoder='original',loss='dual_0.1')]
    ranked=rank(losses,folds)
    # Preserve raw-label objective in new model candidates, even if legacy ranks first.
    bestloss=next(s for s in ranked if s['loss'] not in ['legacy','active'])
    encoders=[dict(bestloss,name='compact_gru',encoder='gru',graph='topk'),
              dict(bestloss,name='temporal_mlp',encoder='mlp',graph='topk'),
              dict(bestloss,name='original_residual',residual=True)]
    ranked2=rank([bestloss]+encoders,folds)
    # Graph modes apply to compact backbone; compare its best encoder with original winner.
    compact=next(s for s in ranked2 if s['encoder']!='original')
    graphs=[dict(compact,name='dense_signed',graph='dense'),dict(compact,name='soft_threshold',graph='soft'),
            dict(compact,name='static_qk',anchor=True),dict(compact,name='lowdim_message',lowdim=True)]
    ranked3=rank([ranked2[0],compact]+graphs,folds)
    allspec=list({s['name']:s for s in losses+encoders+graphs}.values())
    top=rank(allspec,folds)[:2]
    finalists=list({s['name']:s for s in top+[legacy]}.values())
    ranked_final=rank(finalists,folds,(0,1,2));winner=ranked_final[0]
    for f in folds:
        for seed in [3,4]:run(winner,f,seed)
    dump(OUT/'selection_frozen.json',dict(winner=winner,finalists=finalists,rule=CONFIG['selection'],
          note='Retrospective development selection; dev intervals are not unbiased forward strategy performance. No report results used.'))
    # Exact historical report decision rows so archived predictions can be paired.
    report_start=10+int((N-10)*.6)+int((N-10)*.2)
    finalfold=Fold('final',int(.7*N),report_start,N)
    for spec in finalists:
        for seed in ([0,1,2,3,4] if spec['name']==winner['name'] else [0,1,2]):run(spec,finalfold,seed,True)
    # Inference intervention already saved for all; retrained direct-only diagnostic for winner.
    for seed in [0,1,2]:run(dict(winner,name='winner_no_graph',no_graph=True),finalfold,seed,True)
    baselines(finalfold)
    summarize(winner)


def baselines(fold):
    for name in ['majority','logistic']:
        path=OUT/(name+'_report.parquet')
        if path.exists():continue
        y=fold.p.y_raw[:,1:]; ix=fold.report
        if name=='majority':z=np.tile(np.where((y[fold.train]>0).mean(0)>.5,1.,-1.),(len(ix),1))
        else:
            # Shared inputs at decision date, per-currency fixed-regularization classifier.
            p=fold.p; x=np.concatenate([p.x_local,p.x_rate,p.x_equity,p.x_countrymacro,
                                      np.repeat(p.x_global[:,None,:],11,1)],-1)
            z=np.zeros((len(ix),10))
            for j in range(10):
                clf=LogisticRegression(C=.1,max_iter=1000)
                clf.fit(x[fold.train,j+1],y[fold.train,j]>0)
                z[:,j]=clf.decision_function(x[ix,j+1])
        save_predictions(path,fold,ix,{},-1,({},z,z,None))
    event(status='baseline_complete',neural_fits=0,logistic_fits=10,majority_estimates=10)


def bootstrap(diff,block):
    rng=np.random.default_rng(20261029+block);n=len(diff)
    starts=rng.integers(0,n,size=(CONFIG['bootstrap_repeats'],math.ceil(n/block)))
    indices=((starts[:,:,None]+np.arange(block))%n).reshape(len(starts),-1)[:,:n]
    draws=diff[indices].mean(1)
    return float(np.quantile(draws,.025)),float(np.quantile(draws,.975))


def summarize(winner):
    results=[json.loads(p.read_text()) for p in (OUT/'runs').glob('*/result.json')]
    flat=[];years=[];ccys=[]
    for r in results:
        for split in ['train','validation','report']:
            if split not in r:continue
            flat.append(dict(name=r['name'],fold=r['fold'],seed=r['seed'],split=split,epochs=r['epochs'],
                             best_epoch=r['best_epoch'],parameters=r['parameters'],seconds=r['seconds'],**r[split]))
            if split=='train':continue
            p=OUT/'runs'/f"{r['name']}__{r['fold']}__s{r['seed']}"/(split+'.parquet')
            f=pd.read_parquet(p);f['year']=pd.to_datetime(f.target_date).dt.year;f['hit']=(np.sign(f.pred)==np.sign(f.target))
            for year,g in f.groupby('year'):years.append(dict(name=r['name'],fold=r['fold'],seed=r['seed'],split=split,year=year,n=len(g),date_start=g.target_date.min(),date_end=g.target_date.max(),hit=float(g['hit'].mean())))
            for c,g in f.groupby('currency'):ccys.append(dict(name=r['name'],fold=r['fold'],seed=r['seed'],split=split,currency=c,hit=float(g['hit'].mean()),actual_positive=float((g.target>0).mean()),pred_positive=float((g.pred>0).mean())))
    table=pd.DataFrame(flat);table.to_csv(OUT/'all_results.csv',index=False)
    yf=pd.DataFrame(years);yf.to_csv(OUT/'all_seed_year.csv',index=False)
    pd.DataFrame(ccys).to_csv(OUT/'all_currency.csv',index=False)
    # Favorable report explicitly post hoc; includes ranks and full-table denominators.
    favorable=yf[yf.split=='report'].sort_values('hit',ascending=False).copy()
    favorable['post_hoc']=True;favorable['rank']=np.arange(1,len(favorable)+1)
    favorable.to_csv(OUT/'favorable_seed_period_POSTHOC.csv',index=False)
    ci=[]
    w=[]
    for seed in range(5):
        p=OUT/'runs'/f"{winner['name']}__final__s{seed}"/'report.parquet'
        if p.exists():
            f=pd.read_parquet(p);f['hit']=(np.sign(f.pred)==np.sign(f.target)).astype(float);w.append(f)
    wf=pd.concat(w);wh=wf.groupby(['decision_date','currency'])['hit'].mean()
    comparators={name:pd.read_parquet(OUT/(name+'_report.parquet')) for name in ['majority','logistic']}
    for name in ['original_legacy','winner_no_graph']:
        fs=[pd.read_parquet(p) for p in (OUT/'runs').glob(name+'__final__s*/report.parquet')]
        if fs:comparators[name]=pd.concat(fs)
    for name in ['mlp','transformer','gnn','corrlstmgat','fxrp']:
        fs=[]
        for path in (ROOT/'results/pit_main/runs/baselines').glob(name+'_seed*/predictions/*.parquet'):
            f=pd.read_parquet(path);f=f[f.currency!='USD'].rename(columns={'target_date':'decision_date'})
            f['decision_date']=pd.to_datetime(f.decision_date).dt.strftime('%Y-%m-%d');fs.append(f)
        if fs:comparators['archived_'+name]=pd.concat(fs)
    for variant in ['pit_main','pit_arcfx_active_core']:
        fs=[]
        for path in (ROOT/f'results/{variant}/runs/arc_fx').glob('oursmain_seed*/predictions/*.parquet'):
            f=pd.read_parquet(path);f=f[f.currency!='USD'].rename(columns={'target_date':'decision_date'})
            f['decision_date']=pd.to_datetime(f.decision_date).dt.strftime('%Y-%m-%d');fs.append(f)
        if fs:comparators['archived_'+variant]=pd.concat(fs)
    for name in ['original_legacy','winner_no_graph']:
        if name in comparators:comparators[name+'_matched_seeds']=comparators[name].copy()
    for name,f in comparators.items():
        current_wf=wf[wf.seed.isin(f.seed.unique())] if name.endswith('_matched_seeds') else wf
        wh=current_wf.groupby(['decision_date','currency'])['hit'].mean()
        f['hit']=(np.sign(f.pred)==np.sign(f.target)).astype(float)
        bh=f.groupby(['decision_date','currency'])['hit'].mean()
        pairs=pd.concat([wh.rename('winner'),bh.rename('baseline')],axis=1).dropna()
        assert len(pairs)==len(wh),(name,len(pairs),len(wh))
        # Verify identical actual targets, including archived target_date convention.
        wy=wf.groupby(['decision_date','currency']).target.first();by=f.groupby(['decision_date','currency']).target.first()
        assert np.allclose(wy.loc[pairs.index],by.loc[pairs.index],atol=1e-8)
        d=(pairs.winner-pairs.baseline).groupby('decision_date').mean().values
        for block in [5,20,60]:
            lo,hi=bootstrap(d,block);ci.append(dict(comparator=name,block=block,n_dates=len(d),winner_mean_hit=float(pairs.winner.mean()),baseline_mean_hit=float(pairs.baseline.mean()),winner_seeds=current_wf.seed.nunique(),baseline_seeds=f.seed.nunique(),difference=float(d.mean()),ci_low=lo,ci_high=hi,
                   note='Time uncertainty conditional on fitted seeds; development selection effect not corrected. Archived models use earlier train endpoint and stale input.'))
    pd.DataFrame(ci).to_csv(OUT/'paired_bootstrap.csv',index=False)
    aggregate=table.groupby(['name','fold','split']).mean_hit.agg(['mean','std','min','max','count']).reset_index()
    aggregate.to_csv(OUT/'aggregate.csv',index=False)
    ledger=[json.loads(s) for s in (OUT/'ledger.jsonl').read_text().splitlines()]
    dump(OUT/'completion.json',dict(winner=winner,completed_training_runs=len(results),
                                  training_attempts=sum(r.get('status')=='started' for r in ledger),
                                  failed_attempts=sum(r.get('status')=='failed' for r in ledger),
                                  baseline_logistic_fits=10,smoke_steps=250))
    print(aggregate[aggregate.split=='report'].to_string(index=False),flush=True)


if __name__=='__main__':main()
