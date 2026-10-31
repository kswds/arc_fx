"""Controlled 2x2 diagnostics and explicitly exploratory period/currency selection.

Run: python src/arcfx_followup.py
Original data, code, and experiment outputs are read only.
"""
from __future__ import annotations
import argparse, copy, hashlib, inspect, itertools, json, math, platform, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import data_pipeline as dp
import run_arcfx_search as previous
from arcfx_search_models import SearchModel
from models.arc_fx import usd_pin
from baselines.models import build_baseline_model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/arcfx_followup'
CCY=previous.CCY
CFG=dict(seeds=list(range(5)),epochs=20,patience=4,batch_size=256,lr=.0003,
         weight_decay=.0001,hidden=32,dropout=.25,lookback=10,threads=2,
         bootstrap_repeats=2000,blocks=[5,20,60],fx_lag_calendar_days=14,
         nonfx_extra_panel_lag=1,selection='validation sign-hit; tie lower raw-label BCE',
         primary_effects=['A4-A3','A3-A1','A4-A2'],
         policies=['conservative_lag14','market_feed_assumed'],
         matched_baselines=['mlp','transformer'],
         subset_min_dates=126,subset_sizes=list(range(1,11)),
         subset_seed_selection=False,subset_objectives=['hit','minimum_advantage','graph_advantage'])
CONDITIONS=dict(A1=dict(lag=1,graph=False),A2=dict(lag=1,graph=True),
                A3=dict(lag=0,graph=False),A4=dict(lag=0,graph=True))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_hash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):h.update(name.encode());h.update(t.detach().numpy().tobytes())
    return h.hexdigest()
def dump(path,obj):previous.dump(path,obj)
def log(**obj):
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),**obj),default=str)+'\n')


def raw_conservative(train_end):
    f=dp.normalize_date_col(pd.read_csv(ROOT/'data/processed/fx_panel_pit.csv'))
    n=dp.normalize_date_col(pd.read_csv(ROOT/'data/processed/nonfx_features_pit.csv'))
    df=dp.merge_prefer_right(f,n,on='Date').sort_values('Date').reset_index(drop=True)
    df=dp.engineer_nonfx_fallbacks(df,CCY);df,targets=dp.build_targets(df,CCY)
    df,_=dp.engineer_fx_local_features(df,CCY)
    df=df[df[[t for t in targets if t!='TargetRet_USD']].notna().all(axis=1)].reset_index(drop=True)
    y=df[targets].values.astype(np.float32)
    dates=df.Date.values.astype('datetime64[ns]')
    source=np.searchsorted(dates,dates-np.timedelta64(14,'D'),side='right')-1
    localcols=[t.format(ccy=c) for t in dp.LOCAL_FEATURE_TEMPLATES for c in CCY]
    nonfxcols=list(set([t.format(ccy=c) for t in dp.RATE_LOCAL_TEMPLATES+dp.EQUITY_TEMPLATES+dp.COUNTRY_MACRO_TEMPLATES for c in CCY]
                      +dp.RATE_GLOBAL_COLUMNS+dp.GLOBAL_COLUMNS))
    for col in localcols:
        if col in df:
            values=df[col].values.copy();df[col]=np.where(source>=0,values[np.maximum(source,0)],np.nan)
    for col in nonfxcols:
        if col in df:df[col]=df[col].shift(1)
    df,regimes=dp.add_regime_onehot_features(df,train_end)
    arrays={}
    for key,templates,broadcast in [('local',dp.LOCAL_FEATURE_TEMPLATES,None),
               ('rate',dp.RATE_LOCAL_TEMPLATES,dp.RATE_GLOBAL_COLUMNS),
               ('equity',dp.EQUITY_TEMPLATES,None),('countrymacro',dp.COUNTRY_MACRO_TEMPLATES,None)]:
        arrays[key]=dp.build_group_arrays(df,CCY,templates,broadcast)[0]
    arrays['global']=dp.build_global_array(df,dp.GLOBAL_COLUMNS+regimes)[0]
    stats={}
    for key,arr in arrays.items():
        arrays[key],mean,std=dp.normalize_train_only(arr,train_end)
        stats[key]=dict(mean=mean.tolist(),std=std.tolist())
    return SimpleNamespace(merged=df,y_raw=y,dims={k:v.shape[-1] for k,v in arrays.items()},
                           currency_names=CCY,**{'x_'+k:v for k,v in arrays.items()}),stats,source


class Fold:
    def __init__(self,policy):
        self.policy=policy;previous.N=len(pd.read_csv(ROOT/'data/processed/fx_panel_pit.csv'))
        n=previous.N;self.train_end=int(n*.7);self.scale_end=self.train_end-1
        self.val_end=10+int((n-10)*.6)+int((n-10)*.2)
        original=previous.Fold('final',self.train_end,self.val_end,n)
        self.original=original
        if policy=='market_feed_assumed':self.p=original.p;self.stats={};self.fx_source=np.arange(n)
        else:self.p,self.stats,self.fx_source=raw_conservative(self.scale_end)
        self.train=original.train;self.val=original.val;self.report=original.report
        self.y=torch.from_numpy(self.p.y_raw)
        self.blocks={}
        for key in ['local','rate','equity','countrymacro','global']:
            arr=getattr(self.p,'x_'+key)
            w=np.lib.stride_tricks.sliding_window_view(arr,10,axis=0)
            self.blocks['x_'+key]=torch.from_numpy(np.moveaxis(w,-1,1).copy())
        assert np.array_equal(self.p.y_raw,original.p.y_raw)
        assert pd.to_datetime(self.p.merged.Target_Date.iloc[self.train[-1]])<self.p.merged.Date.iloc[self.val[0]]
    def batch(self,ix,lag=0):return {k:v[ix-9-lag] for k,v in self.blocks.items()}


class Model(SearchModel):
    def __init__(self,dims,graph):
        super().__init__(dims,CCY,dict(encoder='mlp',loss='bce',graph='dense',no_graph=not graph))
    def forward(self,batch,intervention='original',adjacency=None):
        x=torch.cat([batch['x_local'],batch['x_rate'],batch['x_equity'],batch['x_countrymacro'],
                    batch['x_global'].unsqueeze(2).expand(-1,-1,11,-1)],-1)
        b,l,n,d=x.shape
        h=self.norm(self.encoder(x.permute(0,2,1,3).reshape(b*n,l,d)).reshape(b,n,-1)+self.embedding)
        e=self.q(h)@self.k(h).transpose(1,2)/math.sqrt(h.shape[-1])
        eye=torch.eye(n,dtype=torch.bool).unsqueeze(0)
        a=e.abs().masked_fill(eye,-torch.inf).softmax(-1)*e.tanh()
        if self.no_graph:a=torch.zeros_like(a)
        if intervention=='mean_adjacency':a=adjacency.unsqueeze(0).expand(b,-1,-1)
        m=a@self.value(h)
        if intervention=='zero_message':m=torch.zeros_like(m)
        g=self.gate(h).sigmoid()
        delta=self.correction(torch.cat([h,m],-1))-self.correction(torch.cat([h,torch.zeros_like(m)],-1))
        strength=self.direct(h)+g*delta
        self.last=dict(adj=a,message=m,gate=g,delta=delta,h=h,strength=strength)
        return usd_pin(strength[:,:,0])


class Baseline(nn.Module):
    def __init__(self,dims,name):
        super().__init__();self.net=build_baseline_model(name,sum(dims.values()),32,11,10,3,.25)
    def forward(self,batch,**unused):
        x=torch.cat([batch['x_local'],batch['x_rate'],batch['x_equity'],batch['x_countrymacro'],
                    batch['x_global'].unsqueeze(2).expand(-1,-1,11,-1)],-1)
        self.last={};return self.net(x)


def metrics(z,y,prediction_sign=None):
    z=np.asarray(z,dtype=np.float64);y=np.asarray(y);label=y>0
    prob=1/(1+np.exp(-np.clip(z,-50,50)))
    auc=roc_auc_score(label.ravel(),prob.ravel()) if np.unique(label).size>1 else np.nan
    return dict(mean_hit=float(((np.sign(z) if prediction_sign is None else prediction_sign)==np.sign(y)).mean()),
                binary_accuracy=float(((z>0)==label).mean()),
                bce=float((np.logaddexp(0,z)-z*label).mean()),brier=float(((prob-label)**2).mean()),
                auc=float(auc),zero_return_fraction=float((y==0).mean()),zero_logit_fraction=float((z==0).mean()),
                actual_positive=float(label.mean()),pred_positive=float((z>0).mean()),logit_mean=float(z.mean()),
                logit_std=float(z.std()),logit_q01=float(np.quantile(z,.01)),logit_q50=float(np.quantile(z,.5)),logit_q99=float(np.quantile(z,.99)))


@torch.no_grad()
def predict(model,fold,ix,lag=0,intervention='original',adjacency=None,details=False):
    model.eval();zs=[];aa=[];mm=[];gg=[]
    for start in range(0,len(ix),512):
        z=model(fold.batch(ix[start:start+512],lag),intervention=intervention,adjacency=adjacency)
        zs.append(z[:,1:].numpy())
        if details and model.last:
            aa.append(model.last['adj'].numpy());mm.append(model.last['message'].numpy());gg.append(model.last['gate'].numpy())
    z=np.concatenate(zs)
    detail=dict(adj=np.concatenate(aa),message=np.concatenate(mm),gate=np.concatenate(gg)) if aa else {}
    return z,detail


def frame(fold,ix,z,condition,seed,intervention='original'):
    f=pd.DataFrame(dict(decision_date=np.repeat(fold.p.merged.Date.iloc[ix].astype(str).values,10),
         target_date=np.repeat(fold.p.merged.Target_Date.iloc[ix].astype(str).values,10),currency=np.tile(CCY[1:],len(ix)),
         target=fold.p.y_raw[ix,1:].ravel(),logit=z.ravel(),seed=seed,condition=condition,
         policy=fold.policy,intervention=intervention))
    f['label']=(f.target>0).astype(int);return f


def read_grid(path):
    f=pd.read_parquet(path)
    return f,f.logit.values.reshape(-1,10),f.target.values.reshape(-1,10)


def calibrated(z,y):
    label=(y>0).astype(float)
    def objective(theta):
        a=np.exp(theta[0]);v=a*z+theta[1:]
        prob=1/(1+np.exp(-np.clip(v,-50,50)))
        residual=(prob-label)/label.size
        value=np.mean(np.logaddexp(0,v)-label*v)+.0001*np.mean(theta[1:]**2)
        gradient=np.r_[np.sum(residual*z)*a,residual.sum(0)+.0002*theta[1:]/10]
        return value,gradient
    fit=minimize(objective,np.zeros(11),jac=True,method='L-BFGS-B',bounds=[(-4,4)]+[(-3,3)]*10)
    return np.exp(fit.x[0]),fit.x[1:],dict(success=bool(fit.success),objective=float(fit.fun),iterations=int(fit.nit))


def diagnose_run(fold,condition,seed):
    dest=OUT/'runs'/fold.policy/f'{condition}_s{seed}'
    if (dest/'diagnostics.json').exists():return
    dp.set_seed(seed);model=Model(fold.p.dims,CONDITIONS[condition]['graph'])
    model.load_state_dict(torch.load(dest/'best.pt',weights_only=True));lag=CONDITIONS[condition]['lag']
    _,zt,yt=read_grid(dest/'train.parquet');_,zv,yv=read_grid(dest/'validation.parquet')
    scale,bias,calinfo=calibrated(zv.astype(float),yv)
    dump(dest/'calibration.json',dict(scale=float(scale),currency_offsets=dict(zip(CCY[1:],bias.tolist())),
                                   usd_offset=0,fit_split='validation',regularizer='1e-4 mean(b_i^2)',**calinfo))
    means=dict(train_mean=zt.mean(0),validation_mean=zv.mean(0))
    results={};transition_rows=[];contribution_frames=[]
    adjacency=None
    if CONDITIONS[condition]['graph']:
        _,details=predict(model,fold,fold.train,lag,details=True)
        adjacency=torch.from_numpy(details['adj'].mean(0));np.save(dest/'train_mean_adjacency.npy',adjacency.numpy())
    for split,ix in [('validation',fold.val),('report',fold.report)]:
        original,z,y=read_grid(dest/(split+'.parquet'))
        results[split]={}
        for label,v in list(means.items())+[('calibrated',None)]:
            zz=np.tile(v,(len(z),1)) if v is not None else scale*z+bias
            results[split][label]=metrics(zz,y)
            frame(fold,ix,zz,condition,seed,label).to_parquet(dest/(split+'_'+label+'.parquet'),index=False)
        if adjacency is None:continue
        original_z,detail=predict(model,fold,ix,lag,details=True)
        np.testing.assert_allclose(original_z,z,atol=1e-7)
        change=np.r_[np.nan,np.abs(np.diff(detail['adj'],axis=0)).mean((1,2))]
        for intervention in ['zero_message','mean_adjacency']:
            zz,d=predict(model,fold,ix,lag,intervention,adjacency,True)
            results[split][intervention]=metrics(zz,y)
            frame(fold,ix,zz,condition,seed,intervention).to_parquet(dest/(split+'_'+intervention+'.parquet'),index=False)
            oc=np.sign(z)==np.sign(y);ic=np.sign(zz)==np.sign(y)
            flipped=np.sign(z)!=np.sign(zz)
            transition_rows.append(dict(policy=fold.policy,condition=condition,seed=seed,split=split,intervention=intervention,
                 n=z.size,sign_changes=int(flipped.sum()),sign_change_fraction=float(flipped.mean()),
                 original_correct_to_wrong=int((flipped&oc&~ic).sum()),original_wrong_to_correct=int((flipped&~oc&ic).sum()),
                 abs_logit_change=float(np.abs(z-zz).mean()),message_rms=float(np.sqrt(np.mean(d['message']**2))),
                 gate_mean=float(d['gate'].mean()),gate_q01=float(np.quantile(d['gate'],.01)),gate_q99=float(np.quantile(d['gate'],.99)),
                 mean_adjacent_adjacency_change=float(np.nanmean(change))))
            cf=original[['decision_date','target_date','currency','seed','condition','policy']].copy()
            cf['split']=split;cf['intervention']=intervention;cf['logit_graph_minus_intervention']=(z-zz).ravel()
            cf['hit_graph_minus_intervention']=(oc.astype(int)-ic.astype(int)).ravel()
            cf['adjacent_adjacency_change']=np.repeat(change,10)
            cf['message_rms']=np.sqrt(np.mean(detail['message'][:,1:,:]**2,axis=-1)).ravel()
            cf['gate']=detail['gate'][:,1:,0].ravel();contribution_frames.append(cf)
    if transition_rows:pd.DataFrame(transition_rows).to_csv(dest/'graph_transitions.csv',index=False)
    if contribution_frames:pd.concat(contribution_frames).to_parquet(dest/'graph_contributions.parquet',index=False)
    _,z,y=read_grid(dest/'report.parquet');rng=np.random.default_rng(20261030)
    shifts=rng.choice(np.arange(61,len(z)-60),size=min(200,len(z)-121),replace=False)
    perm=[dict(shift=int(shift),**metrics(np.roll(z,int(shift),axis=0),y)) for shift in shifts]
    pd.DataFrame(perm).to_csv(dest/'circular_shift.csv',index=False)
    results['permutation']=dict(actual=metrics(z,y),mean_hit=float(np.mean([p['mean_hit'] for p in perm])),
               fraction_shift_hit_ge_actual=float(np.mean([p['mean_hit']>=metrics(z,y)['mean_hit'] for p in perm])),
               note='Date-alignment diagnostic, not a stationarity-robust significance test')
    correlations=[]
    for j,c in enumerate(CCY[1:]):
        zz=z[:,j]-means['train_mean'][j];yy=(y[:,j]>0).astype(float)-(yt[:,j]>0).mean()
        correlations.append(dict(currency=c,logit_label_residual_corr=float(np.corrcoef(zz,yy)[0,1])))
    pd.DataFrame(correlations).to_csv(dest/'residual_correlations.csv',index=False)
    dump(dest/'diagnostics.json',results)


def baseline_frames(fold):
    yt=fold.p.y_raw[fold.train,1:];prob=np.clip((yt>0).mean(0),1e-6,1-1e-6)
    dest=OUT/'baselines'/fold.policy;dest.mkdir(parents=True,exist_ok=True)
    dump(dest/'train_probabilities.json',dict(zip(CCY[1:],prob.tolist())))
    for split,ix in [('validation',fold.val),('report',fold.report)]:
        for name,p in [('half',np.ones(10)*.5),('train_probability',prob),('majority',prob)]:
            # All three predict down at p==0.5, as explicitly requested for probability baselines.
            z=np.tile(np.log(p/(1-p)),(len(ix),1));f=frame(fold,ix,z,name,-1)
            f['prediction_sign']=np.where(f.logit>0,1,-1)
            f.to_parquet(dest/(split+'_'+name+'.parquet'),index=False)


def calibrate_baselines():
    for policy in CFG['policies']:
        for condition in CFG['matched_baselines']:
            for seed in CFG['seeds']:
                dest=OUT/'runs'/policy/f'{condition}_s{seed}'
                _,z,y=read_grid(dest/'validation.parquet')
                scale,bias,info=calibrated(z.astype(float),y)
                dump(dest/'calibration.json',dict(scale=float(scale),currency_offsets=dict(zip(CCY[1:],bias.tolist())),
                    usd_offset=0,fit_split='validation',regularizer='1e-4 mean(b_i^2)',**info))
                for split in ['validation','report']:
                    f,z,y=read_grid(dest/(split+'.parquet'))
                    f['logit']=(scale*z+bias).ravel();f['intervention']='calibrated'
                    f.to_parquet(dest/(split+'_calibrated.parquet'),index=False)


def prediction_metrics():
    overall=[];currencies=[];reliability=[];permutations=[];transitions=[];corr=[]
    for policy in CFG['policies']:
        for seed in CFG['seeds']:
            for condition in list(CONDITIONS)+CFG['matched_baselines']:
                dest=OUT/'runs'/policy/f'{condition}_s{seed}'
                names=['validation','report']
                if condition in CFG['matched_baselines']:names += ['validation_calibrated','report_calibrated']
                if condition in CONDITIONS:
                    names += [sp+'_'+v for sp in ['validation','report'] for v in ['train_mean','validation_mean','calibrated']]
                    if CONDITIONS[condition]['graph']:names += [sp+'_'+v for sp in ['validation','report'] for v in ['zero_message','mean_adjacency']]
                for filename in names:
                    path=dest/(filename+'.parquet');f=pd.read_parquet(path)
                    split=filename.split('_')[0];variant=filename[len(split)+1:] or 'original'
                    for c,g in [('ALL',f)]+list(f.groupby('currency',sort=False)):
                        meta=dict(policy=policy,condition=condition,seed=seed,split=split,variant=variant,currency=c,n=len(g))
                        m=metrics(g.logit.values,g.target.values)
                        (overall if c=='ALL' else currencies).append(dict(**meta,**m))
                        if c=='ALL':continue
                        y=g.target.values>0;pred=g.logit.values>0
                        currencies[-1].update(tn=int((~pred&~y).sum()),fp=int((pred&~y).sum()),fn=int((~pred&y).sum()),tp=int((pred&y).sum()))
                        p=1/(1+np.exp(-g.logit.values));bins=np.minimum((p*10).astype(int),9)
                        for bi in np.unique(bins):
                            mask=bins==bi
                            reliability.append(dict(**meta,bin=int(bi),count=int(mask.sum()),mean_probability=float(p[mask].mean()),positive_fraction=float(y[mask].mean())))
                if condition in CONDITIONS:
                    perm=pd.read_csv(dest/'circular_shift.csv');perm['policy']=policy;perm['condition']=condition;perm['seed']=seed;permutations.append(perm)
                    co=pd.read_csv(dest/'residual_correlations.csv');co['policy']=policy;co['condition']=condition;co['seed']=seed;corr.append(co)
                if (dest/'graph_transitions.csv').exists():transitions.append(pd.read_csv(dest/'graph_transitions.csv'))
        for split in ['validation','report']:
            for name in ['half','train_probability','majority']:
                f=pd.read_parquet(OUT/'baselines'/policy/(split+'_'+name+'.parquet'))
                for c,g in [('ALL',f)]+list(f.groupby('currency',sort=False)):
                    m=metrics(g.logit.values,g.target.values,g.prediction_sign.values)
                    row=dict(policy=policy,condition=name,seed=-1,split=split,variant='original',currency=c,n=len(g),**m)
                    (overall if c=='ALL' else currencies).append(row)
    table=pd.DataFrame(overall);table.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(currencies).to_csv(OUT/'currency_metrics_confusion.csv',index=False)
    pd.DataFrame(reliability).to_csv(OUT/'reliability.csv',index=False)
    pd.concat(permutations).to_csv(OUT/'circular_shift.csv',index=False)
    pd.concat(transitions).to_csv(OUT/'graph_transitions.csv',index=False)
    pd.concat(corr).to_csv(OUT/'residual_correlations.csv',index=False)
    for field in ['mean_hit','binary_accuracy','bce','brier','auc']:
        ag=table.groupby(['policy','condition','split','variant'])[field].agg(['mean','std','count']).reset_index()
        ag.to_csv(OUT/('aggregate_'+field+'.csv'),index=False)
    return table


BOOT={}
def bootstrap(diff,block):
    n=len(diff);key=(n,block)
    if key not in BOOT:
        rng=np.random.default_rng(20261030+block)
        starts=rng.integers(0,n,(CFG['bootstrap_repeats'],math.ceil(n/block)))
        BOOT[key]=((starts[:,:,None]+np.arange(block))%n).reshape(len(starts),-1)[:,:n]
    values=np.asarray(diff)[BOOT[key]].mean(1)
    return tuple(np.quantile(values,[.025,.975]))


def sample_losses(path):
    f,z,y=read_grid(path);hit=(np.sign(z)==np.sign(y)).astype(float)
    if 'prediction_sign' in f:hit=(f.prediction_sign.values.reshape(-1,10)==np.sign(y)).astype(float)
    return dict(mean_hit=hit,binary_accuracy=((z>0)==(y>0)).astype(float),
                bce=-(np.logaddexp(0,z.astype(float))-z*(y>0)))


def paired_analysis():
    rows=[]
    for policy in CFG['policies']:
        grids={}
        for condition in CONDITIONS:
            for variant in ['original','train_mean','validation_mean','calibrated']+(['zero_message','mean_adjacency'] if CONDITIONS[condition]['graph'] else []):
                suffix='' if variant=='original' else '_'+variant
                grids[condition+':'+variant]={k:np.stack([sample_losses(OUT/'runs'/policy/f'{condition}_s{s}'/('report'+suffix+'.parquet'))[k] for s in CFG['seeds']]) for k in ['mean_hit','binary_accuracy','bce']}
        for name in ['train_probability','half']:
            v=sample_losses(OUT/'baselines'/policy/('report_'+name+'.parquet'))
            grids[name]={k:np.repeat(arr[None],5,axis=0) for k,arr in v.items()}
        for condition in CFG['matched_baselines']:
            for variant in ['original','calibrated']:
                suffix='' if variant=='original' else '_'+variant
                grids[condition+':'+variant]={k:np.stack([sample_losses(OUT/'runs'/policy/f'{condition}_s{s}'/('report'+suffix+'.parquet'))[k] for s in CFG['seeds']]) for k in ['mean_hit','binary_accuracy','bce']}
        comparisons=[('A4-A3',[('A4:original',1),('A3:original',-1)],True),
             ('A3-A1',[('A3:original',1),('A1:original',-1)],True),
             ('A4-A2',[('A4:original',1),('A2:original',-1)],True),
             ('A2-A1',[('A2:original',1),('A1:original',-1)],False),
             ('interaction',[('A4:original',1),('A3:original',-1),('A2:original',-1),('A1:original',1)],False)]
        for other in ['train_probability','half','A4:train_mean','A4:validation_mean','A4:zero_message','A4:mean_adjacency']:
            comparisons.append(('A4-'+other,[('A4:original',1),(other,-1)],False))
        comparisons.append(('A4_calibrated-A4',[('A4:calibrated',1),('A4:original',-1)],False))
        for baseline in ['A3']+CFG['matched_baselines']:
            comparisons.append(('A4_calibrated-'+baseline+'_calibrated',[('A4:calibrated',1),(baseline+':calibrated',-1)],False))
        for baseline in CFG['matched_baselines']:
            comparisons.append(('A4-'+baseline,[('A4:original',1),(baseline+':original',-1)],False))
        for name,terms,primary in comparisons:
            for metric in ['mean_hit','binary_accuracy','bce']:
                diff=sum(grids[g][metric]*weight for g,weight in terms).mean(2)
                bydate=diff.mean(0);seed_effect=diff.mean(1)
                factor=100 if metric!='bce' else 1
                for block in CFG['blocks']:
                    lo,hi=bootstrap(bydate,block)
                    rows.append(dict(policy=policy,comparison=name,metric=metric,primary=primary,
                      block=block,n_dates=len(bydate),seeds=5,effect=float(bydate.mean()*factor),
                      ci_low=float(lo*factor),ci_high=float(hi*factor),seed_effect_std=float(seed_effect.std(ddof=1)*factor),
                      unit='percentage points' if metric!='bce' else 'BCE reduction (positive improves)',
                      uncertainty='date resampling conditional on common five fitted seeds; report selection not corrected'))
    pd.DataFrame(rows).to_csv(OUT/'paired_effects.csv',index=False)


def exploratory_selection(folds):
    """All seeds kept; every competitor shares every selected row and currency."""
    rankings=[];validation_selected=[];all_tables=[]
    # 1,023 nonempty USD-target subsets, plus 45 individual additional cross pairs.
    masks=np.array([[int(bool(code&(1<<j))) for j in range(10)] for code in range(1,1024)],dtype=float)
    masks/=masks.sum(1,keepdims=True)
    sets=['+'.join(c for c,v in zip(CCY[1:],row) if v>0) for row in masks]
    pairs=[(j,0,CCY[j]+'/USD') for j in range(1,11)]
    pairs += [(i,j,CCY[i]+'/'+CCY[j]) for i in range(1,11) for j in range(i+1,11)]
    for policy,fold in folds.items():
        paths={}
        for name in ['A3','A4']+CFG['matched_baselines']:
            paths[name]=[OUT/'runs'/policy/f'{name}_s{s}' for s in CFG['seeds']]
        grids={};ys={};dates={}
        for split in ['validation','report']:
            for name,pp in paths.items():
                ff=[read_grid(p/(split+'.parquet')) for p in pp]
                grids[(split,name)]=np.stack([r[1] for r in ff]);ys[split]=ff[0][2]
                dates[split]=pd.to_datetime(ff[0][0].target_date.values.reshape(-1,10)[:,0])
                grids[(split,name+'_calibrated')]=np.stack([read_grid(p/(split+'_calibrated.parquet'))[1] for p in pp])
        yt=fold.p.y_raw[fold.train]
        # Period families fixed before loading report metrics: contiguous calendar-half blocks.
        half=dates['report'].year*2+(dates['report'].month>6).astype(int)
        unique=np.unique(half);windows=[('ALL',np.arange(len(half)))]
        for i,start in enumerate(unique):
            for end in unique[i:]:
                ix=np.flatnonzero((half>=start)&(half<=end))
                if len(ix)>=CFG['subset_min_dates'] and len(ix)<len(half):windows.append((f'{dates["report"][ix[0]].date()}..{dates["report"][ix[-1]].date()}',ix))
        for candidate in ['A4','A4_calibrated']:
            split='report';y=ys[split];z=grids[(split,candidate)]
            suffix='_calibrated' if candidate.endswith('_calibrated') else ''
            correct=(np.sign(z)==np.sign(y)[None]).mean(0)
            baseline_hits={name:(np.sign(grids[(split,name+suffix)])==np.sign(y)[None]).mean(0) for name in ['A3']+CFG['matched_baselines']}
            prob=(yt[:,1:]>0).mean(0);direction=np.where(prob>.5,1,-1)
            baseline_hits['train_probability']=(direction[None]==np.sign(y)).astype(float)
            for period,ix in windows:
                scores=masks@correct[ix].mean(0)
                bs={name:masks@values[ix].mean(0) for name,values in baseline_hits.items()}
                worst=scores-np.stack(list(bs.values())).max(0)
                graphdiff=scores-bs['A3']
                t=pd.DataFrame(dict(policy=policy,candidate=candidate,period=period,start=str(dates[split][ix[0]].date()),
                       end=str(dates[split][ix[-1]].date()),n_dates=len(ix),targets=sets,
                       n_targets=(masks>0).sum(1),mean_hit=scores,minimum_advantage=worst,graph_advantage=graphdiff,
                       source='USD_currency_subset',comparator_variant='calibrated' if suffix else 'original',selection='report-selected; descriptive, uncorrected'))
                for name,values in bs.items():t['baseline_'+name]=values
                all_tables.append(t)
            # Cross-rate evaluation preserves strength-difference prediction and synchronized targets.
            zz=np.concatenate([np.zeros(z.shape[:2]+(1,)),z],axis=2)
            yy=np.c_[np.zeros(len(y)),y]
            pairz=np.stack([zz[:,:,i]-zz[:,:,j] for i,j,_ in pairs],axis=-1)
            pairy=np.stack([yy[:,i]-yy[:,j] for i,j,_ in pairs],axis=-1)
            pairhits=(np.sign(pairz)==np.sign(pairy)[None]).mean(0)
            pairbase={}
            for name in ['A3']+CFG['matched_baselines']:
                bz=grids[(split,name+suffix)];bz=np.concatenate([np.zeros(bz.shape[:2]+(1,)),bz],axis=2)
                pairbase[name]=(np.sign(np.stack([bz[:,:,i]-bz[:,:,j] for i,j,_ in pairs],-1))==np.sign(pairy)[None]).mean(0)
            ptr=np.array([(yt[:,i]-yt[:,j]>0).mean() for i,j,_ in pairs])
            pairbase['train_probability']=(np.where(ptr>.5,1,-1)[None]==np.sign(pairy)).astype(float)
            for period,ix in windows:
                score=pairhits[ix].mean(0);bs={name:v[ix].mean(0) for name,v in pairbase.items()}
                t=pd.DataFrame(dict(policy=policy,candidate=candidate,period=period,start=str(dates[split][ix[0]].date()),
                    end=str(dates[split][ix[-1]].date()),n_dates=len(ix),targets=[name for _,_,name in pairs],n_targets=1,
                    mean_hit=score,minimum_advantage=score-np.stack(list(bs.values())).max(0),graph_advantage=score-bs['A3'],
                    source='single_cross_rate',comparator_variant='calibrated' if suffix else 'original',selection='report-selected; descriptive, uncorrected'))
                for name,v in bs.items():t['baseline_'+name]=v
                all_tables.append(t)
            # Currency selection using validation only, followed by all report dates.
            vy=ys['validation'];vc=(np.sign(grids[('validation',candidate)])==np.sign(vy)[None]).mean(0).mean(0)
            vb={name:(np.sign(grids[('validation',name+suffix)])==np.sign(vy)[None]).mean(0).mean(0) for name in ['A3']+CFG['matched_baselines']}
            vb['train_probability']=(direction[None]==np.sign(vy)).mean(0)
            vscore=masks@vc;vmargin=vscore-np.stack([masks@b for b in vb.values()]).max(0)
            for objective,array in [('hit',vscore),('minimum_advantage',vmargin)]:
                best=int(np.argmax(array));row=dict(policy=policy,candidate=candidate,objective=objective,targets=sets[best],
                        validation_hit=float(vscore[best]),report_hit=float(masks[best]@correct.mean(0)),selection='validation-only currency selection; all report dates')
                for name,values in baseline_hits.items():row['report_'+name]=float(masks[best]@values.mean(0))
                validation_selected.append(row)
    table=pd.concat(all_tables,ignore_index=True)
    table['minimum_advantage_pp']=100*table.minimum_advantage
    table['graph_advantage_pp']=100*table.graph_advantage
    table['canonical_targets']=[target+'/USD' if number==1 and '/' not in target else target for target,number in zip(table.targets,table.n_targets)]
    table.to_parquet(OUT/'ALL_scope_search.parquet',index=False)
    for (policy,candidate),group in table.groupby(['policy','candidate']):
        for objective in CFG['subset_objectives']:
            column='mean_hit' if objective=='hit' else objective
            top=group.sort_values([column,'n_dates'],ascending=False).drop_duplicates(['start','end','canonical_targets']).head(20).copy();top['objective']=objective;rankings.append(top)
    pd.concat(rankings).to_csv(OUT/'selected_scopes_POSTHOC.csv',index=False)
    pd.DataFrame(validation_selected).to_csv(OUT/'validation_selected_currencies.csv',index=False)
    dump(OUT/'scope_search_manifest.json',dict(total_combinations=len(table),unique_combinations=len(table.drop_duplicates(['policy','candidate','start','end','canonical_targets'])),
        period_family='contiguous calendar-half blocks and full report, at least 126 market dates',
        currency_subsets=1023,individual_cross_rates=55,seeds='all 0,1,2,3,4 averaged; no seed selection',
        comparators=['matched A3 no graph','matched MLP','matched Transformer','train-only direction frequency'],
        calibrated_candidate='same positive common slope plus ten node offsets fitted on validation only for A4 and every neural comparator',
        warnings='Report-selected scope ranks are descriptive; nominal confidence intervals do not adjust selection. Graph remains 11-node; only evaluation targets change; all models trained on all ten USD targets.'))


def analyze(folds):
    for policy,fold in folds.items():
        baseline_frames(fold)
        for condition in CONDITIONS:
            for seed in CFG['seeds']:diagnose_run(fold,condition,seed)
    calibrate_baselines()
    table=prediction_metrics();paired_analysis();exploratory_selection(folds);graph_change_analysis();scoped_uncertainty(folds)
    make_report(table);plots()
    results=[json.loads(p.read_text()) for p in (OUT/'runs').glob('*/*/result.json')]
    dump(OUT/'completion.json',dict(total_model_conditions=len(results),new_neural_fits=sum(r['status']=='trained' for r in results),
         reused_fits=sum(r['status']=='reused' for r in results),diagnostic_2x2_conditions=40,matched_baseline_fits=20,
         calibration_fits=60,failures=sum(json.loads(line).get('status')=='failed' for line in (OUT/'ledger.jsonl').read_text().splitlines())))
    m=json.loads((OUT/'manifest.json').read_text())
    m['training_logic_sha256']=training_logic_hash()
    m['analysis_source_sha256']=sha(ROOT/'src/arcfx_followup.py')
    m['protocol_sha256']=sha(ROOT/'docs/arcfx_followup_protocol.md')
    dump(OUT/'manifest.json',m)


def graph_change_analysis():
    rows=[]
    for path in (OUT/'runs').glob('*/*/graph_contributions.parquet'):
        f=pd.read_parquet(path)
        for intervention,g in f.groupby('intervention'):
            val=g[g.split=='validation'].groupby('decision_date').adjacent_adjacency_change.first().dropna()
            thresholds=val.quantile([.25,.5,.75]).values
            for split in ['validation','report']:
                h=g[g.split==split].copy();h['change_bin']=np.searchsorted(thresholds,h.adjacent_adjacency_change.values)
                h=h[h.adjacent_adjacency_change.notna()]
                for bi,b in h.groupby('change_bin'):
                    rows.append(dict(policy=b.policy.iloc[0],condition=b.condition.iloc[0],seed=int(b.seed.iloc[0]),
                         split=split,intervention=intervention,change_bin=int(bi),n_dates=b.decision_date.nunique(),
                         mean_adjacency_change=float(b.adjacent_adjacency_change.mean()),
                         graph_hit_gain_pp=float(100*b.hit_graph_minus_intervention.mean()),
                         mean_abs_logit_change=float(b.logit_graph_minus_intervention.abs().mean()),
                         thresholds_source='validation adjacent-date graph change quartiles'))
    pd.DataFrame(rows).to_csv(OUT/'graph_change_vs_gain.csv',index=False)


def scoped_uncertainty(folds):
    ranked=pd.read_csv(OUT/'selected_scopes_POSTHOC.csv')
    selections=ranked.groupby(['policy','candidate','objective'],sort=False).head(1)
    cirows=[];seedrows=[]
    for selection_id,(_,row) in enumerate(selections.iterrows()):
        fold=folds[row.policy];suffix='_calibrated' if row.candidate.endswith('_calibrated') else ''
        _,z,y=read_grid(OUT/'runs'/row.policy/'A4_s0'/'report.parquet')
        date=pd.to_datetime(fold.p.merged.Target_Date.iloc[fold.report])
        ix=np.flatnonzero((date>=row.start)&(date<=row.end))
        yy=np.c_[np.zeros(len(y)),y];yt=fold.p.y_raw[fold.train]
        if '/' in row.targets:
            base,quote=row.targets.split('/');pairs=[(CCY.index(base),CCY.index(quote))]
        else:pairs=[(CCY.index(c),0) for c in row.targets.split('+')]
        target=np.stack([yy[:,i]-yy[:,j] for i,j in pairs],-1)[ix]
        arrays={}
        for name in ['A4','A3']+CFG['matched_baselines']:
            zz=np.stack([read_grid(OUT/'runs'/row.policy/f'{name}_s{s}'/('report'+suffix+'.parquet'))[1] for s in CFG['seeds']])
            zz=np.concatenate([np.zeros(zz.shape[:2]+(1,)),zz],axis=-1)
            arrays[name]=np.stack([zz[:,:,i]-zz[:,:,j] for i,j in pairs],-1)[:,ix,:]
        p=np.clip(np.array([(yt[:,i]-yt[:,j]>0).mean() for i,j in pairs]),1e-6,1-1e-6)
        arrays['train_probability']=np.broadcast_to(np.log(p/(1-p)),arrays['A4'].shape)
        correctness={name:(np.sign(arr)==np.sign(target)[None]).astype(float) for name,arr in arrays.items()}
        correctness['train_probability']=(np.broadcast_to(np.where(p>.5,1,-1),arrays['A4'].shape)==np.sign(target)[None]).astype(float)
        np.testing.assert_allclose(correctness['A4'].mean(),row.mean_hit,atol=1e-12)
        base=dict(selection_id=selection_id,policy=row.policy,candidate=row.candidate,objective=row.objective,
                  start=row.start,end=row.end,targets=row.targets,n_dates=len(ix),n_targets=len(pairs))
        for seed in CFG['seeds']:
            sr=dict(**base,seed=seed,mean_hit=float(correctness['A4'][seed].mean()))
            for name in ['A3']+CFG['matched_baselines']+['train_probability']:sr['baseline_'+name]=float(correctness[name][seed].mean())
            seedrows.append(sr)
        for name in ['A3']+CFG['matched_baselines']+['train_probability']:
            diff=(correctness['A4']-correctness[name]).mean(2)
            for block in CFG['blocks']:
                lo,hi=bootstrap(diff.mean(0),block)
                cirows.append(dict(**base,comparator=name,block=block,our_hit=float(correctness['A4'].mean()),
                    baseline_hit=float(correctness[name].mean()),effect_pp=float(diff.mean()*100),ci_low_pp=float(lo*100),ci_high_pp=float(hi*100),
                    seed_effect_std_pp=float(diff.mean(1).std(ddof=1)*100),winning_seeds=int((diff.mean(1)>0).sum()),
                    selection_adjustment='NONE: period/targets selected on report labels; interval is descriptive conditional on selection'))
    pd.DataFrame(cirows).to_csv(OUT/'selected_scope_bootstrap_UNCORRECTED.csv',index=False)
    pd.DataFrame(seedrows).to_csv(OUT/'selected_scope_seed_results.csv',index=False)


def make_report(table):
    # Detailed report filled from computed artifacts, including all unfavorable comparisons.
    effects=pd.read_csv(OUT/'paired_effects.csv');primary=effects[(effects.block==20)&(effects.metric=='mean_hit')]
    scopes=pd.read_csv(OUT/'selected_scopes_POSTHOC.csv');trans=pd.read_csv(OUT/'graph_transitions.csv')
    ag=table[(table.split=='report')&(table.variant=='original')].groupby(['policy','condition']).agg(
        mean_hit=('mean_hit','mean'),seed_std=('mean_hit','std'),binary_accuracy=('binary_accuracy','mean'),bce=('bce','mean'),auc=('auc','mean')).reset_index()
    fm=json.loads((OUT/'manifest.json').read_text())
    currencies=pd.read_csv(OUT/'currency_metrics_confusion.csv')
    correlations=pd.read_csv(OUT/'residual_correlations.csv')
    shifts=pd.read_csv(OUT/'circular_shift.csv')
    scope_ci=pd.read_csv(OUT/'selected_scope_bootstrap_UNCORRECTED.csv')
    scoped_seeds=pd.read_csv(OUT/'selected_scope_seed_results.csv')
    search=json.loads((OUT/'scope_search_manifest.json').read_text())
    run_results=[json.loads(p.read_text()) for p in (OUT/'runs').glob('*/*/result.json')]
    def effect_text(policy,comparison,metric='mean_hit'):
        row=effects[(effects.policy==policy)&(effects.comparison==comparison)&(effects.metric==metric)&(effects.block==20)].iloc[0]
        unit='%p' if metric!='bce' else 'BCE 감소량'
        verdict='차이의 신뢰구간이 0을 포함하여 우위 미확인' if row.ci_low<=0<=row.ci_high else '조건부 신뢰구간은 0을 포함하지 않음(탐색 선택 효과 미보정)'
        return f"{policy}: {comparison} = {row.effect:+.5f} {unit}, 95% CI [{row.ci_low:.5f}, {row.ci_high:.5f}]; {verdict}."
    def tab(df):return df.to_markdown(index=False,floatfmt='.5f')
    lines=['# ARC_FX 후속 진단 및 평가 범위 탐색','',
      '사용자 허용에 따라 보고 기간·통화 조합을 탐색했다. 선택 범위에서는 모든 모델에 같은 날짜·대상·학습 규칙을 적용했다. 보고 label을 이용해 선택한 최대값은 해당 표본의 기술적 결과이며 일반화 성능의 독립 검증이 아니다. Seed는 0–4를 모두 평균했고 유리한 seed를 선택하지 않았다.','',
      f"총 {len(run_results)}개 model conditions: 새 neural 학습 {sum(r['status']=='trained' for r in run_results)}, 검증 후 재사용 {sum(r['status']=='reused' for r in run_results)}. 두 가용성 정책의 2×2가 40개, 동일 조건 MLP/Transformer가 20개다. Validation-only 보정은 모든 neural condition에 같은 절차로 60회 수행했다.",'',
      '## 공통 조건과 가용성','',
      'A1=t−1/no graph, A2=t−1/dynamic, A3=t/no graph, A4=t/dynamic. Dense signed MLP, hidden 32, lookback 10, raw-label BCE, AdamW lr=0.0003, weight decay=0.0001, batch=256, epochs≤20, patience=4, validation sign-hit 최대/동률 BCE 최소. 모든 neural 비교는 같은 train/validation/report row, 손실, optimizer, shuffle seed, checkpoint 규칙을 사용한다.','',
      '보수적 정책: FX 파생특징에 14 calendar-day lag, 기존 PIT 비FX 특징에 1 panel-day 추가 lag, 이후 regime 재계산과 train-only scaling. 시장 피드 가정 정책: 기존 탐색의 t 입력을 그대로 사용한다.','',
      '연준은 H.10의 전주 daily rates를 월요일 16:15 ET(연방 공휴일이면 다음 영업일)에 공개한다. 따라서 당일 noon 관측과 당일 공개는 다르다. 2006-05~2009-01 weekly publication 중단, 후속 개정, 비FX 원천별 실제 공표시각은 완전한 vintage 감사를 마치지 않았다. 14일 buffer도 공개 일정 가정 하의 보수적 설계이며 실시간 가용성 인증은 아니다. [연준 공식 안내](https://www.federalreserve.gov/releases/h10/default.htm), [H.10 설명](https://www.federalreserve.gov/releases/h10/about.htm).','',
      json.dumps(fm['evaluation']['train'],ensure_ascii=False)+' 학습; '+json.dumps(fm['evaluation']['validation'],ensure_ascii=False)+' checkpoint validation; '+json.dumps(fm['evaluation']['report_return_end'],ensure_ascii=False)+' return 종료일 보고.','',
      '## 전체 범위 결과','',tab(ag),'',
      '주 지표는 기존 np.sign(logit)==np.sign(return)이며 0 return도 분모에 남긴다. Binary accuracy는 (logit>0)==(return>0)이다. Probability baseline은 p>0.5일 때 상승, 나머지는 하락으로 예측한다. p=0.5의 logit=0을 모델의 3-way zero 예측과 혼동하지 않도록 baseline 방향을 명시했다. 모든 출력은 방향 logit이며 RMSE를 계산하지 않았다.','',
      '## 1. 입력 정렬 효과','',tab(primary[primary.comparison.isin(['A3-A1','A4-A2'])][['policy','comparison','effect','ci_low','ci_high','seed_effect_std']]),'',
      '단위는 %p다. 보수적 정책에서 t는 14일 이상 지난 FX snapshot과 추가 lag를 준 비FX snapshot의 최신 row를 뜻한다. 기존 당일 spot 입력의 효과는 시장 피드 가정 정책에서만 재현한다.','',
      *[effect_text(p,c) for p in CFG['policies'] for c in ['A3-A1','A4-A2']],'',
      '## 2. 최신 입력의 실제 가용성','',
      '원천 코드의 availability audit상 위반은 0건이나, H.10 공식 공개 지연이 그 audit의 FX 파생특징에 반영되지 않았다. 원래 50.5%는 시장 피드 가정에 의존한다. 신규 보수적 결과도 역사적 release별 인증 완료 결과로 해석해서는 안 된다.','',
      '## 3. 통화별 방향 빈도 이상의 정보','',
      tab(primary[primary.comparison.isin(['A4-train_probability','A4-A4:train_mean','A4-A4:validation_mean'])][['policy','comparison','effect','ci_low','ci_high']]),'',
      '통화별 train 상수 확률, 학습/validation 평균 logit 고정, 실제 동적 예측을 비교했다. BCE 개선은 paired_effects.csv의 metric=bce에서 양수가 개선이다. 통화별 AUC와 residual correlation은 currency_metrics_confusion.csv, residual_correlations.csv에 있다. Circular shift는 모든 통화에 같은 shift를 주며 ±60일 이내를 제외했다. 비정상성 때문에 엄밀한 유의성 검정으로 해석하지 않는다.','',
      *[effect_text(p,'A4-train_probability',m) for p in CFG['policies'] for m in ['mean_hit','bce']],'',
      '## 4. 확률 보정과 구분 능력','',
      'AUC는 구분 능력, BCE/Brier/reliability는 확률 품질을 본다. z′=a z+b_i, a>0, b_USD=0 보정은 validation BCE로만 학습했다. 보정은 새로운 시점별 정보를 추가하지 않는다. 보정 전후는 별도 variant로 저장했다. 전체 및 통화별 수치는 metrics.csv/currency_metrics_confusion.csv, 그림은 reliability.png/confusion_A4.png를 참조한다.','',
      *[effect_text(p,'A4_calibrated-A4',m) for p in CFG['policies'] for m in ['mean_hit','bce']],'',
      '## 5. 동일 조건에서 dynamic graph 추가 가치','',tab(primary[primary.comparison=='A4-A3'][['policy','effect','ci_low','ci_high','seed_effect_std']]),'',
      'No-graph는 동일 초기화된 모든 모듈을 유지하고 메시지를 0으로 만든 학습 variant다. Gate와 direct head는 각 노드 h만 사용하고 foreign-node pooling 우회 경로는 없다. 사전 정의 common/global/regime 특징과 USD gauge는 공통으로 남는다. 따라서 no-graph도 공통 시장 특징은 사용할 수 있다.','',
      *[effect_text(p,'A4-A3') for p in CFG['policies']],'',
      '## 6. Graph가 바꾼 예측의 정답/오답','',
      tab(trans[(trans.condition=='A4')&(trans.split=='report')].groupby(['policy','intervention']).agg(
          sign_changes=('sign_changes','sum'),correct_to_wrong=('original_correct_to_wrong','sum'),wrong_to_correct=('original_wrong_to_correct','sum'),
          abs_logit_change=('abs_logit_change','mean'),message_rms=('message_rms','mean')).reset_index()),'',
      '이 표는 원래 graph 예측에서 intervention으로 변경했을 때의 전환이다. correct_to_wrong가 크면 원래 graph가 맞힌 사례를 제거한 것이다. 고정 평균 adjacency는 학습 구간에서만 계산했으며 static graph 재학습과 같지 않다. zero_message는 실제 집계 메시지 m을 0으로 만들고 direct h/gate/weights는 유지한다. per-date/currency 기여는 각 run의 graph_contributions.parquet에 저장했다.','',
      '## 7. 기존 약 50.5% 해석과 유리한 범위','',
      '기존 전체 결과를 그대로 일반적인 예측 우위로 주장할 수 없다. 동일 조건의 no-graph·MLP·Transformer·train 빈도 baseline과 paired 결과를 우선 봐야 한다. 아래 범위는 사용자 요청에 따라 보고 구간에서 선택한 결과이며 그 선택 효과를 bootstrap이 보정하지 않는다.','']
    for policy in CFG['policies']:
        best=scopes[(scopes.policy==policy)&(scopes.objective=='minimum_advantage')].sort_values('minimum_advantage',ascending=False).head(3)
        lines += [f'### {policy}: 모든 지정 baseline 대비 최소 차이를 최대화한 조합','',tab(best[['candidate','start','end','n_dates','targets','mean_hit','minimum_advantage_pp','baseline_A3','baseline_mlp','baseline_transformer','baseline_train_probability']]),'']
        absolute=scopes[(scopes.policy==policy)&(scopes.objective=='hit')].sort_values('mean_hit',ascending=False).iloc[0]
        lines += [f"절대 Hit 최고는 {absolute.candidate}, {absolute.targets}, {absolute.start}–{absolute.end}의 {absolute.mean_hit:.5f}이며 지정 baseline 중 최고 대비 차이는 {absolute.minimum_advantage_pp:+.5f}%p다. 절대 Hit 최고와 baseline 대비 우위 최고는 서로 다른 목적이다.",'']
        chosen=best.iloc[0]
        relevant=scope_ci[(scope_ci.policy==policy)&(scope_ci.candidate==chosen.candidate)&(scope_ci.objective=='minimum_advantage')&(scope_ci.block==20)]
        lines += ['선택한 최고 조합의 조건부 bootstrap (범위 선택 효과를 보정하지 않음):','',tab(relevant[['comparator','effect_pp','ci_low_pp','ci_high_pp','winning_seeds']]),'']
        sg=scoped_seeds[(scoped_seeds.policy==policy)&(scoped_seeds.candidate==chosen.candidate)&(scoped_seeds.objective=='minimum_advantage')]
        lines += ['같은 조합의 모든 seed 결과:','',tab(sg[['seed','mean_hit','baseline_A3','baseline_mlp','baseline_transformer','baseline_train_probability']]),'']
    lines += [f"평가 universe는 최대 55개 strength-difference cross rates까지 진단했다. USD 대상 1,023개 부분집합, 37개 기간, 2개 보정 상태, 2개 가용성 정책을 포함해 {search['total_combinations']:,}개 행({search['unique_combinations']:,}개 고유 조합)을 계산했다. 많은 조합 중 최대값을 고른 것이므로 독립 검증과 구분해야 한다. 보정한 후보 행은 상대 neural 모델도 동일하게 보정한 결과와 비교한다.",'',
      'Cross-rate i/j의 logit은 z_i−z_j, 실제 target은 r_i,USD−r_j,USD다. 예를 들어 KRW/NOK는 NOK 대비 KRW 절상 방향을 예측한다. 모델 graph는 계속 11-node이며 학습 타깃은 동일한 USD 대비 10개 통화다. 선택 후 학습 대상을 바꿔 재학습한 결과가 아니다. 전체 조합은 ALL_scope_search.parquet, 세 목적별 순위는 selected_scopes_POSTHOC.csv, validation만으로 통화를 선택한 결과는 validation_selected_currencies.csv다.','',
      '## 8. 다음 실험 (최대 2개)','',
      '1. 가장 유망한 통화·기간 규칙을 지금 고정하고, 아직 보지 않은 이후 기간 또는 사전에 고정한 walk-forward 일정에서 모든 모델을 같은 방식으로 재학습·평가한다. 보고 label로 고른 성능이 미래에도 유지되는지 확인한다.',
      '2. 당일 spot 피드의 실제 timestamp 또는 FX vintage archive를 확보해 공개 가능 시각을 재구성하고, 그 정보집합에서 dynamic graph와 no-graph를 동일 조건으로 다시 비교한다. 가용성 불확실성이 architecture 효과와 섞이지 않게 한다.','',
      '## 재현과 파일','',
      '```powershell','python src/arcfx_followup.py','python src/arcfx_followup.py --analysis-only','```','',
      'manifest.json과 configs/arcfx_followup.json에 정의/해시/가용성 가정을 저장했다. 각 run에 초기화 해시·재사용 근거·best.pt·resume.pt·학습 추이·예측이 있다. 재사용은 원본 data/model hash, loss/graph/seed/설정, 예측 재계산 일치를 확인했다. 중단 후 완료 run을 건너뛰고 epoch checkpoint에서 재개한다. paired_effects.csv는 공통 5 seeds의 날짜별 평균 차이를 날짜 단위 5/20/60일 circular block으로 2,000회 재표집했다. Seed를 시장 표본처럼 취급하지 않았다.','']
    details=['## 시점별 구분 능력과 permutation 수치','']
    for policy in CFG['policies']:
        a=currencies[(currencies.policy==policy)&(currencies.condition=='A4')&(currencies.split=='report')&(currencies.variant=='original')]
        ca=currencies[(currencies.policy==policy)&(currencies.condition=='A4')&(currencies.split=='report')&(currencies.variant=='calibrated')]
        co=correlations[(correlations.policy==policy)&(correlations.condition=='A4')]
        sh=shifts[(shifts.policy==policy)&(shifts.condition=='A4')]
        orig=table[(table.policy==policy)&(table.condition=='A4')&(table.variant=='original')&(table.split=='report')]
        cal=table[(table.policy==policy)&(table.condition=='A4')&(table.variant=='calibrated')&(table.split=='report')]
        details += [f"{policy}: 통화 내 AUC 평균 {a.auc.mean():.5f} (보정 후 {ca.auc.mean():.5f}), logit/label residual correlation 평균 {co.logit_label_residual_corr.mean():.5f}. 원래 Hit {orig.mean_hit.mean():.5f}, calibration Hit {cal.mean_hit.mean():.5f}; 원래 BCE {orig.bce.mean():.5f}, calibration BCE {cal.bce.mean():.5f}. 실제 정렬 Hit의 seed 평균 {orig.mean_hit.mean():.5f}, shifted prediction Hit 평균 {sh.mean_hit.mean():.5f}.",'']
        if abs(a.auc.mean()-.5)<.02:
            details += ['통화 내 구분 능력이 0.5 근처에 머문다. 보정으로 확률/기본 방향을 바꿀 수 있어도, 이를 강한 날짜별 예측 신호의 증거로 설명하기는 어렵다. 보정 문제와 약한 구분 능력을 함께 검토해야 한다.','']
        t=trans[(trans.policy==policy)&(trans.condition=='A4')&(trans.split=='report')&(trans.intervention=='zero_message')]
        net=int(t.original_correct_to_wrong.sum()-t.original_wrong_to_correct.sum())
        details += [f"실제 메시지의 정답 순기여(원래 graph 정답→메시지 제거 오답 수 − 반대 전환 수)는 5 seeds 전체에서 {net:+d}건이다. 양수는 graph가 더 많이 맞힌 경우, 음수는 메시지를 제거한 쪽이 더 많이 맞힌 경우다. 반복 seed는 독립 시장 관측 수로 해석하지 않는다.",'']
    lines+=details
    (OUT/'FINAL_REPORT_KO.md').write_text('\n'.join(lines),encoding='utf-8')


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rel=pd.read_csv(OUT/'reliability.csv');currency=pd.read_csv(OUT/'currency_metrics_confusion.csv')
    fig,axes=plt.subplots(2,5,figsize=(15,6),sharex=True,sharey=True)
    for ax,ccy in zip(axes.flat,CCY[1:]):
        for policy,style in [('conservative_lag14','-'),('market_feed_assumed','--')]:
            for variant,color in [('original','tab:blue'),('calibrated','tab:orange')]:
                g=rel[(rel.currency==ccy)&(rel.policy==policy)&(rel.condition=='A4')&(rel.split=='report')&(rel.variant==variant)]
                xx=[];yy=[];counts=[]
                for _,b in g.groupby('bin'):
                    xx.append(np.average(b.mean_probability,weights=b['count']));yy.append(np.average(b.positive_fraction,weights=b['count']));counts.append(b['count'].sum())
                label=('lag14' if policy=='conservative_lag14' else 'feed')+'/'+('raw' if variant=='original' else 'cal')
                ax.plot(xx,yy,style,color=color,alpha=.7,label=label)
                sizes=8+80*np.sqrt(np.asarray(counts)/max(1,sum(counts)))
                ax.scatter(xx,yy,s=sizes,color=color)
        ax.plot([0,1],[0,1],':',color='gray');ax.set_title(ccy);ax.set_xlim(0,1);ax.set_ylim(0,1)
    axes.flat[0].legend(fontsize=6);fig.supxlabel('Predicted probability');fig.supylabel('Realized up fraction')
    fig.suptitle('A4 reliability: marker area reflects bin support; pooled five seeds',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'reliability.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(13,5))
    for ax,c in zip(axes.flat,CCY[1:]):
        g=currency[(currency.currency==c)&(currency.policy=='conservative_lag14')&(currency.condition=='A4')&(currency.split=='report')&(currency.variant=='original')]
        mat=np.array([[g.tn.sum(),g.fp.sum()],[g.fn.sum(),g.tp.sum()]])
        ax.imshow(mat,cmap='Blues');ax.set_title(c);ax.set_xticks([0,1],['Down','Up']);ax.set_yticks([0,1],['Down','Up'])
        for i,j in itertools.product(range(2),range(2)):ax.text(j,i,str(int(mat[i,j])),ha='center',va='center',fontsize=9)
    fig.supxlabel('Predicted (pooled 5 seeds, descriptive counts)');fig.supylabel('Actual binary direction')
    fig.tight_layout();fig.savefig(OUT/'confusion_A4.png',dpi=150);plt.close(fig)


def reuse_path(condition,seed):
    if condition=='A4':name='dense_signed'
    elif condition=='A3' and seed<3:name='winner_no_graph'
    else:return None
    path=ROOT/'results/arcfx_search/runs'/f'{name}__final__s{seed}'
    return path if (path/'result.json').exists() else None


def run(fold,condition,seed):
    dest=OUT/'runs'/fold.policy/f'{condition}_s{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():return
    graph=CONDITIONS.get(condition,{}).get('graph',False);lag=CONDITIONS.get(condition,{}).get('lag',0)
    dp.set_seed(seed);model=Model(fold.p.dims,graph) if condition in CONDITIONS else Baseline(fold.p.dims,condition)
    init_hash=state_hash(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
    old=reuse_path(condition,seed) if fold.policy=='market_feed_assumed' else None
    history=[];best=(-np.inf,-np.inf);beststate=None;noimp=0;first=1;rng=np.random.RandomState(seed)
    resume=dest/'resume.pt';start=time.perf_counter();status='trained'
    if old is not None:
        rr=json.loads((old/'result.json').read_text())
        assert rr['spec']['loss']=='bce' and rr['spec']['encoder']=='mlp'
        assert rr['spec'].get('no_graph',False)==(not graph)
        assert rr['spec']['graph']=='dense'
        for key in ['epochs','patience','batch_size','lr','weight_decay']:
            assert previous.CONFIG[key]==CFG[key]
        model.load_state_dict(torch.load(old/'best.pt',weights_only=True))
        saved=pd.read_parquet(old/'report.parquet')
        np.testing.assert_allclose(predict(model,fold,fold.report)[0].ravel(),saved.logit.values,atol=1e-7)
        history=pd.read_csv(old/'history.csv').to_dict('records');status='reused'
        log(policy=fold.policy,condition=condition,seed=seed,status='reused',path=str(old))
    else:
        log(policy=fold.policy,condition=condition,seed=seed,status='started')
        if resume.exists():
            ck=torch.load(resume,weights_only=False);model.load_state_dict(ck['state']);opt.load_state_dict(ck['opt'])
            best=ck['best'];beststate=ck['beststate'];history=ck['history'];noimp=ck['noimp'];first=ck['epoch']+1
            rng.set_state(ck['rng']);torch.set_rng_state(ck['torch_rng'])
        try:
            for epoch in range(first,CFG['epochs']+1):
                model.train();order=rng.permutation(fold.train);losses=[]
                for i in range(0,len(order),CFG['batch_size']):
                    ix=order[i:i+CFG['batch_size']];z=model(fold.batch(ix,lag))[:,1:]
                    loss=F.binary_cross_entropy_with_logits(z,(fold.y[ix,1:]>0).float())
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
                vz=predict(model,fold,fold.val,lag)[0];vm=metrics(vz,fold.p.y_raw[fold.val,1:])
                tz=predict(model,fold,fold.train,lag)[0];tm=metrics(tz,fold.p.y_raw[fold.train,1:])
                history.append(dict(epoch=epoch,train_loss=np.mean(losses),train_hit=tm['mean_hit'],val_hit=vm['mean_hit'],val_bce=vm['bce']))
                score=(vm['mean_hit'],-vm['bce'])
                if score>best:best=score;beststate=copy.deepcopy(model.state_dict());noimp=0
                else:noimp+=1
                temporary=resume.with_suffix('.tmp')
                torch.save(dict(state=model.state_dict(),opt=opt.state_dict(),best=best,beststate=beststate,
                    history=history,noimp=noimp,epoch=epoch,rng=rng.get_state(),torch_rng=torch.get_rng_state()),temporary);temporary.replace(resume)
                if noimp>=CFG['patience']:break
            model.load_state_dict(beststate)
        except Exception:
            log(policy=fold.policy,condition=condition,seed=seed,status='failed',error=traceback.format_exc());raise
    torch.save(model.state_dict(),dest/'best.pt');pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
    result=dict(policy=fold.policy,condition=condition,seed=seed,lag=lag,graph=graph,status=status,reuse_source=str(old) if old else None,
                seconds=time.perf_counter()-start,epochs=len(history),initial_state_hash=init_hash,
                best_epoch=max(history,key=lambda r:(r['val_hit'],-r['val_bce']))['epoch'],
                parameters=sum(p.numel() for p in model.parameters()),splits={})
    for split,ix in [('train',fold.train),('validation',fold.val),('report',fold.report)]:
        z,_=predict(model,fold,ix,lag)
        frame(fold,ix,z,condition,seed).to_parquet(dest/(split+'.parquet'),index=False)
        result['splits'][split]=metrics(z,fold.p.y_raw[ix,1:])
    dump(dest/'result.json',result)
    log(policy=fold.policy,condition=condition,seed=seed,status='completed',origin=status)
    print(fold.policy,condition,seed,status,'report',round(result['splits']['report']['mean_hit'],5),flush=True)


def manifest(folds):
    paths=[ROOT/'data/processed/fx_panel_pit.csv',ROOT/'data/processed/nonfx_features_pit.csv',
           ROOT/'src/arcfx_search_models.py',ROOT/'src/data_pipeline.py',ROOT/'src/arcfx_followup.py']
    old=json.loads((ROOT/'results/arcfx_search/manifest.json').read_text())
    checks={p:sha(ROOT/p)==v for p,v in old.items() if p.endswith('fx_panel_pit.csv') or p.endswith('nonfx_features_pit.csv') or p.endswith('arcfx_search_models.py')}
    assert all(checks.values()),checks
    f=folds['conservative_lag14']
    rules=dict(decision_timestamp='panel date t, 12:00 America/New_York; non-FX builder cutoff 11:30 ET',
         return_start='t 12:00 America/New_York',return_end='Target_Date 12:00 America/New_York',
         currencies=CCY,quotation='positive = local currency appreciation against USD; USD-per-foreign EUR/GBP/AUD/NZD, inverse other FX quotes',
         direction_label='raw return > 0',zero_target='negative binary class; legacy three-way sign-hit requires exactly zero logit to hit zero return',
         zero_logit='binary down (z>0 false), legacy sign 0',mean_hit='equal-weight dates and target currencies, np.sign(z)==np.sign(r)',
         horizon='next panel noon fixing',lookback=10,common_training_target_rows=[int(f.train[0]),int(f.train[-1])],
         train=[str(f.p.merged.Date.iloc[f.train[0]]),str(f.p.merged.Date.iloc[f.train[-1]])],
         validation=[str(f.p.merged.Date.iloc[f.val[0]]),str(f.p.merged.Date.iloc[f.val[-1]])],
         report_return_end=[str(f.p.merged.Target_Date.iloc[f.report[0]]),str(f.p.merged.Target_Date.iloc[f.report[-1]])],
         target_counts=dict(train=len(f.train),validation=len(f.val),report=len(f.report)),
         scaler='train-only feature mean/std, missing -> standardized zero; lag applied before scaler in conservative policy',
         feature_lists=dict(local=dp.LOCAL_FEATURE_TEMPLATES,rate=dp.RATE_LOCAL_TEMPLATES+dp.RATE_GLOBAL_COLUMNS,
                            equity=dp.EQUITY_TEMPLATES,countrymacro=dp.COUNTRY_MACRO_TEMPLATES,global_features=dp.GLOBAL_COLUMNS),
         regimes='six causal regime indicators, thresholds from training only',
         conservative_policy='FX-derived features observed at least 14 calendar days earlier; all non-FX supplied PIT features an additional panel-date lag; regimes recomputed after these lags',
         conservative_limitation='14-day schedule buffer, not per-release vintage certification; 2006-2009 H10 weekly publication interruption and subsequent revisions remain unverified',
         conditional_policy='same inputs/scalers as prior exploration; current noon spot requires an unverified contemporaneous market feed',
         no_graph='zero cross-currency message at training and inference; no per-node pooling or foreign-node gate; predefined common market/regime inputs and USD gauge retained',
         public_source='https://www.federalreserve.gov/releases/h10/default.htm',
         source_fact='H10 previous-week daily rates published Mondays 16:15 ET, next business day if federal holiday; weekly release interrupted May 2006-Jan 2009',
         subset_authorization='User explicitly permits exploratory choice of evaluation period and currency subset; all models share each selected denominator; all seeds retained.',
         selection_caveat='Subset ranking uses report labels; descriptive selected-sample advantage, not independent generalization or selection-corrected significance.')
    dump(OUT/'manifest.json',dict(config=CFG,evaluation=rules,hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},training_logic_sha256=training_logic_hash(),
         prior_reuse_hash_checks=checks,availability_violations=pd.read_csv(ROOT/'data/processed/availability_audit.csv').filter(like='violations').sum().to_dict(),
         python=platform.python_version(),torch=torch.__version__,numpy=np.__version__))
    dump(OUT/'conservative_scalers.json',f.stats)
    dump(ROOT/'configs/arcfx_followup.json',CFG)


def training_logic_hash():
    body='\n'.join(inspect.getsource(obj) for obj in [raw_conservative,Fold,Model,Baseline,run,metrics])
    body+=json.dumps(CFG,sort_keys=True)
    body+=sha(ROOT/'src/data_pipeline.py')+sha(ROOT/'src/arcfx_search_models.py')+sha(ROOT/'src/baselines/models.py')
    return hashlib.sha256(body.encode()).hexdigest()


def verify_contract():
    path=OUT/'manifest.json'
    if not path.exists():return
    m=json.loads(path.read_text())
    for rel,value in m['hashes'].items():
        if rel.endswith('arcfx_followup.py'):continue
        assert sha(ROOT/rel)==value,('input/source changed; use a new experiment directory',rel)
    assert m['config']==CFG,'Configuration changed: use a new experiment directory'
    if 'training_logic_sha256' in m:assert m['training_logic_sha256']==training_logic_hash(),'Training logic changed: use a new experiment directory'


def smoke(fold):
    dp.set_seed(0);a=Model(fold.p.dims,True);dp.set_seed(0);b=Model(fold.p.dims,False)
    assert state_hash(a.state_dict())==state_hash(b.state_dict())
    ix=fold.train[30:34];batch=fold.batch(ix);a.eval();b.eval()
    assert torch.equal(batch['x_local'][:,-1],torch.from_numpy(fold.p.x_local[ix]))
    assert torch.equal(fold.batch(ix,1)['x_local'][:,-1],torch.from_numpy(fold.p.x_local[ix-1]))
    with torch.no_grad():
        za=a(batch);zb=b(batch);zero=a(batch,intervention='zero_message')
        torch.testing.assert_close(zero,zb,rtol=0,atol=0)
        altered={k:v.clone() for k,v in batch.items()};altered['x_local'][:,:,2]+=3
        changed=b(altered);keep=[i for i in range(11) if i!=2]
        torch.testing.assert_close(changed[:,keep],zb[:,keep],rtol=0,atol=0)
        assert float((za-zb).abs().max())>1e-7
    loss=F.binary_cross_entropy_with_logits(a(batch)[:,1:],(fold.y[ix,1:]>0).float())
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in a.parameters() if p.grad is not None)
    dump(OUT/'smoke.json',dict(passed=True,checks=['shared initialization','exact message-zero equivalence','no foreign-node bypass','input alignment','finite BCE gradient'],neural_fits=0))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    verify_contract()
    folds={p:Fold(p) for p in CFG['policies']}
    if not args.analysis_only:
        manifest(folds);smoke(folds['conservative_lag14'])
        for policy,fold in folds.items():
            for seed in CFG['seeds']:
                for condition in CONDITIONS:run(fold,condition,seed)
            for seed in CFG['seeds']:
                for condition in CFG['matched_baselines']:run(fold,condition,seed)
    analyze(folds)


if __name__=='__main__':main()
