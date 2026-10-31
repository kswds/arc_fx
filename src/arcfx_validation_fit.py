"""Fit on the former validation set and evaluate on exactly the same rows.

This is an explicitly requested resubstitution diagnostic, not a held-out test.
Existing checkpoints' validation scores are reported separately for comparison.
"""
import copy
import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import arcfx_followup as f

OUT=f.ROOT/'results/arcfx_validation_fit'
MODELS=['dynamic_graph','no_graph','mlp','transformer']
OLD_NAMES=dict(dynamic_graph='A4',no_graph='A3',mlp='mlp',transformer='transformer')
CFG=dict(learning=dict(epochs=20,patience=4,batch_size=256,lr=.0003,weight_decay=.0001,
                      clip_grad_norm=1.,loss='unweighted raw-return direction BCE'),
         seeds=list(range(5)),models=MODELS,policies=f.CFG['policies'],
         initialization='fresh seeded initialization, no pretrained weights',
         fit_rows='all 527 former validation decision dates',
         evaluation_rows='identical to fitting rows; resubstitution',
         checkpoint='same-set sign-hit maximum, tie lower same-set BCE',
         feature_scaling='unchanged: original pre-validation train-only scalers and regime thresholds',
         primary_targets='all ten non-USD currency returns against USD',
         secondary_targets='all 55 strength-difference currency pairs, no target filtering',
         known_overlap=True,independent_test=False)


def event(**values):
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(time=time.time(),**values),default=str)+'\n')


def create_model(fold,name):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='enable_nested_tensor is True')
        return f.Model(fold.p.dims,name=='dynamic_graph') if name in ['dynamic_graph','no_graph'] else f.Baseline(fold.p.dims,name)


def pair_outputs(z,y):
    zz=np.c_[np.zeros(len(z)),z];yy=np.c_[np.zeros(len(y)),y]
    pairs=[(i,0) for i in range(1,11)]+[(i,j) for i in range(1,11) for j in range(i+1,11)]
    return np.stack([zz[:,i]-zz[:,j] for i,j in pairs],-1),np.stack([yy[:,i]-yy[:,j] for i,j in pairs],-1),[f.CCY[i]+'/'+f.CCY[j] for i,j in pairs]


def fit_one(fold,name,seed):
    dest=OUT/'runs'/fold.policy/f'{name}_s{seed}'
    if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
    dest.mkdir(parents=True,exist_ok=True);f.dp.set_seed(seed)
    model=create_model(fold,name);initial_hash=f.state_hash(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    rng=np.random.RandomState(seed);ix=fold.val
    best=(-np.inf,-np.inf);best_state=None;history=[];noimp=0;first=1;resume=dest/'resume.pt'
    if resume.exists():
        ck=torch.load(resume,weights_only=False);model.load_state_dict(ck['state']);optimizer.load_state_dict(ck['optimizer'])
        rng.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);best=ck['best'];best_state=ck['best_state']
        history=ck['history'];noimp=ck['noimp'];first=ck['epoch']+1
    start=time.perf_counter();event(policy=fold.policy,model=name,seed=seed,status='started',first_epoch=first)
    try:
        for epoch in range(first,21):
            order=rng.permutation(ix);model.train();loss_sum=0.;sample_count=0
            for i in range(0,len(order),256):
                rows=order[i:i+256];z=model(fold.batch(rows))[:,1:]
                loss=F.binary_cross_entropy_with_logits(z,(fold.y[rows,1:]>0).float())
                if not torch.isfinite(loss):raise RuntimeError('nonfinite BCE')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                loss_sum+=float(loss.detach())*len(rows);sample_count+=len(rows)
            z,_=f.predict(model,fold,ix);m=f.metrics(z,fold.p.y_raw[ix,1:])
            history.append(dict(epoch=epoch,training_bce=loss_sum/sample_count,**m))
            score=(m['mean_hit'],-m['bce'])
            if score>best:best=score;best_state=copy.deepcopy(model.state_dict());noimp=0
            else:noimp+=1
            temp=resume.with_suffix('.tmp')
            torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),numpy_rng=rng.get_state(),
                torch_rng=torch.get_rng_state(),best=best,best_state=best_state,history=history,noimp=noimp,epoch=epoch),temp);temp.replace(resume)
            pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
            if noimp>=4:break
        model.load_state_dict(best_state);torch.save(best_state,dest/'best.pt')
        z,_=f.predict(model,fold,ix);y=fold.p.y_raw[ix,1:]
        prediction=f.frame(fold,ix,z,name,seed,'validation_fit_and_eval')
        prediction.to_parquet(dest/'same_set_predictions.parquet',index=False)
        pz,py,names=pair_outputs(z,y)
        pair_frame=pd.DataFrame(dict(decision_date=np.repeat(fold.p.merged.Date.iloc[ix].astype(str).values,55),
            target_date=np.repeat(fold.p.merged.Target_Date.iloc[ix].astype(str).values,55),
            pair=np.tile(names,len(ix)),logit=pz.ravel(),target=py.ravel(),seed=seed))
        pair_frame.to_parquet(dest/'all_55_pairs.parquet',index=False)
        result=dict(policy=fold.policy,model=name,seed=seed,initial_state_hash=initial_hash,
            n_fit_dates=len(ix),n_evaluation_dates=len(ix),primary_samples=z.size,fit_eval_overlap_fraction=1.,
            epochs=len(history),best_epoch=max(history,key=lambda row:(row['mean_hit'],-row['bce']))['epoch'],
            seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),
            all_10_usd=f.metrics(z,y),all_55_pairs=f.metrics(pz,py))
        f.dump(dest/'result.json',result);event(policy=fold.policy,model=name,seed=seed,status='completed')
        print(fold.policy,name,seed,'same-set hit',round(result['all_10_usd']['mean_hit'],5),flush=True)
        return result
    except Exception:
        event(policy=fold.policy,model=name,seed=seed,status='failed',error=traceback.format_exc());raise


def summarize(folds):
    rows=[];currency=[];pairrows=[];old=[];baseline=[]
    for policy,fold in folds.items():
        y=fold.p.y_raw[fold.val,1:]
        p=np.clip((y>0).mean(0),1e-6,1-1e-6);z=np.tile(np.log(p/(1-p)),(len(y),1))
        baseline.append(dict(policy=policy,model='validation_fitted_constant_probability',
                             **f.metrics(z,y,np.where(z>0,1,-1))))
        for name in MODELS:
            for seed in CFG['seeds']:
                dest=OUT/'runs'/policy/f'{name}_s{seed}';r=json.loads((dest/'result.json').read_text())
                for scope in ['all_10_usd','all_55_pairs']:
                    rows.append(dict(policy=policy,model=name,seed=seed,scope=scope,epochs=r['epochs'],best_epoch=r['best_epoch'],seconds=r['seconds'],**r[scope]))
                frame=pd.read_parquet(dest/'same_set_predictions.parquet')
                for c,g in frame.groupby('currency'):
                    currency.append(dict(policy=policy,model=name,seed=seed,currency=c,**f.metrics(g.logit.values,g.target.values)))
                frame=pd.read_parquet(dest/'all_55_pairs.parquet')
                for pair,g in frame.groupby('pair'):
                    pairrows.append(dict(policy=policy,model=name,seed=seed,pair=pair,**f.metrics(g.logit.values,g.target.values)))
                source=f.OUT/'runs'/policy/f'{OLD_NAMES[name]}_s{seed}'/'validation.parquet'
                original,oz,oy=f.read_grid(source)
                np.testing.assert_array_equal(y,oy)
                old.append(dict(policy=policy,model=name,seed=seed,**f.metrics(oz,oy)))
    table=pd.DataFrame(rows);table.to_csv(OUT/'all_results.csv',index=False)
    pd.DataFrame(currency).to_csv(OUT/'currency_results.csv',index=False)
    pd.DataFrame(pairrows).to_csv(OUT/'pair_results.csv',index=False)
    pd.DataFrame(old).to_csv(OUT/'original_checkpoint_validation.csv',index=False)
    pd.DataFrame(baseline).to_csv(OUT/'validation_fitted_constant_baseline.csv',index=False)
    aggregate=table.groupby(['policy','model','scope']).agg(mean_hit=('mean_hit','mean'),seed_std=('mean_hit','std'),
         binary_accuracy=('binary_accuracy','mean'),bce=('bce','mean'),seeds=('seed','count')).reset_index()
    aggregate.to_csv(OUT/'aggregate.csv',index=False)
    original=pd.DataFrame(old).groupby(['policy','model']).agg(mean_hit=('mean_hit','mean'),seed_std=('mean_hit','std'),
         binary_accuracy=('binary_accuracy','mean'),bce=('bce','mean')).reset_index()
    original.to_csv(OUT/'original_checkpoint_validation_aggregate.csv',index=False)
    fold=next(iter(folds.values()));date=fold.p.merged.Date.iloc[fold.val];end=fold.p.merged.Target_Date.iloc[fold.val]
    def tab(df):return df.to_markdown(index=False,floatfmt='.5f')
    text=['# Validation 구간으로 학습하고 같은 구간에서 평가한 결과','',
      '사용자 요청을 두 가지로 구분하여 모두 계산했다. (1) 이전 train에서 학습하고 validation으로 checkpoint를 선택한 기존 모델의 validation 성능, (2) 동일한 모델 조건으로 새로 초기화하여 validation label로 학습하고 정확히 같은 validation 날짜에서 평가한 성능.','',
      '**두 번째 실험은 fit/evaluation overlap 100%인 resubstitution이다. 이 수치를 독립 test 성능으로 해석할 수 없다. 첫 번째 수치도 checkpoint 선택에 사용한 validation 점수다.**','',
      f'Decision date: {date.min().date()}–{date.max().date()}; return 종료일: {pd.to_datetime(end).min().date()}–{pd.to_datetime(end).max().date()}. 전체 {len(date)}일, 10개 비USD 통화 = {len(date)*10:,}개 날짜·통화 표본. 보조 지표는 전체 55개 통화쌍 = {len(date)*55:,}개 표본.','',
      '기간·통화·seed를 제외하지 않았다. Seeds 0,1,2,3,4를 모두 평균했다. 각 seed의 fresh initialization과 shuffle은 graph/no-graph 간 동일하다. MLP hidden32/lookback10/dropout0.25, 전체 USD-target raw BCE, AdamW lr0.0003/weight_decay0.0001, batch256, max20 epochs, patience4를 유지했다. Checkpoint는 같은 학습·평가 구간의 sign-hit 최대, 동률 BCE 최소로 고른다. 원래 pre-validation train-only scaler/regime은 그대로 사용했다. Validation label만 optimizer 학습에 사용하며 이전 checkpoint로 fine-tuning하지 않았다.','',
      '## Validation에서 새로 학습 → 같은 validation에서 평가: 전체 10개 USD 대상','',tab(aggregate[aggregate.scope=='all_10_usd']),'',
      '## 동일 checkpoint의 전체 55개 strength-difference 통화쌍','',tab(aggregate[aggregate.scope=='all_55_pairs']),'',
      '55개 통화쌍은 별도 학습하지 않고 latent logit 차분으로 생성했다. 주 학습 손실은 기존과 같이 USD 대비 10개 통화에 적용했다. 어느 통화쌍도 평가에서 제외하지 않았다.','',
      '## 비교: 기존 train에서 학습한 checkpoint를 validation 전체에서 평가','',tab(original),'',
      '## 같은 validation에서 추정한 상수 확률 baseline','',tab(pd.DataFrame(baseline)[['policy','mean_hit','binary_accuracy','bce']]),'',
      'Mean Hit는 np.sign(logit)==np.sign(raw return), Binary accuracy는 (logit>0)==(raw return>0)다. 0 target을 제외하지 않았다. 상수 확률 baseline은 p>0.5일 때 상승, 나머지는 하락이다. 방향 logits에 수익률 RMSE를 붙이지 않았다. 보수적/시장피드 입력 가정은 기존 후속 진단 프로토콜을 그대로 따른다.','',
      '## 재현','', '```powershell','python src/arcfx_validation_fit.py','```','',
      'run별 best.pt/resume.pt/history.csv와 예측을 보존했다. aggregate.csv, all_results.csv, currency_results.csv, pair_results.csv, original_checkpoint_validation_aggregate.csv에 전체 결과가 있다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(text),encoding='utf-8')
    print(aggregate[aggregate.scope=='all_10_usd'].to_string(index=False),flush=True)
    f.dump(OUT/'completion.json',dict(new_neural_fits=40,failed_runs=sum(json.loads(line).get('status')=='failed' for line in (OUT/'ledger.jsonl').read_text().splitlines()),
           former_validation_dates=len(fold.val),primary_targets=10,secondary_pairs=55,fit_eval_overlap_fraction=1.))


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);f.verify_contract()
    expected=dict(config=CFG,source_sha256=f.sha(Path(__file__)),
                  followup_training_logic_sha256=f.training_logic_hash(),input_hashes={name:f.sha(f.ROOT/name) for name in ['data/processed/fx_panel_pit.csv','data/processed/nonfx_features_pit.csv']})
    manifest=OUT/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==expected,'Use a new output directory after configuration/source changes.'
    else:f.dump(manifest,expected)
    folds={policy:f.Fold(policy) for policy in CFG['policies']}
    for fold in folds.values():
        for seed in CFG['seeds']:
            for name in MODELS:fit_one(fold,name,seed)
    summarize(folds)


if __name__=='__main__':main()
