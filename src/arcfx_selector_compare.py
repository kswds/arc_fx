"""Controlled graph-selector comparison across two backbones and two validation protocols.

Only validation labels choose checkpoints and the common soft threshold. Test maxima
over the five selected seeds are explicitly descriptive/post-hoc maxima.
"""
from __future__ import annotations
import argparse,copy,hashlib,inspect,json,math,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import data_pipeline as dp
import training_core as tc
import arcfx_followup as helpers
from train import as_namespace,load_config
from arcfx_selector_models import CompactSelector,OriginalSelector,select_edges

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/arcfx_selector_compare'
CCY=helpers.CCY
CFG=dict(backbones=['compact_mlp','original_arcfx'],seeds=list(range(5)),top_k=3,
         tau_grid=[0.,.025,.05,.1,.2],epochs=20,patience=4,batch_size=256,
         lr=.0003,weight_decay=.0001,clip_grad_norm=1.,threads=2,
         lookback=10,input_lag=0,input_policy='supplied PIT features, current decision row; same for all candidates, no new lag14',
         scaler_split=[.6,.2,.2],tau_selection='maximum five-seed mean validation sign-hit; tie lower mean validation objective, then smaller tau',
         checkpoint='validation sign-hit maximum; tie lower validation objective',
         optimizer='AdamW constant learning rate; no validation-driven LR scheduler',
         overlap='same train and test, same validation size; shift validation forward by floor(test length / 2)',
         primary_targets='all 10 non-USD currencies against USD; no period/target filtering',
         maximum_reporting='highest test seed within validation-selected model settings; marked post hoc',
         bootstrap_repeats=2000,blocks=[5,20,60])


def dump(path,value):helpers.dump(path,value)
def sha(path):return helpers.sha(path)
def event(**values):
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),**values),default=str)+'\n')


class Data:
    def __init__(self):
        self.p=dp.prepare_data(str(ROOT/'data/processed/fx_panel_pit.csv'),
            str(ROOT/'data/processed/nonfx_features_pit.csv'),CCY,10,[.6,.2,.2],True)
        n=len(self.p.merged)-10;nt=int(n*.6);nv=int(n*.2)
        self.train=np.arange(10,10+nt)
        original=np.arange(10+nt,10+nt+nv)
        self.test=np.arange(10+nt+nv,len(self.p.merged))
        shift=len(self.test)//2
        self.validation=dict(pit_original=original,overlap50=original+shift)
        self.overlap=np.intersect1d(self.validation['overlap50'],self.test)
        self.future=np.setdiff1d(self.test,self.overlap)
        self.y=torch.from_numpy(self.p.y_raw);self.yn=torch.from_numpy(self.p.y_norm)
        self.blocks={}
        for key in ['local','rate','equity','countrymacro','global']:
            w=np.lib.stride_tricks.sliding_window_view(getattr(self.p,'x_'+key),10,axis=0)
            self.blocks['x_'+key]=torch.from_numpy(np.moveaxis(w,-1,1).copy())
        cfg=load_config(ROOT/'configs/pit_main.yaml')
        self.loss_args=as_namespace(cfg,0,'oursmain','cpu',None,None)
        self.threshold=tc.compute_small_return_threshold(self.p,[.6,.2,.2],.4)
        self.non_usd=torch.tensor([False]+[True]*10)
    def batch(self,ix):return {k:v[ix-9-CFG['input_lag']] for k,v in self.blocks.items()}
    def dates(self,ix):
        return dict(n_dates=len(ix),decision_start=str(self.p.merged.Date.iloc[ix[0]].date()),
             decision_end=str(self.p.merged.Date.iloc[ix[-1]].date()),
             return_end_start=str(self.p.merged.Target_Date.iloc[ix[0]]),return_end_end=str(self.p.merged.Target_Date.iloc[ix[-1]]))


def candidates():
    return [('topk',None),('dense_signed',None)]+[('soft_threshold',tau) for tau in CFG['tau_grid']]
def candidate_name(mode,tau):return mode if tau is None else f'soft_threshold_tau{tau:g}'
def create(backbone,data,mode,tau):
    cls=CompactSelector if backbone=='compact_mlp' else OriginalSelector
    return cls(data.p.dims,CCY,mode,0. if tau is None else tau)


def objective(backbone,out,data,ix):
    if backbone=='compact_mlp':return F.binary_cross_entropy_with_logits(out['rhat'][:,1:],(data.y[ix,1:]>0).float())
    return tc.compute_relational_loss('oursmain',out,data.yn[ix],data.non_usd,data.p,data.loss_args,data.threshold)


@torch.no_grad()
def evaluate(model,backbone,data,ix):
    model.eval();raw=[];model_outputs=[];losses=[];weights=[];adj=[];gates=[];messages=[];prev=None
    for start in range(0,len(ix),256):
        rows=ix[start:start+256];batch=data.batch(rows)
        if backbone=='original_arcfx' and prev is not None:batch['A_prev']=prev
        out=model(batch);margin=out['rhat'].numpy()[:,1:]
        pred=margin if backbone=='compact_mlp' else margin*data.p.y_std[1:]+data.p.y_mean[1:]
        raw.append(pred);model_outputs.append(margin)
        losses.append(float(objective(backbone,out,data,rows)));weights.append(len(rows))
        adj.append(out['adj'].numpy());gates.append(out['graph_gate'].numpy());messages.append(out['message'].numpy())
        prev=out['adj'].detach()
    pred=np.concatenate(raw);margin=np.concatenate(model_outputs);y=data.p.y_raw[ix,1:]
    a=np.concatenate(adj);g=np.concatenate(gates);m=np.concatenate(messages)
    off=~np.eye(11,dtype=bool)
    metrics=dict(mean_hit=float((np.sign(pred)==np.sign(y)).mean()),
         binary_accuracy=float(((pred>0)==(y>0)).mean()),validation_objective=float(np.average(losses,weights=weights)),
         zero_return_fraction=float((y==0).mean()),pred_positive=float((pred>0).mean()),actual_positive=float((y>0).mean()),
         active_edge_fraction=float((a[:,off]!=0).mean()),mean_neighbors=float((a!=0).sum(-1).mean()),
         zero_neighbor_fraction=float(((a!=0).sum(-1)==0).mean()),row_l1_max=float(a.__abs__().sum(-1).max()),
         message_rms=float(np.sqrt(np.mean(m*m))),gate_mean=float(g.mean()),
         gate_q01=float(np.quantile(g,.01)),gate_q99=float(np.quantile(g,.99)),
         prediction_std=float(pred.std()),model_output_std=float(margin.std()))
    return metrics,pred,margin


def write_predictions(path,data,ix,pred,margin,backbone,mode,tau,seed,protocol):
    frame=pd.DataFrame(dict(decision_date=np.repeat(data.p.merged.Date.iloc[ix].astype(str).values,10),
        input_end_date=np.repeat(data.p.merged.Date.iloc[ix-CFG['input_lag']].astype(str).values,10),
        target_date=np.repeat(data.p.merged.Target_Date.iloc[ix].astype(str).values,10),currency=np.tile(CCY[1:],len(ix)),
        prediction=pred.ravel(),model_output=margin.ravel(),target=data.p.y_raw[ix,1:].ravel(),
        seed=seed,backbone=backbone,selector=mode,tau=tau,protocol=protocol,
        validation_overlap=np.repeat(np.isin(ix,data.validation[protocol]),10)))
    frame['label']=(frame.target>0).astype(int)
    path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path,index=False)


def run_trajectory(data,backbone,mode,tau,seed):
    name=candidate_name(mode,tau);dest=OUT/'trajectories'/backbone/f'{name}_s{seed}'
    if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
    dest.mkdir(parents=True,exist_ok=True);dp.set_seed(seed);model=create(backbone,data,mode,tau)
    initial_hash=helpers.state_hash(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
    rng=np.random.RandomState(seed);first=1;history=[]
    states={key:dict(best=(-np.inf,-np.inf),weights=None,noimp=0,stopped=False,best_epoch=0,stop_epoch=None) for key in data.validation}
    resume=dest/'resume.pt';started=time.perf_counter()
    if resume.exists():
        ck=torch.load(resume,weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['opt'])
        rng.set_state(ck['rng']);torch.set_rng_state(ck['torch_rng']);first=ck['epoch']+1;states=ck['protocol_states'];history=ck['history']
    event(backbone=backbone,selector=mode,tau=tau,seed=seed,status='started',first_epoch=first)
    initial_metrics=evaluate(model,backbone,data,data.train[:256])[0] if first==1 else None
    try:
        for epoch in range(first,CFG['epochs']+1):
            model.train();order=rng.permutation(data.train);loss_total=0.;seen=0;prev=None
            for i in range(0,len(order),256):
                rows=order[i:i+256];batch=data.batch(rows)
                if backbone=='original_arcfx' and prev is not None:batch['A_prev']=prev
                out=model(batch);loss=objective(backbone,out,data,rows)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite objective')
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                prev=out['adj'].detach();loss_total+=float(loss.detach())*len(rows);seen+=len(rows)
            record=dict(epoch=epoch,train_objective=loss_total/seen)
            for protocol,validation in data.validation.items():
                state=states[protocol]
                if state['stopped']:continue
                vm=evaluate(model,backbone,data,validation)[0]
                record[protocol+'_hit']=vm['mean_hit'];record[protocol+'_objective']=vm['validation_objective']
                score=(vm['mean_hit'],-vm['validation_objective'])
                if score>state['best']:
                    state.update(best=score,weights=copy.deepcopy(model.state_dict()),noimp=0,best_epoch=epoch)
                else:state['noimp']+=1
                if state['noimp']>=CFG['patience']:state.update(stopped=True,stop_epoch=epoch)
            history.append(record);pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
            temporary=resume.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),rng=rng.get_state(),torch_rng=torch.get_rng_state(),
                epoch=epoch,protocol_states=states,history=history),temporary);temporary.replace(resume)
            if all(s['stopped'] for s in states.values()):break
        result=dict(backbone=backbone,selector=mode,tau=tau,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
             initial_state_hash=initial_hash,initial_training_diagnostics=initial_metrics,trained_epochs=len(history),
             seconds=time.perf_counter()-started,protocols={})
        for protocol,state in states.items():
            model.load_state_dict(state['weights']);path=dest/protocol;path.mkdir(parents=True,exist_ok=True)
            torch.save(state['weights'],path/'best.pt')
            m,pred,margin=evaluate(model,backbone,data,data.validation[protocol])
            write_predictions(path/'validation.parquet',data,data.validation[protocol],pred,margin,backbone,mode,tau,seed,protocol)
            result['protocols'][protocol]=dict(validation=m,best_epoch=state['best_epoch'],stop_epoch=state['stop_epoch'] or len(history))
        dump(dest/'result.json',result);event(backbone=backbone,selector=mode,tau=tau,seed=seed,status='completed',seconds=result['seconds'])
        print(backbone,name,'seed',seed,'epochs',len(history),'val',
              {p:round(r['validation']['mean_hit'],5) for p,r in result['protocols'].items()},'sec',round(result['seconds']),flush=True)
        return result
    except Exception:
        event(backbone=backbone,selector=mode,tau=tau,seed=seed,status='failed',error=traceback.format_exc());raise


def select_models(data):
    records=[json.loads(p.read_text()) for p in (OUT/'trajectories').glob('*/*/result.json')]
    rows=[]
    for r in records:
        for protocol,v in r['protocols'].items():
            rows.append(dict(backbone=r['backbone'],selector=r['selector'],tau=r['tau'],seed=r['seed'],protocol=protocol,
                            best_epoch=v['best_epoch'],**v['validation']))
    table=pd.DataFrame(rows);table.to_csv(OUT/'all_validation_results.csv',index=False)
    selection=[]
    for backbone in CFG['backbones']:
        for protocol in data.validation:
            t=table[(table.backbone==backbone)&(table.protocol==protocol)&(table.selector=='soft_threshold')]
            ranked=t.groupby('tau').agg(mean_hit=('mean_hit','mean'),mean_objective=('validation_objective','mean'),count=('seed','count')).reset_index()
            assert ranked['count'].eq(5).all()
            ranked=ranked.sort_values(['mean_hit','mean_objective','tau'],ascending=[False,True,True])
            tau=float(ranked.iloc[0].tau)
            for mode,value in [('topk',None),('dense_signed',None),('soft_threshold',tau)]:
                selection.append(dict(backbone=backbone,protocol=protocol,selector=mode,tau=value,
                    source='validation only; five-seed mean',test_results_used=False))
    dump(OUT/'selection_frozen.json',selection)
    return selection


def test_selected(data,selection):
    rows=[];currency=[];years=[]
    for choice in selection:
        backbone=choice['backbone'];protocol=choice['protocol'];mode=choice['selector'];tau=choice['tau']
        name=candidate_name(mode,tau)
        for seed in CFG['seeds']:
            model=create(backbone,data,mode,tau)
            trajectory=OUT/'trajectories'/backbone/f'{name}_s{seed}'
            rr=json.loads((trajectory/'result.json').read_text())
            model.load_state_dict(torch.load(trajectory/protocol/'best.pt',weights_only=True))
            path=OUT/'selected_runs'/backbone/protocol/f'{mode}_s{seed}';path.mkdir(parents=True,exist_ok=True)
            result=dict(**choice,seed=seed,best_epoch=rr['protocols'][protocol]['best_epoch'],parameters=rr['parameters'],subsets={})
            for label,ix in [('test_all',data.test),('test_common_early_half',data.overlap),('test_common_late_half',data.future)]:
                m,pred,margin=evaluate(model,backbone,data,ix)
                write_predictions(path/(label+'.parquet'),data,ix,pred,margin,backbone,mode,tau,seed,protocol)
                result['subsets'][label]=m
                rows.append(dict(backbone=backbone,protocol=protocol,selector=mode,tau=tau,seed=seed,subset=label,n_dates=len(ix),best_epoch=result['best_epoch'],**m))
                if label=='test_all':
                    f=pd.read_parquet(path/(label+'.parquet'))
                    for c,g in f.groupby('currency'):
                        currency.append(dict(backbone=backbone,protocol=protocol,selector=mode,tau=tau,seed=seed,currency=c,
                            mean_hit=float((np.sign(g.prediction)==np.sign(g.target)).mean())))
                    f['year']=pd.to_datetime(f.target_date).dt.year
                    for year,g in f.groupby('year'):
                        years.append(dict(backbone=backbone,protocol=protocol,selector=mode,tau=tau,seed=seed,year=int(year),n=len(g),
                            mean_hit=float((np.sign(g.prediction)==np.sign(g.target)).mean())))
            dump(path/'result.json',result)
    table=pd.DataFrame(rows);table.to_csv(OUT/'test_results.csv',index=False)
    pd.DataFrame(currency).to_csv(OUT/'currency_results.csv',index=False)
    pd.DataFrame(years).to_csv(OUT/'year_results.csv',index=False)
    summaries=[]
    for keys,g in table.groupby(['backbone','protocol','selector','subset']):
        best=g.sort_values(['mean_hit','seed'],ascending=[False,True]).iloc[0]
        summaries.append(dict(zip(['backbone','protocol','selector','subset'],keys))|
            dict(tau=None if pd.isna(best.tau) else float(best.tau),mean_hit=float(g.mean_hit.mean()),seed_std=float(g.mean_hit.std()),
                 best_test_hit=float(best.mean_hit),best_test_seed=int(best.seed),best_epoch=int(best.best_epoch),
                 best_binary_accuracy=float(best.binary_accuracy),n_dates=int(best.n_dates),
                 best_selection='post-hoc maximum over five test seeds; tau/checkpoints were validation-selected'))
    aggregate=pd.DataFrame(summaries);aggregate.to_csv(OUT/'summary.csv',index=False)
    return table,aggregate


def paired_bootstrap(data):
    rows=[];cache={}
    for backbone in CFG['backbones']:
        for protocol in data.validation:
            hits={}
            for mode in ['topk','dense_signed','soft_threshold']:
                values=[]
                for seed in CFG['seeds']:
                    f=pd.read_parquet(OUT/'selected_runs'/backbone/protocol/f'{mode}_s{seed}'/'test_all.parquet')
                    values.append((np.sign(f.prediction.values)==np.sign(f.target.values)).reshape(-1,10).mean(1))
                hits[mode]=np.stack(values)
            for left,right in [('dense_signed','topk'),('soft_threshold','topk'),('soft_threshold','dense_signed')]:
                seed_diff=(hits[left]-hits[right]).mean(1);diff=(hits[left]-hits[right]).mean(0)
                for block in CFG['blocks']:
                    if block not in cache:
                        rng=np.random.default_rng(20261031+block);n=len(diff)
                        starts=rng.integers(0,n,(2000,math.ceil(n/block)))
                        cache[block]=((starts[:,:,None]+np.arange(block))%n).reshape(2000,-1)[:,:n]
                    boot=diff[cache[block]].mean(1);lo,hi=np.quantile(boot,[.025,.975])
                    rows.append(dict(backbone=backbone,protocol=protocol,comparison=left+'-'+right,block=block,
                        mean_effect_pp=100*diff.mean(),ci_low_pp=100*lo,ci_high_pp=100*hi,
                        seed_effect_std_pp=100*seed_diff.std(ddof=1),
                        warning='conditional time uncertainty; maximum-seed and threshold selection not corrected; overlap50 has intentional validation/test overlap'))
    pd.DataFrame(rows).to_csv(OUT/'paired_bootstrap.csv',index=False)


def make_report(data,summary):
    validation=pd.read_csv(OUT/'all_validation_results.csv')
    tau=validation[validation.selector=='soft_threshold'].groupby(['backbone','protocol','tau']).agg(
          validation_hit=('mean_hit','mean'),seed_std=('mean_hit','std'),objective=('validation_objective','mean')).reset_index()
    tau.to_csv(OUT/'tau_validation_grid.csv',index=False)
    full=summary[summary.subset=='test_all'].copy()
    ci=pd.read_csv(OUT/'paired_bootstrap.csv');counts=len(list((OUT/'trajectories').glob('*/*/result.json')))
    def tab(df):return df.to_markdown(index=False,floatfmt='.5f')
    lines=['# Top-k / Dense signed / Validation-selected Signed soft-threshold 비교','',
      '두 본체 각각에서 graph 선택 연산만 변경했다. 기간·통화는 성능을 보고 고르지 않았다. 전체 10개 USD 대비 대상, seeds 0–4, 동일한 test 1,049일을 사용했다.','',
      '## 비교 조건','',
      '- Compact: 최근 temporal MLP hidden32, QK score, direct + graph residual strength, 전체 raw-label BCE.',
      '- Original: 기존 채널별 GRU hidden48, static + QK + macro low-rank score, 원래 strength/gate/message 경로, 기존 Q40 masked directional core + component/graph penalties.',
      '- 두 본체 모두 이번 실행에서는 공통 max20 epochs/patience4/batch256/AdamW lr0.0003/weight decay0.0001/gradient clip1, validation Mean Hit 최대/동률 validation 학습목적 최소를 적용했다. 원래 pit_main의 80 epochs/patience10/batch128/mse_hit checkpoint/LR scheduler를 그대로 재현한 실험은 아니다.',
      '- 입력은 공급된 PIT 패널의 decision t까지 10개 row로 통일했다. 원래 sequence 코드의 추가 t−1 지연은 이번 통제 비교에서 사용하지 않았다. 당일 FX의 실시간 공개/시장 피드 가용성 가정은 기존 논의와 같다. 추가 14일 lag는 이번 실험에 도입하지 않았다.',
      '- Top-k=3, dense와 Top-k는 softmax(abs score) × tanh(score). Soft-threshold는 sign(score) relu(abs(score)−tau), diagonal=0, 행별 abs-sum을 max(1,sum)으로 제한.',
      '- Original의 static reference adjacency/graph regularizer 계수/edge-dropout0.05는 세 selector에서 동일하게 유지했다. Static reference의 기존 Top-k 처리도 공통으로 고정했다. Compact는 기존처럼 edge dropout과 추가 graph penalty가 없다.',
      '- tau grid=[0,0.025,0.05,0.1,0.2]. 각 본체·분할에서 seeds 0–4의 validation 평균 Hit로 공통 tau 하나를 선택한다. 동률이면 평균 validation 목적이 작고, 그다음 tau가 작은 후보를 선택한다. Test는 tau/checkpoint 선택에 사용하지 않았다.','',
      '## 기간','']
    date_rows=[dict(role='train (both)',**data.dates(data.train)),dict(role='validation: pit_original',**data.dates(data.validation['pit_original'])),
               dict(role='validation: overlap50',**data.dates(data.validation['overlap50'])),dict(role='test (both)',**data.dates(data.test)),
               dict(role='test early half / overlapping under overlap50',**data.dates(data.overlap)),dict(role='test late half / not overlapping',**data.dates(data.future))]
    lines += [tab(pd.DataFrame(date_rows)),'',
      f"Overlap50의 공통 날짜는 {len(data.overlap)}일: validation의 {100*len(data.overlap)/len(data.validation['overlap50']):.2f}%, test의 {100*len(data.overlap)/len(data.test):.2f}%다. Validation을 뒤로 이동했기 때문에 train과 shifted validation 사이 524개 decision row는 학습 label이나 validation label로 사용하지 않는다. Lookback의 과거 입력으로는 사용 가능하다.",'',
      'Overlap은 optimizer가 test label로 직접 학습한다는 뜻이 아니다. 그러나 겹친 label로 checkpoint와 tau를 고르므로 test 전체를 독립 일반화 성능으로 해석할 수 없다. 뒤쪽 525일의 평가도 함께 보고한다. 이미 반복 확인한 역사적 기간이라는 제한도 남는다.','',
      '## 요청한 모델별 최고 성능: validation-selected 설정에서 test 최고 seed','',
      tab(full[['backbone','protocol','selector','tau','best_test_hit','best_test_seed','best_epoch','mean_hit','seed_std']]),'',
      'best_test_hit는 선택된 설정의 공통 다섯 seed 중 test Hit 최대값이다. 이 seed는 test를 보고 선택한 사후 최고값이며, 대표 성능은 mean_hit와 seed_std로 함께 확인한다. Tau를 test 최고값으로 다시 바꾸거나 epoch를 test로 선택하지 않았다.','',
      '## Test의 공통 앞·뒤 구간 결과','',
      tab(summary[summary.subset!='test_all'][['backbone','protocol','selector','tau','subset','mean_hit','seed_std','best_test_hit','best_test_seed']]),'',
      '앞 524일과 뒤 525일은 원래 분할에서도 동일한 날짜로 평가했다. 따라서 두 protocol의 부분구간 점수를 날짜가 다른 점수와 혼동하지 않고 비교할 수 있다.','',
      '## Validation tau 탐색 전체 결과','',tab(tau),'',
      '## 동일 seed·동일 날짜의 평균 효과와 조건부 시간 불확실성','',
      tab(ci[ci.block==20][['backbone','protocol','comparison','mean_effect_pp','ci_low_pp','ci_high_pp','seed_effect_std_pp']]),'',
      'Paired circular block bootstrap은 같은 날짜의 통화를 함께 평균하고 공통 seed 평균 차이에 적용했다. Block=20, 2,000회이며 5/60 민감도를 CSV에 저장했다. 이 CI는 모델/seed 집합 조건부 시간 불확실성으로, 최고 seed 선택·tau 선택·의도적 overlap을 보정하지 않는다.','',
      '## 실행·검증·산출물','',
      f"물리적 optimizer trajectories는 {counts}개다(본체 2 × 후보 7 × seed 5). 각 trajectory에서 두 validation protocol의 checkpoint/early-stop 상태를 독립적으로 추적했다. 고정 learning rate이며 validation이 gradient나 optimizer update를 바꾸지 않으므로, 같은 epoch의 학습 가중치는 두 protocol에서 같다. 한 protocol이 멈추면 해당 선택 상태를 고정하고 다른 protocol만 계속 평가한다. 논리적 validation 조건은 140개이며, tau 선택 후 test 조건은 60개다.",'',
      '각 trajectory에 history.csv/resume.pt와 protocol별 best.pt/validation predictions를 저장했다. 선택된 60개 test 결과에는 전체·앞·뒤 구간의 날짜/통화별 예측을 저장했다. 초기화 해시와 source/data manifest를 함께 저장했다.','',
      '```powershell','python src/arcfx_selector_compare.py','python src/arcfx_selector_compare.py --analysis-only','python src/check_arcfx_selector_compare.py','```','']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    print(full.to_string(index=False),flush=True)


def save_manifest(data):
    paths=['src/arcfx_selector_models.py','src/arcfx_selector_compare.py','src/models/arc_fx.py',
           'src/training_core.py','src/data_pipeline.py','src/arcfx_search_models.py',
           'data/processed/fx_panel_pit.csv','data/processed/nonfx_features_pit.csv','configs/pit_main.yaml']
    manifest=dict(config=CFG,source_data_hashes={p:sha(ROOT/p) for p in paths},
         periods=dict(train=data.dates(data.train),test=data.dates(data.test),
                      validation={p:data.dates(ix) for p,ix in data.validation.items()}),
         validation_test_overlap=dict(pit_original=0,overlap50=len(data.overlap)),
         model_changes='selection operator only within each backbone',
         original_loss_hyperparameters={k:v for k,v in vars(data.loss_args).items() if k.startswith('lambda_')},
         torch=torch.__version__,numpy=np.__version__)
    path=OUT/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest,'Code/data/config changed. Use a new output directory.'
    else:dump(path,manifest)
    dump(ROOT/'configs/arcfx_selector_compare.json',CFG)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--analysis-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(CFG['threads']);data=Data();save_manifest(data)
    if not args.analysis_only:
        for backbone in CFG['backbones']:
            for seed in CFG['seeds']:
                for mode,tau in candidates():run_trajectory(data,backbone,mode,tau,seed)
    selection=select_models(data);_,summary=test_selected(data,selection);paired_bootstrap(data);make_report(data,summary)
    records=[json.loads(p.read_text()) for p in (OUT/'trajectories').glob('*/*/result.json')]
    ledger=[json.loads(line) for line in (OUT/'ledger.jsonl').read_text().splitlines()]
    dump(OUT/'completion.json',dict(optimizer_trajectories=len(records),logical_validation_conditions=len(records)*2,
        selected_test_runs=len(selection)*5,failed_attempts=sum(r.get('status')=='failed' for r in ledger),
        total_optimizer_epochs=sum(r['trained_epochs'] for r in records),total_train_seconds=sum(r['seconds'] for r in records)))


if __name__=='__main__':main()
