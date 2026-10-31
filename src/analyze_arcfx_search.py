"""Post-training diagnostics and honest complete/post-hoc report generation."""
import hashlib, json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_arcfx_search as s
from arcfx_search_models import SearchModel
from data_pipeline import set_seed, prepare_data


def graph_sample(model,fold,spec):
    ix=fold.val[:128];model.eval()
    with torch.no_grad():
        z,_=model(fold.batch(ix,spec.get('legacy_alignment',False)))
        o=model.last; a=o['adj'].numpy();g=o['graph_gate'].numpy()
        m=o['message'].numpy()
        offdiag=~np.eye(11,dtype=bool)
        result=dict(edge_active_offdiagonal=float((a[:,offdiag]!=0).mean()),
                    message_rms=float(np.sqrt(np.mean(m*m))),gate_q01=float(np.quantile(g,.01)),
                    gate_q50=float(np.quantile(g,.5)),gate_q99=float(np.quantile(g,.99)),
                    gate_saturated=float(((g<.01)|(g>.99)).mean()),strength_diff_std=float(z[:,1:].std()),
                    adjacency_temporal_std=float(a.std(0).mean()))
        for name,gate in o.get('component_gates',{}).items():
            gg=gate.numpy()
            result['component_gate_'+name+'_mean']=float(gg.mean())
            result['component_gate_'+name+'_saturated']=float(((gg<.01)|(gg>.99)).mean())
        original=z.detach().clone()
        if model.original:
            b=fold.batch(ix,spec.get('legacy_alignment',False));b['disable_relational_path']=True
            disabled,_=model(b)
            result['whole_relational_path_abs_logit_change']=float((original-disabled).abs().mean())
    # Measure named gradient paths and all weighted terms on one explicit, fixed past minibatch.
    model.eval();ix=fold.train[:256];b=fold.batch(ix,spec.get('legacy_alignment',False))
    if spec['loss']=='legacy':
        model(fold.batch(fold.train[256:512],spec.get('legacy_alignment',False)))
        b['A_prev']=model.last['adj'].detach()
    z,r=model(b);loss,terms=s.loss_fn(model,z,r,ix,fold,spec)
    model.zero_grad();loss.backward()
    result.update(s.gradients(model));result.update({'loss_'+k:v for k,v in terms.items()})
    return result


def main():
    torch.set_num_threads(2)
    s.N=len(pd.read_csv(s.ROOT/'data/processed/fx_panel_pit.csv'))
    final_start=10+int((s.N-10)*.6)+int((s.N-10)*.2)
    folds=dict(dev1=s.Fold('dev1',int(.5*s.N),int(.6*s.N)),
               dev2=s.Fold('dev2',int(.6*s.N),int(.7*s.N)),
               final=s.Fold('final',int(.7*s.N),final_start,s.N))
    diagnostics=[]
    for path in sorted((s.OUT/'runs').glob('*/result.json')):
        result=json.loads(path.read_text());spec=result['spec'];fold=folds[result['fold']]
        set_seed(result['seed']);model=SearchModel(fold.p.dims,s.CCY,spec)
        diagnostics.append(dict(run=path.parent.name,phase='initial',**graph_sample(model,fold,spec)))
        model.load_state_dict(torch.load(path.parent/'best.pt',weights_only=True))
        diagnostics.append(dict(run=path.parent.name,phase='selected',**graph_sample(model,fold,spec)))
        # Recompute message statistics with actual aggregated messages (the screening logger's
        # fallback was the relational score for the original model, not the message tensor).
        for split,ix in [('train',fold.train),('validation',fold.val)]+([('report',fold.report)] if 'report' in result else []):
            pack=s.predict(model,fold,ix,spec);metrics=pack[0]
            saved=pd.read_parquet(path.parent/(split+'.parquet'))
            np.testing.assert_allclose(pack[2].ravel(),saved.pred.values,rtol=1e-6,atol=1e-8)
            result[split]['message_rms']=metrics['message_rms']
        s.dump(path,result)
    df=pd.DataFrame(diagnostics);df.to_csv(s.OUT/'graph_diagnostics.csv',index=False)
    selected=json.loads((s.OUT/'selection_frozen.json').read_text())['winner']
    s.summarize(selected)
    # Data/source fingerprints and exact provenance for reproductions.
    files=[s.ROOT/'data/processed/fx_panel_pit.csv',s.ROOT/'data/processed/nonfx_features_pit.csv']
    files+=list((s.ROOT/'src').glob('*arcfx_search*.py'))+[s.ROOT/'src/arcfx_search_models.py']
    s.dump(s.OUT/'manifest.json',{str(p.relative_to(s.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    s.dump(s.OUT/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,
           numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),threads=2,
           device='cpu',historical_results_preserved=True))
    p=prepare_data(str(files[0]),str(files[1]),s.CCY,10,[.6,.2,.2],True)
    q=float(np.quantile(abs(p.y_norm[:int(.6*s.N),1:]),.4));active=abs(p.y_norm[:,1:])>=q
    s.dump(s.OUT/'active_label_audit.json',dict(q40_normalized=q,
           binary_label_disagreement=float(((p.y_raw[:,1:]>0)!=(p.y_norm[:,1:]>0)).mean()),
           three_way_sign_disagreement=float((np.sign(p.y_raw[:,1:])!=np.sign(p.y_norm[:,1:])).mean()),
           active_three_way_sign_disagreement=float((np.sign(p.y_raw[:,1:])!=np.sign(p.y_norm[:,1:]))[active].mean()),
           interpretation='Centering changes tiny-move labels, but zero active-label discrepancies: not established as the cause of historical near-chance results.'))
    favorable_relative(selected)
    make_report(selected,df)
    make_plots(selected)


def favorable_relative(winner):
    rows=[]
    comparisons={name:pd.read_parquet(s.OUT/(name+'_report.parquet')) for name in ['majority','logistic']}
    for name in ['mlp','transformer','gnn','corrlstmgat','fxrp']:
        ff=[]
        for p in (s.ROOT/'results/pit_main/runs/baselines').glob(name+'_seed*/predictions/*.parquet'):
            f=pd.read_parquet(p);f=f[f.currency!='USD'].rename(columns={'target_date':'decision_date'})
            f['decision_date']=pd.to_datetime(f.decision_date).dt.strftime('%Y-%m-%d');ff.append(f)
        if ff:comparisons['archived_'+name]=pd.concat(ff)
    for p in (s.OUT/'runs').glob(winner['name']+'__final__s*/report.parquet'):
        w=pd.read_parquet(p);w['hit']=(np.sign(w.pred)==np.sign(w.target)).astype(float)
        w['year']=pd.to_datetime(w.target_date).dt.year
        for name,f in comparisons.items():
            f=f.copy();f['baseline_hit']=(np.sign(f.pred)==np.sign(f.target)).astype(float)
            bh=f.groupby(['decision_date','currency']).baseline_hit.mean().reset_index()
            matched=w.merge(bh,on=['decision_date','currency'],validate='one_to_one')
            assert len(matched)==len(w)
            for year,g in matched.groupby('year'):
                rows.append(dict(seed=int(g.seed.iloc[0]),year=int(year),comparator=name,n=len(g),
                                 date_start=g.target_date.min(),date_end=g.target_date.max(),
                                 our_hit=g['hit'].mean(),baseline_hit=g.baseline_hit.mean(),
                                 difference=(g['hit']-g.baseline_hit).mean(),post_hoc=True))
    pd.DataFrame(rows).sort_values('difference',ascending=False).to_csv(s.OUT/'favorable_relative_POSTHOC.csv',index=False)


def make_plots(winner):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=pd.read_csv(s.OUT/'aggregate.csv')
    dev=a[(a.split=='validation')&(a.fold!='final')].copy()
    # Different seed counts are labelled; all candidates remain visible.
    pivot=dev.pivot(index='name',columns='fold',values='mean')
    fig,ax=plt.subplots(figsize=(10,6))
    for fold,marker in [('dev1','o'),('dev2','s')]:
        ax.scatter(pivot[fold],np.arange(len(pivot)),marker=marker,label=fold)
    counts=dev.groupby('name')['count'].first()
    ax.set_yticks(np.arange(len(pivot)),[f'{name} (n={int(counts[name])})' for name in pivot.index]);ax.axvline(.5,color='gray',linestyle='--')
    ax.axvline(.52,color='gray',linestyle=':');ax.set_xlabel('Development Mean Hit (selection data)');ax.legend()
    fig.tight_layout();fig.savefig(s.OUT/'development_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for seed in range(5):
        path=s.OUT/'runs'/f"{winner['name']}__final__s{seed}"/'history.csv'
        if not path.exists():continue
        f=pd.read_csv(path)
        axes[0].plot(f.epoch,f.train_hit,label=f'seed {seed}')
        axes[1].plot(f.epoch,f.val_hit,label=f'seed {seed}')
    axes[0].set_title('Final-fit training');axes[1].set_title('Checkpoint validation')
    for ax in axes:
        ax.set_xlabel('Epoch');ax.set_ylabel('Mean Hit');ax.legend(fontsize=8);ax.axhline(.5,color='gray',linestyle='--')
    fig.tight_layout();fig.savefig(s.OUT/'training_curves.png',dpi=160);plt.close(fig)


def make_report(winner,diag):
    a=pd.read_csv(s.OUT/'aggregate.csv');allr=pd.read_csv(s.OUT/'all_results.csv')
    y=pd.read_csv(s.OUT/'all_seed_year.csv');ci=pd.read_csv(s.OUT/'paired_bootstrap.csv')
    w=a[(a.name==winner['name'])&(a.fold=='final')&(a.split=='report')].iloc[0]
    wf=allr[(allr.name==winner['name'])&(allr.fold=='final')&(allr.split=='report')]
    wy=y[(y.name==winner['name'])&(y.fold=='final')&(y.split=='report')]
    favorable=wy.sort_values('hit',ascending=False).iloc[0]
    overall_favorable=y[y.split=='report'].sort_values('hit',ascending=False).iloc[0]
    bestseed=wf.sort_values('mean_hit',ascending=False).iloc[0]
    count=json.loads((s.OUT/'completion.json').read_text())
    fm=json.loads(next((s.OUT/'runs').glob(winner['name']+'__final__s0/result.json')).read_text())
    report_predictions=pd.read_parquet(s.OUT/'runs'/f"{winner['name']}__final__s0"/'report.parquet')
    dev=a[(a.split=='validation')&(a.fold!='final')]
    report=a[(a.split=='report')]
    def tab(frame):return frame.to_markdown(index=False,floatfmt='.5f')
    rows=['# ARC_FX 탐색 결과', '',
          '결과를 꾸미거나 유리한 seed·기간으로 전체 성능을 대체하지 않았다. 아래 유리한 사례는 사후 탐색 결과로 별도 표시한다.', '',
          f"Development 선택 모델: **{winner['name']}**. 전체 보고 구간 Mean Hit: **{w['mean']:.5f}**, seed 표준편차 **{w['std']:.5f}**, {int(w['count'])} seeds.",
          f"전체 평균 기준 0.52 목표: **{'도달' if w['mean']>=.52 else '미달'}**. 범위 {w['min']:.5f}–{w['max']:.5f}.", '',
          f"평가 return 종료일: {report_predictions.target_date.min()}–{report_predictions.target_date.max()}, 1,049일 × 10개 통화. 대응 decision date는 {report_predictions.decision_date.min()}–{report_predictions.decision_date.max()}다.", '',
          '## 1. 진단과 해석', '',
          '- 기존 PIT 두 설정의 checkpoint와 예측은 다르다. 부호 불일치는 seed별 0.87–1.85%; 같은 0.4997은 반올림된 집계 결과다.',
          '- 평균을 뺀 target과 원수익률의 이진 label 불일치는 약 0.80%, 3-way sign 불일치는 약 1.06%다. 그러나 Q40 이상 기존 active 표본에서는 0%다. 따라서 label 문제만으로 기존 성능을 설명할 수 없다.',
          '- 기존 입력은 t−1에서 끝나지만 label은 t→t+1이다. 신규 입력은 t까지 사용하고 실제 return 종료일을 별도 저장했다. 기존 방식보다 정보시점이 하루 신선해졌다는 점을 비교 해석에 포함해야 한다.',
          '- 기존 smoothness는 셔플된 배치 간 비교이며 실제 시간 인접성 규제가 아니다. 기존 relational branch는 메시지가 없어도 자기 상태에서 출력이 생긴다.',
          '- 16일 subset 과적합은 sign-hit 0.99375로 통과했다. 0 target 처리 때문에 binary-perfect와 sign-hit-perfect는 다르다. 전체 모델의 일반화가 입증된 것은 아니다.',
          '- 공급된 availability audit의 시각 위반은 0건이다. 이는 코드의 가용시각 가정 아래 결과이며, 실제 역사적 공표시각·H.10 실시간 가용성을 완전히 보증하지 않는다.', '',
          '## 2. 전체 결과: 유리한 seed를 선택하지 않음', '',tab(report), '',
          '## 3. Development 결과', '',tab(dev), '',
          '각 fold는 과거 데이터만으로 scaler와 threshold를 계산했다. 후보 선택에 사용된 구간이므로 이 표의 최고값은 선택 편향을 포함한다. 최종 보고 기간은 이미 본 기간이며 untouched test가 아니다. 선택은 seeds 0,1,2 기준이며 dense_signed의 seeds 3,4는 선택을 고정한 뒤 추가 검증했다. 따라서 위 표의 5-seed 평균은 원래 선택 순위와 다를 수 있다. 1-seed 행의 std=nan은 추정 불가를 의미한다.', '',
          '## 4. 선택 모델의 seed·연도 안정성', '',
          tab(wf[['seed','mean_hit','bce','pred_positive','actual_positive','logit_std','zero_graph_hit']]), '',
          tab(wy.groupby('year').agg(hit_mean=('hit','mean'),hit_std=('hit','std'),samples_per_seed=('n','first')).reset_index()), '',
          '## 5. 사용자 요청: 유리한 seed·기간 (POST HOC)', '',
          f"선택 모델의 전체 보고 기간에서 가장 유리한 seed는 **{int(bestseed.seed)}**, Mean Hit **{bestseed.mean_hit:.5f}**다.",
          f"선택 모델의 seed×달력연도 조합 최고는 **seed {int(favorable.seed)}, {int(favorable.year)}년**, Hit **{favorable.hit:.5f}**, {int(favorable.n)} 통화-날짜 표본이다. 정확한 return 종료일 범위는 {favorable.date_start}–{favorable.date_end}다.",
          f"전체 보고 모델을 통틀어 가장 높은 연간 단일 seed 값은 {overall_favorable['name']}, seed {int(overall_favorable.seed)}, {int(overall_favorable.year)}년의 {overall_favorable.hit:.5f}다. 이 값을 선택 모델의 성능으로 섞지 않는다.",
          '이 값은 다중 비교 후 최대값이며 대표 성능이나 재현성의 근거가 아니다. 전체 후보의 모든 보고 seed×연도를 좋은 순서부터 나쁜 순서까지 `favorable_seed_period_POSTHOC.csv`에 저장했다. 임의 이동창이나 유리한 시작·종료일은 추가 탐색하지 않았다.', '',
          '같은 날짜 baseline 대비 유리한 차이와 불리한 차이는 `favorable_relative_POSTHOC.csv`에 모두 저장했다. 이것도 seed·연도·baseline을 반복 비교한 사후 통계로, 독립 검증이 아니다.', '',
          '## 6. 그래프 기여', '',
          f"선택 모델의 graph를 추론 시 제거한 Mean Hit 평균은 {wf.zero_graph_hit.mean():.5f}; 원래 출력 대비 차이는 {(wf.mean_hit-wf.zero_graph_hit).mean():+.5f}다.",
          '이는 고정된 모델의 의존성 검사다. 재학습 no-graph 결과와 paired bootstrap을 함께 봐야 실제 성능 이득을 주장할 수 있다. 원래 backbone에서는 adjacency=0이 전체 relational branch 제거와 다르다.',
          '선택된 dense graph의 off-diagonal edge는 모두 활성이고 gate는 포화되지 않았으며 상태에 따라 adjacency가 변한다. Graph gradient도 0이 아니다. 그러나 전체 예측 개선에 대한 독립적 기여는 확인되지 않았다.',
          '`graph_diagnostics.csv`는 초기/선택 checkpoint의 실제 메시지 RMS, gate 분위수·포화, off-diagonal 활성률, 상태별 adjacency 변화, encoder/graph/head gradient와 손실 항을 담는다. 초기 비교는 동일한 validation 128일, gradient는 고정된 과거 256일을 사용한다. Epoch 로그의 gradient/regularization은 마지막 minibatch 진단값이다.', '',
          '## 7. Baseline 차이와 불확실성', '',tab(ci[ci.block==20][['comparator','n_dates','baseline_mean_hit','difference','ci_low','ci_high']]), '',
          '기본 paired circular block bootstrap은 20일 블록, 2,000회다. 5/60일 민감도는 CSV에 있다. 같은 날짜 통화들을 함께 뽑고 seed별 correctness를 먼저 평균했다. 따라서 CI는 고정된 seed 집합 조건부 시간 불확실성이고 seed 표준편차와 별개다. 선택 효과·기존 test 반복 열람을 보정하지 않았다. Archived baseline은 학습 종료일과 입력 정렬이 달라 인과적 공정 비교가 아니다.', '',
          '## 8. 추천 구조와 손실', '',
          '```json',json.dumps(winner,ensure_ascii=False,indent=2),'```', '',
          'Direction BCE: L_dir = mean[BCEWithLogits(u_i−u_USD, 1[r_i>0])], T=1. Dual 후보는 L=L_dir+λ mean[Huber(v_i−v_USD,r_i/σ_train;δ=1)], λ∈{0.03,0.1}. 수익률 예측은 σ_train(v_i−v_USD), σ_train은 모든 통화의 학습수익률 pooled std다. Direction logit은 수익률 크기가 아니다. 기존 loss가 선택되었다면 JSON의 legacy/active를 따른다(프로토콜 참조).',
          '추천은 이번 제한된 development 탐색의 결과다. 유의한 전체 우위가 없으면 기존 모델을 대체하는 확정 개선으로 추천할 수 없다. 구조/손실 변경, 입력 정렬, 과거 데이터량 차이를 분리한 인과 추론은 이번 실험으로 불가능하다.', '',
          '선택 모델이 BCE 단독이면 방향 연구 후보이며 수익률 크기를 출력하는 완성된 점예측 모델을 대체하지 않는다. 해당 후보의 logit에 RMSE를 붙이지 않았다. 수익률 점예측이 필수인 경우 dual-head 후보의 별도 RMSE와 방향 지표를 사용해야 하며, 좋은 방향 지표와 다른 모델의 좋은 RMSE를 결합해서는 안 된다.', '',
          '## 9. 디펜스에서 주장할 수 있는 범위', '',
          '- PIT 패널과 명시된 가용시각 가정 아래, currency strength 차분과 상태 의존 그래프를 유지한 제한적 모델·손실 탐색을 수행했다.',
          '- 전체 날짜·통화 평가, 모든 후보·seed 로그, 재현 가능한 출력과 사후 유리한 결과를 분리해 제시할 수 있다.',
          '- 전체 0.52 도달 여부와 baseline 대비 신뢰구간은 위 표의 실제 수치에 한해서 주장한다. 일부 연도/seed 최고값을 일반적인 예측 우위로 제시할 수 없다.',
          '- 반복해서 본 보고 기간, 작은 후보/seed 수, 짧은 epoch 예산, vintage/시장시각 가정, 거래비용·실행 가능성 미평가를 한계로 명시해야 한다.', '',
          '## 10. 실행과 산출물', '',
          f"완료 neural fits {count['completed_training_runs']}, 시도 {count['training_attempts']}, 실패 {count['failed_attempts']}. Logistic baseline은 하나의 pipeline 안에서 통화별 추정기 10개를 fit했다. Majority 추정과 250-step smoke는 별도다.",
          f"최종 fit 학습: {fm['train_start']}–{fm['train_end']}; checkpoint 검증: {fm['validation_start']}–{fm['validation_end']}.", '',
          '```powershell','python src/check_arcfx_search.py','python -u src/run_arcfx_search.py','python src/analyze_arcfx_search.py','```', '',
          '전체 설정/손실/구조는 각 run의 result.json, 학습 추이는 history.csv, weights는 best.pt, 재개 상태는 resume.pt, 날짜·통화별 예측은 parquet다. 집계는 all_results.csv, aggregate.csv, all_seed_year.csv, all_currency.csv, paired_bootstrap.csv에 저장된다. SHA256 manifest와 원래 예측 비교표도 같은 결과 폴더에 있다.', '',
          '상세 정의와 제한은 `docs/arcfx_search_protocol.md`를 참조한다.']
    (s.OUT/'FINAL_REPORT_KO.md').write_text('\n'.join(rows),encoding='utf-8')


if __name__=='__main__':main()
