"""Readable tables/plots and paired effects of deliberate validation/test overlap."""
import hashlib,json,math
import numpy as np
import pandas as pd
import arcfx_selector_compare as r


def bootstrap(values,block):
    n=len(values);rng=np.random.default_rng(20261101+block)
    starts=rng.integers(0,n,(2000,math.ceil(n/block)))
    index=((starts[:,:,None]+np.arange(block))%n).reshape(2000,-1)[:,:n]
    samples=values[index].mean(1)
    return np.quantile(samples,[.025,.975])


def merge_worker_ledgers():
    marker=r.OUT/'worker_ledgers_merged.json'
    if marker.exists():return
    files=sorted(r.OUT.glob('worker_*_ledger.jsonl'));rows=[]
    for path in files:
        rows += [json.loads(line) for line in path.read_text().splitlines() if line]
    if not files:return
    assert all((r.OUT/f'worker_{i}_complete.json').exists() for i in range(3))
    with (r.OUT/'ledger.jsonl').open('a',encoding='utf-8') as stream:
        for row in sorted(rows,key=lambda x:x['time']):stream.write(json.dumps(row)+'\n')
    r.dump(marker,dict(merged=True,rows=len(rows),hashes={p.name:r.sha(p) for p in files}))


def overlap_effects():
    rows=[]
    for backbone in r.CFG['backbones']:
        for mode in ['topk','dense_signed','soft_threshold']:
            for subset in ['test_all','test_common_early_half','test_common_late_half']:
                hits={}
                for protocol in ['pit_original','overlap50']:
                    vals=[]
                    for seed in r.CFG['seeds']:
                        path=r.OUT/'selected_runs'/backbone/protocol/f'{mode}_s{seed}'/(subset+'.parquet')
                        f=pd.read_parquet(path)
                        vals.append((np.sign(f.prediction.values)==np.sign(f.target.values)).reshape(-1,10).mean(1))
                    hits[protocol]=np.stack(vals)
                diff=hits['overlap50']-hits['pit_original']
                for block in [5,20,60]:
                    lo,hi=bootstrap(diff.mean(0),block)
                    rows.append(dict(backbone=backbone,selector=mode,subset=subset,n_dates=diff.shape[1],block=block,
                        original_hit=float(hits['pit_original'].mean()),overlap_hit=float(hits['overlap50'].mean()),
                        difference_pp=float(diff.mean()*100),ci_low_pp=float(lo*100),ci_high_pp=float(hi*100),
                        seed_effect_std_pp=float(diff.mean(1).std(ddof=1)*100),
                        meaning='overlap-selected model minus original-validation-selected model on identical dates/seeds; no correction for selection'))
    table=pd.DataFrame(rows);table.to_csv(r.OUT/'validation_overlap_effects.csv',index=False)
    return table


def plots(full):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True)
    low=100*(full.mean_hit-full.seed_std).min()-.4
    high=100*np.maximum(full.best_test_hit,full.mean_hit+full.seed_std).max()+.4
    for i,backbone in enumerate(r.CFG['backbones']):
        for j,protocol in enumerate(['pit_original','overlap50']):
            ax=axes[i,j];g=full[(full.backbone==backbone)&(full.protocol==protocol)].set_index('selector').loc[['topk','dense_signed','soft_threshold']]
            xx=np.arange(3)
            ax.errorbar(xx-.05,g.mean_hit*100,yerr=g.seed_std*100,fmt='o',capsize=4,label='Five-seed mean +/- SD')
            ax.scatter(xx+.05,g.best_test_hit*100,marker='D',color='tab:orange',label='Highest test seed (post hoc)')
            for k,(_,row) in enumerate(g.iterrows()):
                ax.annotate(f's{int(row.best_test_seed)}: {row.best_test_hit*100:.2f}',(k+.05,row.best_test_hit*100),
                            xytext=((-5,5) if k==2 else (5,5)),ha=('right' if k==2 else 'left'),textcoords='offset points',fontsize=8)
            tau=g.loc['soft_threshold','tau']
            ax.set_xticks(xx,['Top-k=3','Dense signed',f'Soft threshold\ntau={tau:g}'])
            ax.set_title(backbone+' / '+protocol);ax.set_ylim(low,high)
            ax.axhline(50,color='gray',linestyle=':',linewidth=1);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8);fig.supylabel('Mean Hit (%) — all ten USD targets')
    fig.tight_layout();fig.savefig(r.OUT/'selector_comparison.png',dpi=170);fig.savefig(r.OUT/'selector_comparison.pdf');plt.close(fig)


def summary_document(full,effects):
    def tab(frame):return frame.to_markdown(index=False,floatfmt='.3f')
    readable=full.copy()
    validation=pd.read_csv(r.OUT/'all_validation_results.csv');test=pd.read_csv(r.OUT/'test_results.csv')
    val_seeds=[];val_test=[]
    for _,row in readable.iterrows():
        group=validation[(validation.backbone==row.backbone)&(validation.protocol==row.protocol)&(validation.selector==row.selector)]
        if row.selector=='soft_threshold':group=group[np.isclose(group.tau,row.tau)]
        chosen=group.sort_values(['mean_hit','validation_objective','seed'],ascending=[False,True,True]).iloc[0]
        val_seeds.append(int(chosen.seed))
        match=test[(test.backbone==row.backbone)&(test.protocol==row.protocol)&(test.selector==row.selector)&(test.seed==chosen.seed)&(test.subset=='test_all')].iloc[0]
        val_test.append(float(match.mean_hit)*100)
    readable['validation_selected_seed']=val_seeds
    readable['test_hit_validation_selected_seed_pct']=val_test
    for col in ['mean_hit','seed_std','best_test_hit']:readable[col]=100*readable[col]
    readable=readable.rename(columns={'mean_hit':'mean_hit_pct','seed_std':'seed_sd_pp','best_test_hit':'best_hit_pct'})
    readable.to_csv(r.OUT/'comparison_readable.csv',index=False)
    manifest=json.loads((r.OUT/'manifest.json').read_text())
    lines=['# Graph 선택 연산 비교 요약','',
      '두 본체 × 세 선택 연산 × 두 validation protocol을 비교했다. Tau와 epoch는 validation으로 선택했다. 요청한 최고 성능은 선택된 설정에서 test Hit가 가장 높은 seed이며 사후 최고값으로 표시한다. 전체 10개 USD 대비 대상과 공통 test 1,049일을 사용한다.','',
      '## 공통 test에서 모델별 최고값과 평균','',
      '아래 best_hit_pct는 %, seed_sd_pp는 %p다.','',
      tab(readable[['backbone','protocol','selector','tau','best_hit_pct','best_test_seed','best_epoch','mean_hit_pct','seed_sd_pp']]),'',
      '원래 분할에서는 두 본체 모두 Top-k의 최고값과 5-seed 평균이 가장 높았다. 겹침 분할의 MLP에서는 dense signed가 최고값/평균 모두 가장 높았고, 원래 ARC_FX에서는 Top-k가 가장 높았다. 이번 범위에서 validation-selected soft-threshold가 Top-k를 개선한다는 우위는 확인되지 않았다.','',
      'Seed도 test를 보지 않고 validation으로 고른 경우의 test 결과:','',
      tab(readable[['backbone','protocol','selector','tau','validation_selected_seed','test_hit_validation_selected_seed_pct']]),'',
      '## Validation과 test의 겹침 효과','',
      'Same-seed 평균 차이이며 overlap-selected − original-validation-selected 방향이다. 두 protocol의 test 날짜는 동일하다.','',
      tab(effects[(effects.block==20)&(effects.subset=='test_all')][['backbone','selector','difference_pp','ci_low_pp','ci_high_pp']]),'',
      '## 선택된 모델의 실제 graph 활성 정도','']
    selected=pd.read_csv(r.OUT/'test_results.csv')
    diag=selected[(selected.subset=='test_all')&(selected.selector=='soft_threshold')].groupby(['backbone','protocol']).agg(
        tau=('tau','first'),mean_neighbors=('mean_neighbors','mean'),zero_neighbor_fraction=('zero_neighbor_fraction','mean'),
        active_edge_fraction=('active_edge_fraction','mean'),message_rms=('message_rms','mean')).reset_index()
    lines += [tab(diag),'']
    for _,row in diag.iterrows():
        if row.tau==0:lines += [f"{row.backbone}/{row.protocol}은 validation에서 tau=0을 선택했다. 이 경우 임계값에 의한 희소화는 없으며 row-mass cap을 가진 signed score 변환이다.",'']
    lines += ['## 기간과 해석','',
      '- Train: 2005-01-18–2017-07-24 (decision date).',
      '- 원래 validation: 2017-07-25–2021-10-05, 1,048일.',
      '- 겹침 validation: 2019-08-30–2023-11-09, 1,048일.',
      '- 공통 test: 2021-10-06–2025-12-17, 1,049일. 수익률 종료일은 2021-10-07–2025-12-18.',
      '- 겹침: 524일, validation 50% / test 49.95%. 뒤쪽 비중복 test는 525일이다.',
      '- 겹친 test label은 optimizer의 fitting label에 포함되지 않지만 checkpoint/tau 선택에 사용된다. 따라서 겹침 점수는 독립적인 test 성능이 아니다.',
      '- 최고 seed를 고른 효과도 반영되지 않으므로 최고값과 5-seed 평균을 함께 보고한다. 날짜 블록 bootstrap은 선택 편향을 교정하지 않는다.',
      '- 두 본체의 구조/손실은 각각 유지했으나 이번 training budget은 공통 max20/patience4/batch256, constant LR다. 기존 PIT의 80-epoch 학습 전체를 재현한 것이 아니라 원래 기간 설정을 사용한 통제 비교다.',
      '- 입력 window는 양쪽 본체 모두 t까지 10개 row로 고정했다. 이는 원래 sequence 구현의 t−1 추가 지연과 다르다. 당일 spot 가용성에 관한 시장 피드 가정은 남아 있다.','',
      '상세한 validation tau grid, 부분기간 성능, 모델 간 paired CI와 재현 설명은 REPORT_KO.md에 있다.','',
      '![Selector comparison](selector_comparison.png)','',
      '```powershell','python src/arcfx_selector_compare.py','python src/check_arcfx_selector_compare.py','python src/arcfx_selector_supplement.py','```','']
    (r.OUT/'SUMMARY_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    r.dump(r.OUT/'supplement_manifest.json',dict(script_sha256=r.sha(__file__),
        worker_script_sha256=r.sha(r.ROOT/'src/arcfx_selector_worker.py'),test_script_sha256=r.sha(r.ROOT/'src/check_arcfx_selector_compare.py'),
        protocol_sha256=r.sha(r.ROOT/'docs/arcfx_selector_compare_protocol.md'),
        core_manifest_sha256=r.sha(r.OUT/'manifest.json')))


def main():
    merge_worker_ledgers();summary=pd.read_csv(r.OUT/'summary.csv');full=summary[summary.subset=='test_all'].copy()
    effects=overlap_effects();plots(full);summary_document(full,effects)


if __name__=='__main__':main()
