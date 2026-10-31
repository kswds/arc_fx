"""Check saved experimental outcomes, timing, calibration, and scope comparability."""
import json
import warnings
import numpy as np
import pandas as pd
import torch
import arcfx_followup as f


def main():
    torch.set_num_threads(2);f.verify_contract()
    completion=json.loads((f.OUT/'completion.json').read_text())
    assert completion['total_model_conditions']==60
    assert completion['new_neural_fits']==52 and completion['reused_fits']==8
    assert completion['failures']==0
    folds={p:f.Fold(p) for p in f.CFG['policies']}
    for fold in folds.values():f.smoke(fold)
    c=folds['conservative_lag14']
    dates=c.p.merged.Date.values.astype('datetime64[D]');valid=c.fx_source>=0
    assert np.all(dates[valid]-dates[c.fx_source[valid]]>=np.timedelta64(14,'D'))
    rows=[]
    for policy,fold in folds.items():
        initial={}
        for seed in f.CFG['seeds']:
            for condition in list(f.CONDITIONS)+f.CFG['matched_baselines']:
                dest=f.OUT/'runs'/policy/f'{condition}_s{seed}'
                result=json.loads((dest/'result.json').read_text());rows.append(result)
                if condition in f.CONDITIONS:
                    if seed in initial:assert initial[seed]==result['initial_state_hash']
                    initial[seed]=result['initial_state_hash']
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',message='enable_nested_tensor is True')
                    model=f.Model(fold.p.dims,f.CONDITIONS[condition]['graph']) if condition in f.CONDITIONS else f.Baseline(fold.p.dims,condition)
                model.load_state_dict(torch.load(dest/'best.pt',weights_only=True));model.eval()
                lag=f.CONDITIONS.get(condition,{}).get('lag',0)
                for split,ix in [('train',fold.train),('validation',fold.val),('report',fold.report)]:
                    frame,z,y=f.read_grid(dest/(split+'.parquet'))
                    assert len(frame)==len(ix)*10
                    assert frame.groupby('decision_date').currency.nunique().eq(10).all()
                    assert not frame.duplicated(['decision_date','currency']).any()
                    np.testing.assert_array_equal(y,fold.p.y_raw[ix,1:])
                    actual=f.metrics(z,y)
                    for key in ['mean_hit','binary_accuracy','bce','brier']:
                        np.testing.assert_allclose(actual[key],result['splits'][split][key],atol=1e-12)
                pos=np.array([0,13,201,len(fold.report)-1]);z,_=f.predict(model,fold,fold.report[pos],lag)
                saved=pd.read_parquet(dest/'report.parquet').logit.values.reshape(-1,10)
                np.testing.assert_allclose(z,saved[pos],rtol=2e-5,atol=2e-6)
                for split in ['validation','report']:
                    orig,z,y=f.read_grid(dest/(split+'.parquet'))
                    _,cal,_=f.read_grid(dest/(split+'_calibrated.parquet'))
                    params=json.loads((dest/'calibration.json').read_text())
                    assert params['scale']>0 and params['usd_offset']==0
                    expected=params['scale']*z.astype(np.float64)+np.array([params['currency_offsets'][ccy] for ccy in f.CCY[1:]])
                    np.testing.assert_allclose(expected,cal,atol=1e-12)
    assert sum(r['status']=='reused' for r in rows)==8
    # A positive shared slope/node offset changes levels but cannot improve within-currency ranks.
    cm=pd.read_csv(f.OUT/'currency_metrics_confusion.csv')
    orig=cm[cm.variant=='original'];cal=cm[cm.variant=='calibrated']
    matched=orig.merge(cal,on=['policy','condition','seed','split','currency'],suffixes=('_o','_c'))
    np.testing.assert_allclose(matched.auc_o,matched.auc_c,atol=1e-12)
    # Preserve both zero conventions instead of changing denominators.
    m=f.metrics(np.array([0.,-1.,1.]),np.array([0.,0.,1.]))
    assert m['binary_accuracy']==1 and m['mean_hit']==2/3
    scopes=pd.read_csv(f.OUT/'selected_scope_seed_results.csv')
    for _,g in scopes.groupby('selection_id'):
        assert set(g.seed)==set(range(5))
    grid=pd.read_parquet(f.OUT/'ALL_scope_search.parquet')
    assert (grid.n_dates>=126).all()
    np.testing.assert_allclose(grid.minimum_advantage,
        grid.mean_hit-grid[['baseline_A3','baseline_mlp','baseline_transformer','baseline_train_probability']].max(axis=1),atol=1e-12)
    assert (grid[grid.candidate=='A4_calibrated'].comparator_variant=='calibrated').all()
    print('PASS: 60 conditions; equal initial states/samples; 14-day FX lag; checkpoint prediction equality; raw labels/zero conventions; same calibration for all neural models; unchanged within-currency ranks; all five seeds and matched scopes.')


if __name__=='__main__':main()
