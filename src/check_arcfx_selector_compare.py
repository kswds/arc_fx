"""Behavioral and artifact checks for the controlled selector comparison."""
import argparse,json
import numpy as np
import pandas as pd
import torch
import arcfx_selector_compare as r
from arcfx_selector_models import select_edges
from arcfx_search_models import SearchModel
from models.arc_fx import StableRelationalScoreGraphFX


def smoke(data):
    score=torch.tensor([[[100.,.05,.3],[-.3,100.,.4],[3.,-2.,100.]]],requires_grad=True)
    a=select_edges(score,'soft_threshold',.1)
    torch.testing.assert_close(a[0,0],torch.tensor([0.,0.,.2]))
    torch.testing.assert_close(a[0,1],torch.tensor([-.2,0.,.3]))
    torch.testing.assert_close(a[0,2],torch.tensor([2.9/4.8,-1.9/4.8,0.]))
    assert float(a.abs().sum(-1).max())<=1.000001
    assert torch.equal(select_edges(torch.zeros(2,11,11),'soft_threshold',.1),torch.zeros(2,11,11))
    a.square().sum().backward();assert torch.isfinite(score.grad).all()
    torch.manual_seed(2);e=torch.randn(2,11,11)
    for mode,expected in [('topk',3),('dense_signed',10)]:
        a=select_edges(e,mode)
        assert ((a!=0).sum(-1)==expected).all()
        assert torch.diagonal(a,dim1=-2,dim2=-1).eq(0).all()
        assert float(a.abs().sum(-1).max())<=1.000001
    assert len(np.intersect1d(data.train,data.test))==0
    assert len(np.intersect1d(data.validation['pit_original'],data.test))==0
    assert len(data.validation['pit_original'])==len(data.validation['overlap50'])==1048
    assert len(data.overlap)==524 and len(data.test)==1049 and len(data.future)==525
    assert len(np.intersect1d(data.train,data.validation['overlap50']))==0
    batch=data.batch(data.train[:4])
    for backbone in r.CFG['backbones']:
        hashes=[]
        for mode,tau in r.candidates():
            torch.manual_seed(42);m=r.create(backbone,data,mode,tau);hashes.append(r.helpers.state_hash(m.state_dict()))
            m.eval();out=m(batch);loss=r.objective(backbone,out,data,data.train[:4]);loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        assert len(set(hashes))==1
    torch.manual_seed(42);m=r.create('original_arcfx',data,'topk',None)
    native=StableRelationalScoreGraphFX(n_ccy=11,currency_names=r.CCY,dims=data.p.dims,hidden=48,top_k=3,graph_rank=8,dropout=.25,edge_dropout=.05)
    native.load_state_dict(m.state_dict())
    for training in [False,True]:
        m.train(training);native.train(training)
        torch.manual_seed(99);a=m(batch)
        torch.manual_seed(99);b=native(batch)
        for key in ['rhat','adj','graph_gate']:
            torch.testing.assert_close(a[key],b[key],rtol=1e-5,atol=1e-6)
    m=r.create('compact_mlp',data,'dense_signed',None)
    native=SearchModel(data.p.dims,r.CCY,dict(encoder='mlp',loss='bce',graph='dense'))
    native.load_state_dict(m.state_dict());m.eval();native.eval()
    torch.testing.assert_close(m(batch)['rhat'],native(batch)[0],rtol=0,atol=0)


def artifacts(data):
    records=[json.loads(p.read_text()) for p in (r.OUT/'trajectories').glob('*/*/result.json')]
    assert len(records)==70
    for path in (r.OUT/'trajectories').glob('*/*/result.json'):
        record=json.loads(path.read_text());history=pd.read_csv(path.parent/'history.csv')
        assert len(history)==record['trained_epochs']<=20
        for protocol,state in record['protocols'].items():
            h=history.dropna(subset=[protocol+'_hit',protocol+'_objective'])
            best=h.sort_values([protocol+'_hit',protocol+'_objective','epoch'],ascending=[False,True,True]).iloc[0]
            assert int(best.epoch)==state['best_epoch']
            assert int(h.epoch.max())==state['stop_epoch']
            frame=pd.read_parquet(path.parent/protocol/'validation.parquet')
            assert len(frame)==10480
            np.testing.assert_array_equal(frame.target.values.reshape(-1,10),data.p.y_raw[data.validation[protocol],1:])
            np.testing.assert_allclose((np.sign(frame.prediction)==np.sign(frame.target)).mean(),state['validation']['mean_hit'])
    for backbone in r.CFG['backbones']:
        for seed in r.CFG['seeds']:
            group=[row for row in records if row['backbone']==backbone and row['seed']==seed]
            assert len(group)==7 and len({row['initial_state_hash'] for row in group})==1
    val=pd.read_csv(r.OUT/'all_validation_results.csv')
    selection=json.loads((r.OUT/'selection_frozen.json').read_text())
    assert len(selection)==12
    for choice in selection:
        if choice['selector']!='soft_threshold':continue
        g=val[(val.backbone==choice['backbone'])&(val.protocol==choice['protocol'])&(val.selector=='soft_threshold')]
        rank=g.groupby('tau').agg(hit=('mean_hit','mean'),loss=('validation_objective','mean')).reset_index().sort_values(['hit','loss','tau'],ascending=[False,True,True])
        assert choice['tau']==float(rank.iloc[0].tau)
        assert not choice['test_results_used']
    results=pd.read_csv(r.OUT/'test_results.csv');assert len(results)==180
    summary=pd.read_csv(r.OUT/'summary.csv')
    for _,row in summary.iterrows():
        g=results[(results.backbone==row.backbone)&(results.protocol==row.protocol)&(results.selector==row.selector)&(results.subset==row.subset)]
        assert set(g.seed)==set(range(5))
        np.testing.assert_allclose(g.mean_hit.mean(),row.mean_hit)
        np.testing.assert_allclose(g.mean_hit.max(),row.best_test_hit)
    for choice in selection:
        for seed in r.CFG['seeds']:
            dest=r.OUT/'selected_runs'/choice['backbone']/choice['protocol']/f"{choice['selector']}_s{seed}"
            f=pd.read_parquet(dest/'test_all.parquet')
            assert len(f)==10490 and not f.duplicated(['decision_date','currency']).any()
            assert f.groupby('decision_date').currency.nunique().eq(10).all()
            expected_overlap=5240 if choice['protocol']=='overlap50' else 0
            assert f.validation_overlap.sum()==expected_overlap
            np.testing.assert_array_equal(f.target.values.reshape(-1,10),data.p.y_raw[data.test,1:])
            result=json.loads((dest/'result.json').read_text())
            np.testing.assert_allclose((np.sign(f.prediction)==np.sign(f.target)).mean(),result['subsets']['test_all']['mean_hit'])
            m=r.create(choice['backbone'],data,choice['selector'],choice['tau'])
            source=r.OUT/'trajectories'/choice['backbone']/f"{r.candidate_name(choice['selector'],choice['tau'])}_s{seed}"/choice['protocol']/'best.pt'
            m.load_state_dict(torch.load(source,weights_only=True))
            pos=np.array([0,100,550,1048]);_,pred,_=r.evaluate(m,choice['backbone'],data,data.test[pos])
            np.testing.assert_allclose(pred,f.prediction.values.reshape(-1,10)[pos],rtol=2e-4,atol=2e-6)
    manifest=json.loads((r.OUT/'manifest.json').read_text())
    for path,value in manifest['source_data_hashes'].items():assert r.sha(r.ROOT/path)==value,path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);data=r.Data();smoke(data)
    if not args.smoke_only:artifacts(data)
    print('PASS: selector behavior, native model equivalence, fixed initialization, exact 50% validation overlap'+('' if args.smoke_only else ', validation-only tau selection, all 60 selected test runs, checkpoint/metric integrity'))
    if not args.smoke_only:r.dump(r.OUT/'verification.json',dict(passed=True,optimizer_trajectories=70,selected_test_runs=60,
        initialization_fixed=True,native_equivalence=True,validation_tau_only=True,all_targets_and_seeds=True))


if __name__=='__main__':main()
