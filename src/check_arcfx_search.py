"""Behavior checks for alignment, leakage boundaries, graph paths, and labels."""
import numpy as np
import torch
import run_arcfx_search as search
from arcfx_search_models import SearchModel
from data_pipeline import set_seed, fx_to_log
import pandas as pd


def main():
    torch.set_num_threads(2)
    search.N=len(pd.read_csv(search.ROOT/'data/processed/fx_panel_pit.csv'))
    fold=search.Fold('check',int(search.N*.5),int(search.N*.6))
    ix=fold.train[20:24]
    b=fold.batch(ix)
    for key,value in b.items():
        source=getattr(fold.p,key)
        np.testing.assert_array_equal(value[:,-1].numpy(),source[ix])
        np.testing.assert_array_equal(fold.batch(ix,True)[key][:,-1].numpy(),source[ix-1])
    assert pd.to_datetime(fold.p.merged.Target_Date.iloc[fold.train[-1]]) < fold.p.merged.Date.iloc[fold.val[0]]
    raw=pd.read_csv(search.ROOT/'data/processed/fx_panel_pit.csv')
    for ccy in search.CCY[1:]:
        expected=fx_to_log(raw['Target_'+ccy+'_FX'],ccy)-fx_to_log(raw[ccy+'_FX'],ccy)
        np.testing.assert_allclose(expected,raw['TARGET_'+ccy+'_FX_RET_FWD1'],atol=1e-12)
    for encoder in ['original','gru','mlp']:
        for loss in ['bce','dual_0.03']:
            for residual in ([False,True] if encoder=='original' else [True]):
                spec=dict(encoder=encoder,loss=loss,residual=residual)
                set_seed(123);model=SearchModel(fold.p.dims,search.CCY,spec);model.eval()
                z,r=model(b);assert torch.equal(z[:,0],torch.zeros(len(ix)))
                objective,_=search.loss_fn(model,z,r,ix,fold,spec)
                objective.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if r is not None: assert torch.equal(r[:,0],torch.zeros(len(ix)))
                original=z.detach().clone();z,_=model(b,zero_graph=True)
                assert float((z-original).abs().max())>1e-7
                if residual:
                    assert float(model.last['components']['rel'].abs().max())==0
    # The BCE objective must follow raw sign, even when the centered target flips.
    spec=dict(encoder='gru',loss='bce');model=SearchModel(fold.p.dims,search.CCY,spec)
    z,r=model(b);before,_=search.loss_fn(model,z,r,ix,fold,spec)
    fold.yn[ix]=-fold.yn[ix]*100
    after,_=search.loss_fn(model,z,r,ix,fold,spec)
    assert before.item()==after.item()
    print('PASS: quote signs, window timing, purged boundaries, raw labels, USD gauge, finite gradients, graph influence, exact zero-message residual.')


if __name__=='__main__':main()
