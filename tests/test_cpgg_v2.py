"""CPGG-v2 unit tests: cell-reliability leakage, exact channel decomposition, structural consistency, deterministic inference.

cd src && PYTHONPATH=. python ../tests/test_cpgg_v2.py
"""
import numpy as np

from cpgg_v2 import cell_reliability as CR
from cpgg_v2 import message_linear as ML
from cpgg_v2 import selection as SL


def test_cell_features_no_future_leak():
    rng = np.random.default_rng(0)
    n = 200
    e = rng.standard_normal((n, 10)); d = rng.standard_normal((n, 10, 11)) * .1
    reg = rng.integers(0, 3, n); sd = np.full((n, 10, 11), .1)
    f0 = CR.features(e, d, reg, 3, sd)
    for t in (5, 50, 150):
        e2, d2 = e.copy(), d.copy()
        e2[t - 1:] = rng.standard_normal(e2[t - 1:].shape); d2[t - 1:] *= 7.          # records at t-1 and later are not usable at t
        f1 = CR.features(e2, d2, reg, 3, sd)
        for k in ("rho_cell", "rho_pair", "n_cell", "se_cell"):
            assert np.allclose(np.nan_to_num(f0[k][:t + 1]), np.nan_to_num(f1[k][:t + 1])), (k, t)
        assert (f0["n_cell"][t] == (reg[:t - 1] == reg[t]).sum()).all()                # records 0..t-2 of the same regime are used


def test_decomposition_and_structure():
    rng = np.random.default_rng(1)
    T, F, R = 60, 32, 3
    P = rng.standard_normal((T, 11, F)); P[:, 0, :8] = 0.
    reg = rng.integers(0, R, T); ch = np.array([0] * 8 + [1] * 11 + [2] * 13)
    E = rng.standard_normal((T, 10))
    W, _ = ML.fit(P, reg, R, E, np.arange(T))
    d, part = ML.messages(P, reg, R, W, ch, np.arange(T))
    assert np.abs(part.sum(-1) - d).max() < 1e-12
    for i in range(10):
        assert np.abs(d[:, i, i + 1]).max() == 0
    idx = np.tile(np.arange(T), (10, 11, 1))
    df, pf = ML.messages(P, reg, R, W, ch, np.arange(T), shift_idx=idx)             # zero shift == real messages
    assert np.abs(df - d).max() < 1e-10


class _Cal:
    def __init__(self, w):
        self.w = w

    def __call__(self, data, cross=0.):
        cross = np.broadcast_to(cross, data["d"].shape)
        return 2 * (self.w * data["d"] ** 2 - data["d"] * cross) - data["d"] ** 2


def test_deterministic_inference():
    rng = np.random.default_rng(2)
    data = dict(d=rng.standard_normal((30, 100)) * .1, X=rng.standard_normal((30, 100, 5)))
    cal = _Cal(.8)
    M1, M2 = SL.independent(cal, data, 0.), SL.independent(cal, data, 0.)
    G1, G2 = SL.greedy(cal, data, 0.), SL.greedy(cal, data, 0.)
    assert (M1 == M2).all() and (G1 == G2).all()
    assert SL.as3(G1).sum(-1).max() <= 10 and (SL.as3(G1).sum(-1) <= SL.as3(M1).sum(-1) + 10).all()
    R1, R2 = SL.random_same_k(M1, 3), SL.random_same_k(M1, 3)
    assert (R1 == R2).all() and (SL.as3(R1).sum(-1) == SL.as3(M1).sum(-1)).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
