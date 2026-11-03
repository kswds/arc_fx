import sys
from pathlib import Path

import numpy as np
import torch

try:
    import pytest
except ImportError:                                  # allow `python tests/test_cpgg.py` without pytest
    pytest = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpgg import data as D                      # noqa: E402
from cpgg import gain_dataset as GD             # noqa: E402
from cpgg.message_model import MessageModel     # noqa: E402
from cpgg.own_model import OwnModel             # noqa: E402
from cpgg.selection import greedy, predict      # noqa: E402
from cpgg.train import TD                       # noqa: E402


def _base():
    merged, y = D.load_merged()
    return merged, y, D.build("lag1", merged, y)


base = pytest.fixture(scope="module")(_base) if pytest else None


def test_no_future_leakage(base):
    merged, y, d = base
    t = 3000
    m2 = merged.copy()
    for c in m2.columns:
        if c not in ("Date", "Target_Date") and m2[c].dtype.kind == "f":
            m2.loc[t + 1:, c] = m2.loc[t + 1:, c] * 3 + 1
    d2 = D.build("lag1", m2, y)
    for ch in D.CHANNELS:
        np.testing.assert_allclose(d.X[ch][: t + 1], d2.X[ch][: t + 1], atol=1e-5)
    np.testing.assert_allclose(d.G[: t + 1], d2.G[: t + 1], atol=1e-5)
    np.testing.assert_allclose(d.vol[: t + 1], d2.vol[: t + 1])


def test_lag1_excludes_same_day_fx(base):
    merged, y, d = base
    t = 3000
    m2 = merged.copy()
    m2.loc[t, "JPY_FX_ret_1d"] = 0.05
    for col in ("JPY_FX_mom_5", "JPY_FX_mom_20", "JPY_FX_vol_5", "JPY_FX_vol_20", "JPY_FX_zscore_20"):
        m2.loc[t, col] = 0.05
    lag = D.build("lag1", m2, y); same = D.build("same_day", m2, y); same0 = D.build("same_day", merged, y)
    np.testing.assert_allclose(lag.X["fx"][t], d.X["fx"][t])
    np.testing.assert_allclose(lag.vol[t], d.vol[t])
    assert np.abs(same.X["fx"][t] - same0.X["fx"][t]).max() > 0


def test_no_target_columns_in_features(base):
    for names in base[2].names.values():
        assert not any("TARGET" in n.upper() for n in names)


def _ctx(seed=0, B=16):
    torch.manual_seed(seed)
    dims = dict(fx=15, rate=14, equity=16, macro=6)
    ex = dict(fx=np.arange(5, 15), rate=np.arange(3, 14), equity=np.arange(3, 16), macro=np.arange(4, 6))
    X = {c: torch.randn(B, 11, dims[c]) for c in dims}
    G = torch.randn(B, 20)
    own = OwnModel(dims, ex, 20)
    for p in own.parameters():
        torch.nn.init.normal_(p, std=.1)
    msg = MessageModel(dims, ex, 20)
    for p in msg.parameters():
        torch.nn.init.normal_(p, std=.1)
    own.eval(); msg.eval()
    with torch.no_grad():
        b, comps = own(X, G)
        out = msg(X, G)
    ctx = GD.Ctx(b, out, G, torch.rand(B, 10) + .5, X, torch.randn(B, 10), torch.randn(B, 10))
    return ctx, comps, out


def test_additive_subsets_and_decomposition():
    ctx, comps, out = _ctx()
    torch.testing.assert_close(comps.sum(-1), ctx.b)
    M = torch.rand(16, 10, 11) > .5
    M &= ~torch.eye(11, dtype=torch.bool)[1:]
    p = predict(ctx, M)
    recon = comps.sum(-1) + (out["part"] * M.unsqueeze(-1)).sum((2, 3))
    assert (p - recon).abs().max() < 1e-5
    assert out["d"][:, torch.arange(10), torch.arange(1, 11)].abs().max() == 0          # no self message
    s = torch.cat([torch.zeros(16, 1), p], 1); r = s[:, :, None] - s[:, None]
    assert (r + r.transpose(1, 2)).abs().max() < 1e-6
    assert (r[:, :, :, None] + r[:, None] - r[:, :, None, :]).abs().max() < 1e-5


class _Const:
    def __init__(self, v):
        self.v = v

    def gain(self, x, d):
        return torch.full((len(x),), self.v) + 1e-6 * d.abs()


def test_greedy_range_and_determinism():
    ctx, _, _ = _ctx()
    g_pos = greedy(ctx, [_Const(1.)]); g_neg = greedy(ctx, [_Const(-1.)])
    assert int(g_pos["M"].sum(-1).min()) == 10 and int(g_neg["M"].sum()) == 0
    g2 = greedy(ctx, [_Const(1.)])
    assert torch.equal(g_pos["M"], g2["M"]) and torch.equal(g_pos["order"], g2["order"])


if __name__ == "__main__":
    b = _base()
    for f in (test_no_future_leakage, test_lag1_excludes_same_day_fx, test_no_target_columns_in_features):
        f(b); print("pass", f.__name__)
    for f in (test_additive_subsets_and_decomposition, test_greedy_range_and_determinism):
        f(); print("pass", f.__name__)
