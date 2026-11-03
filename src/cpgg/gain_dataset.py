"""Out-of-sample predictive-gain records and the selector feature map.

All quantities are in receiver volatility units. For a (date t, receiver i, current set S, candidate j):
  D = loss(z, b + C_S) - loss(z, b + C_S + d_ij),   C_S = sum_{k in S} d_ik
computed with predictors (own fold models, message model) that never trained on date t.
The same feature map is used for training (random S) and for greedy inference (S built step by step).
"""
from __future__ import annotations

import numpy as np
import torch

from .own_model import pred_loss

BASE_NAMES = (["b", "cum", "pred_S", "set_size", "d", "part_fx", "part_rate", "part_equity", "part_macro", "a",
               "cum_fx", "cum_rate", "cum_equity", "cum_macro", "d_x_cum", "d_x_pred", "same_sign", "abs_d", "abs_cum_sum",
               "log_vol", "send_fx0", "send_eq0", "recv_fx0"] + [f"g{k}" for k in range(20)]
              + [f"recv_{k}" for k in range(10)] + [f"send_{k}" for k in range(11)])


class Ctx:
    """Per-date tensors needed to build selector features: b (N,10), d (N,10,11), part (N,10,11,4), a (N,10,11), G, vol, shocks."""

    def __init__(self, b, out, G, vol, X, z=None, y=None, rows=None):
        self.b, self.d, self.part, self.a = b, out["d"], out["part"], out["a"]
        self.G, self.logvol = G, torch.log(vol)
        fx, eq = X["fx"], X["equity"]
        self.send_fx0 = fx[:, :, 5]                   # FX_ret_1d lag0 (standardised), (N, 11)
        self.send_eq0 = eq[:, :, 3]                   # Stock_ret lag0
        self.z, self.y, self.rows = z, y, rows

    @property
    def N(self):
        return self.b.shape[0]


def features(ctx, n, i, j, cum_part, absum, size):
    """Feature rows for index vectors n (date), i (receiver 0..9), j (sender 0..10); cum_part (M,4), absum (M,), size (M,)."""
    b = ctx.b[n, i]
    part = ctx.part[n, i, j]
    d = part.sum(-1)
    cum = cum_part.sum(-1)
    predS = b + cum
    cols = [b, cum, predS, size / 10., d, part, ctx.a[n, i, j], cum_part, d * cum, d * predS,
            (torch.sign(d) == torch.sign(cum)).float(), d.abs(), absum, ctx.logvol[n, i],
            ctx.send_fx0[n, j], ctx.send_eq0[n, j], ctx.send_fx0[n, i + 1], ctx.G[n],
            torch.nn.functional.one_hot(i, 10).float(), torch.nn.functional.one_hot(j, 11).float()]
    return torch.cat([c if c.dim() == 2 else c.unsqueeze(-1) for c in cols], -1)


def realized_gain(ctx, n, i, predS, d, cfg):
    z, y = ctx.z[n, i], ctx.y[n, i]
    L = lambda p: pred_loss(p, z, y, cfg["huber_delta"], cfg["eta"], cfg["direction_temperature"], reduce=False)
    return L(predS) - L(predS + d)


def build(ctx, cfg_loss, n_sub, gen, include_empty=True):
    """Random-subset gain records for every (date, receiver). Returns dict of tensors (features, D, e, d, date index...)."""
    N = ctx.N
    cand = ~torch.eye(11, dtype=torch.bool)[1:]
    recs = {k: [] for k in ("x", "D", "e", "d", "n", "i", "j", "size")}
    for s in range(n_sub):
        if s == 0 and include_empty:
            S = torch.zeros(N, 10, 11, dtype=torch.bool)
        else:
            k = torch.randint(1, 10, (N, 10, 1), generator=gen)
            score = torch.rand(N, 10, 11, generator=gen).masked_fill(~cand, 2.)
            S = (score.argsort(-1).argsort(-1) < k) & cand
        cum_part = (ctx.part * S.unsqueeze(-1)).sum(2)                   # (N,10,4)
        absum = (ctx.d.abs() * S).sum(2)
        size = S.sum(2).float()
        free = cand.unsqueeze(0) & ~S
        n, i, j = free.nonzero(as_tuple=True)
        cp = cum_part[n, i]
        x = features(ctx, n, i, j, cp, absum[n, i], size[n, i])
        d = ctx.d[n, i, j]
        predS = ctx.b[n, i] + cp.sum(-1)
        recs["x"].append(x); recs["d"].append(d); recs["n"].append(n); recs["i"].append(i); recs["j"].append(j)
        recs["D"].append(realized_gain(ctx, n, i, predS, d, cfg_loss)); recs["e"].append(ctx.z[n, i] - predS)
        recs["size"].append(size[n, i])
    out = {k: torch.cat(v) for k, v in recs.items()}
    cnt = torch.bincount(out["n"], minlength=N).float()
    w = 1. / cnt[out["n"]]
    out["w"] = w / w.mean()                                              # every date carries the same total weight
    return out


def singles(ctx, cfg_loss):
    """D for adding each candidate to the empty set: (N,10,11), NaN on self pairs."""
    N = ctx.N
    D = torch.full((N, 10, 11), float("nan"))
    cand = ~torch.eye(11, dtype=torch.bool)[1:]
    n, i, j = cand.unsqueeze(0).expand(N, -1, -1).nonzero(as_tuple=True)
    D[n, i, j] = realized_gain(ctx, n, i, ctx.b[n, i], ctx.d[n, i, j], cfg_loss)
    return D
