"""Greedy dynamic graph construction from estimated conditional gains (deterministic).

Main rule : start from the empty set; add the candidate with the largest estimated gain while it is > 0.
Aux rule  : three selectors (fit on three time blocks of B); a candidate is eligible only if all three gains are > 0;
            among eligible candidates the largest mean gain is added. Fixed before any run.
No fixed K, no minimum degree, no connection cost. Greedy does not guarantee the globally best subset.
"""
from __future__ import annotations

import torch

from .gain_dataset import features


@torch.no_grad()
def greedy(ctx, selectors, rule="main", chunk=256):
    """Returns M (N,10,11) bool, order (N,10,10) sender index per step (-1 = stop), est (N,10,10) chosen estimated gain,
    first_est (N,10,11) estimated gain of every candidate at the empty set."""
    N = ctx.N
    Ms, orders, ests, firsts = [], [], [], []
    for s in range(0, N, chunk):
        n0 = torch.arange(s, min(N, s + chunk))
        n_ = len(n0)
        M = torch.zeros(n_, 10, 11, dtype=torch.bool)
        cand = ~torch.eye(11, dtype=torch.bool)[1:]
        done = torch.zeros(n_, 10, dtype=torch.bool)
        order = torch.full((n_, 10, 10), -1, dtype=torch.long)
        est = torch.full((n_, 10, 10), float("nan"))
        first = None
        nn_, ii, jj = torch.meshgrid(n0, torch.arange(10), torch.arange(11), indexing="ij")
        nf, i_f, j_f = nn_.reshape(-1), ii.reshape(-1), jj.reshape(-1)
        for step in range(10):
            cum_part = (ctx.part[n0] * M.unsqueeze(-1)).sum(2)                      # (n,10,4)
            absum = (ctx.d[n0].abs() * M).sum(2)
            size = M.sum(2).float()
            ln, li = (nf - s), i_f
            x = features(ctx, nf, i_f, j_f, cum_part[ln, li], absum[ln, li], size[ln, li])
            d = ctx.d[nf, i_f, j_f]
            gs = torch.stack([sel.gain(x, d) for sel in selectors]).view(len(selectors), n_, 10, 11)
            if rule == "main":
                score = gs[0]
            else:
                score = torch.where((gs > 0).all(0), gs.mean(0), torch.full_like(gs[0], -1.))
            if step == 0:
                first = gs.mean(0)
            free = cand.unsqueeze(0) & ~M & ~done.unsqueeze(-1)
            score = score.masked_fill(~free, -float("inf"))
            best, arg = score.max(-1)
            add = best > 0
            r_n, r_i = add.nonzero(as_tuple=True)
            M[r_n, r_i, arg[r_n, r_i]] = True
            order[r_n, r_i, step] = arg[r_n, r_i]
            est[r_n, r_i, step] = best[r_n, r_i]
            done |= ~add
            if done.all():
                break
        Ms.append(M); orders.append(order); ests.append(est); firsts.append(first)
    return dict(M=torch.cat(Ms), order=torch.cat(orders), est=torch.cat(ests), first_est=torch.cat(firsts))


def topk_by_magnitude(ctx, k):
    cand = ~torch.eye(11, dtype=torch.bool)[1:]
    score = ctx.d.abs().masked_fill(~cand.unsqueeze(0), -1.)
    idx = score.topk(k, -1).indices
    M = torch.zeros_like(cand.unsqueeze(0).expand(ctx.N, -1, -1))
    return M.scatter(-1, idx, True) & cand


def predict(ctx, M):
    """Prediction in vol units for a selection mask."""
    return ctx.b + (ctx.d * M).sum(-1)
