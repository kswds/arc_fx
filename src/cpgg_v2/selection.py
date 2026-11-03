"""Inference rules on flattened candidates (n dates, 100 candidates = 10 receivers x 10 senders, self excluded).

Main:  M = 1[calibrated g > c0], A = M, s_i = b_i + sum_j A_ij d_ij  (independent decisions, no fixed K, 0..10 allowed).
Aux:   threshold 0; structured sequential greedy (S only); shrinkage A = M * clip(rho_cell_EB, 0, 1);
       cell rule (no selector): rho_cell_EB - 1.645 se > 0 (and > 0.5); random same-K; static (i, j, regime) cells.
"""
from __future__ import annotations

import numpy as np

from .message_linear import candidate_mask


def as3(x):
    return x.reshape(len(x), 10, 10)


def independent(cal, data, c0):
    return (cal(data) > c0).astype(float)


def greedy(cal, data, c0):
    """S selector only: gain of j given set S uses the known cross term, g_j|S = 2 (q_j - d_j sum_{k in S} d_k) - d_j^2."""
    n = len(data["d"])
    d3 = as3(data["d"])
    A = np.zeros((n, 10, 10))
    active = np.ones((n, 10), bool)
    for _ in range(10):
        cross = (A * d3).sum(-1, keepdims=True).repeat(10, -1).reshape(n, 100)
        g = as3(cal(data, cross=cross))
        g = np.where(A > 0, -np.inf, g)
        k = g.argmax(-1); best = np.take_along_axis(g, k[..., None], -1)[..., 0]
        add = active & (best > c0)
        if not add.any():
            break
        ii = np.where(add)
        A[ii[0], ii[1], k[add]] = 1.
        active &= add
    return A.reshape(n, 100)


def shrink(M, data):
    return M * np.clip(data["X"][..., 0], 0, 1)


def cell_rule(data, level=0.):
    rho, se = data["X"][..., 0], data["X"][..., 3]
    return ((rho - 1.645 * se) > level).astype(float)


def random_same_k(M, seed):
    rng = np.random.default_rng(seed)
    K = as3(M).sum(-1).astype(int)
    n = len(M)
    u = rng.random((n, 10, 10))
    rank = u.argsort(-1).argsort(-1)
    return (rank < K[..., None]).astype(float).reshape(n, 100)


def static_cells(records, reg_C, R):
    """records: flattened real candidates of blocks 3-5. Cells (pair, regime) with mean realised D > 0 are always connected."""
    pair, reg, D = records["pair"].ravel(), records["reg"].ravel(), records["D"].ravel()
    keep = np.zeros((100, R), bool)
    for r in range(R):
        for p in range(100):
            m = (pair == p) & (reg == r)
            keep[p, r] = m.any() and D[m].mean() > 0
    return keep[:, reg_C].T.astype(float), keep


def predict(b, d, A):
    """b (n,10), d (n,100), A (n,100) -> (n,10)"""
    return b + as3(A * d).sum(-1)


def full_d(data):
    return data["d"]
