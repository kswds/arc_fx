"""M-v2: per-(receiver i, sender j) linear message with regime interaction (hierarchical ridge).

    d_ij,t = sum_r 1[regime_t = r] (w_ij + u_ij,r)^T phi_j,t        penalty  lam*|w|^2 + lam*mult*|u|^2

phi_j,t (sender, all available at decision t under the lag1 policy): FX-return lag slots 0..4 of the CPGG FX channel (= returns up
to t-1, i.e. "lag 1..5"), |r_j|/v_j 1/5/20-day means, rate-change lags + flags, equity-return lags + flags. No global inputs.
Channel blocks (for the exact decomposition): fx = 8 columns, rate = 11, equity = 13.
Regimes (main): 0 calm, 1 RiskOff, 2 HighVol; RiskOff has priority when both hold. RiskOff uses the decision-row value; HighVol is an
FX-based regime and is shifted by the FX lag (as in cpgg.data). Thresholds come from the legacy pipeline (first 60% of merged rows).
"""
from __future__ import annotations

import numpy as np

from cpgg.data import CCY

CHANNELS = ("fx", "rate", "equity")
LAMBDAS = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
MULTS = (1., 10., 100.)


def regimes(d, kind="main"):
    m = d.merged
    if kind == "vix":                                         # synthetic check B: the injected regime itself
        return (np.nan_to_num(np.asarray(m["Global_VIX_change"], float)) > 0).astype(int), 2
    ro = np.nan_to_num(np.asarray(m["Regime_RiskOff"], float)) > .5
    hv = np.roll(np.nan_to_num(np.asarray(m["Regime_HighVol"], float)) > .5, d.meta.get("fx_lag", 1))
    r = np.zeros(len(m), int); r[hv] = 2; r[ro] = 1
    return r, 3


def _roll_mean(x, k):
    c = np.nancumsum(np.nan_to_num(x), 0); n = np.cumsum(np.isfinite(x), 0)
    cs = np.concatenate([np.zeros((k,) + x.shape[1:]), c])[:-k]
    ns = np.concatenate([np.zeros((k,) + x.shape[1:]), n])[:-k]
    with np.errstate(all="ignore"):
        return (c - cs) / np.maximum(n - ns, 1)


def sender_features(d, scale_rows):
    """(T, 11, 32) standardised with ``scale_rows`` statistics, plus channel index per column."""
    T = len(d.y)
    lag_slots = [d.names["fx"].index(f"FX_ret_1d_lag{k}") for k in range(5)]
    fxl = d.X["fx"][:, :, lag_slots]                                              # already lag1-consistent, USD = 0
    rr = np.full((T, 10), np.nan); rr[2:] = d.y[:-2, 1:]                           # latest return known at t (= y[t-2])
    a = np.abs(rr) / d.vol
    absf = np.zeros((T, 11, 3)); absf[:, 1:] = np.stack([np.nan_to_num(a), _roll_mean(a, 5), _roll_mean(a, 20)], -1)
    ex = lambda c: d.X[c][:, :, d.explicit[c]]
    P = np.concatenate([fxl, absf, ex("rate"), ex("equity")], -1).astype(float)
    ch = np.array([0] * 8 + [1] * ex("rate").shape[-1] + [2] * ex("equity").shape[-1])
    mu, sd = P[scale_rows].mean(0, keepdims=True), P[scale_rows].std(0, keepdims=True)
    sd = np.where(sd > 1e-9, sd, 1.)
    P = np.clip((P - mu) / sd, -8, 8)
    P[:, 0, :8] = 0.                                                               # USD has no FX return of its own
    return np.nan_to_num(P), ch


def design(P, reg, R):
    """(n, 11, F) x regime -> (n, 11, F*(1+R)) = [phi, phi*1[r=0], ..., phi*1[r=R-1]]"""
    oh = np.eye(R)[reg]
    return np.concatenate([P] + [P * oh[:, k, None, None] for k in range(R)], -1)


def penalty(F, R, lam, mult):
    return np.concatenate([np.full(F, lam), np.full(F * R, lam * mult)])


def fit(P, reg, R, E, rows, purge=5, n_chunks=4):
    """E: (T, 10) residual target (valid on rows). Returns W (11, F(1+R), 10) and chosen (lam, mult) per (j, i).
    Hyper-parameters per pair by time-ordered expanding CV (3 splits over 4 chunks, 5-row purge)."""
    rows = np.asarray(rows)
    Z = design(P[rows], reg[rows], R); Er = E[rows]
    F = P.shape[-1]; n = len(rows)
    edges = np.linspace(0, n, n_chunks + 1).astype(int)
    splits = [(np.arange(0, max(edges[k] - purge, 1)), np.arange(edges[k], edges[k + 1])) for k in range(1, n_chunks)]
    combos = [(l, m) for l in LAMBDAS for m in MULTS]
    W = np.zeros((11, Z.shape[-1], 10)); chosen = np.zeros((11, 10, 2))
    for j in range(11):
        X = Z[:, j]
        sse = np.zeros((len(combos), 10))
        for tr, va in splits:
            G, C = X[tr].T @ X[tr], X[tr].T @ Er[tr]
            for k, (l, m) in enumerate(combos):
                w = np.linalg.solve(G + np.diag(penalty(F, R, l, m)), C)
                sse[k] += ((Er[va] - X[va] @ w) ** 2).sum(0)
        pick = sse.argmin(0)
        G, C = X.T @ X, X.T @ Er
        sol = {k: np.linalg.solve(G + np.diag(penalty(F, R, *combos[k])), C) for k in set(pick.tolist())}
        for i in range(10):
            W[j, :, i] = sol[pick[i]][:, i]; chosen[j, i] = combos[pick[i]]
    for j in range(1, 11):                                                          # self pair (i = j-1) never used
        W[j, :, j - 1] = 0.
    return W, chosen


def messages(P, reg, R, W, ch, rows, shift_idx=None):
    """d (n, 10, 11) and channel parts (n, 10, 11, 3). shift_idx: optional (10, 11, n) source-row indices per pair (placebo)."""
    rows = np.asarray(rows)
    chf = np.tile(ch, 1 + R)
    if shift_idx is None:
        Z = design(P[rows], reg[rows], R)                                          # (n, 11, FR)
        prod = Z[:, :, :, None] * W[None]                                           # (n, 11, FR, 10)
        parts = np.stack([prod[:, :, chf == c].sum(2) for c in range(3)], -1)       # (n, 11, 10, 3)
        parts = parts.transpose(0, 2, 1, 3)
    else:
        parts = np.zeros((len(rows), 10, 11, 3))
        oh = np.eye(R)[reg[rows]]
        for i in range(10):
            for j in range(11):
                if j == i + 1:
                    continue
                Pf = P[shift_idx[i, j], j]                                          # sender inputs from shifted dates
                Zf = np.concatenate([Pf] + [Pf * oh[:, k, None] for k in range(R)], -1)
                pr = Zf * W[j, :, i]
                parts[:, i, j] = np.stack([pr[:, chf == c].sum(1) for c in range(3)], -1)
    for i in range(10):
        parts[:, i, i + 1] = 0.
    return parts.sum(-1), parts


def candidate_mask():
    m = np.ones((10, 11), bool)
    for i in range(10):
        m[i, i + 1] = False
    return m


NAMES = CCY
