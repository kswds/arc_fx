"""Past realised reliability per (receiver i, sender j, regime r) cell, using only OOS records whose targets are known at t.

A record at timeline position s (decision row t_s, target ends at noon of t_s + 1) is usable at decision t only if t_s <= t - 2.
rho_cell = S_ed / S_dd over usable dates of the cell. Empirical Bayes:
    rho_pair_EB = S_ed,pair / (S_dd,pair + K_PAIR * m)                     (prior 0: "no information")
    rho_cell_EB = (S_ed,cell + K_CELL * m * rho_pair_EB) / (S_dd,cell + K_CELL * m)
with m = mean d^2 of the pair over the usable history (fallback: the current reference scale sd_ref^2).
se_cell = sqrt(sum (ed - rho d^2)^2) / (S_dd,cell + K_CELL m)  (no autocorrelation correction).
"""
from __future__ import annotations

import numpy as np

K_CELL, K_PAIR, LAG = 50., 100., 2


def lagged_cumsum(x, lag=LAG):
    c = np.cumsum(x, 0)
    out = np.zeros_like(c)
    out[lag:] = c[:-lag]
    return out


def features(e, d, reg, R, sd_ref):
    """e (n,10), d (n,10,11), reg (n,), sd_ref (n,10,11). Returns dict of (n,10,11) arrays."""
    oh = np.eye(R)[reg][:, None, None, :]                              # (n,1,1,R)
    ed = e[:, :, None] * d; dd = d ** 2
    def cell(v):
        return lagged_cumsum(v[..., None] * oh)                        # (n,10,11,R)
    S_ed, S_dd, N = cell(ed), cell(dd), cell(np.ones_like(dd))
    S_e2, S_edd, S_d4 = cell(ed ** 2), cell(ed * dd), cell(dd ** 2)
    pick = lambda A: np.take_along_axis(A, reg[:, None, None, None].repeat(10, 1).repeat(11, 2), -1)[..., 0]
    s_ed, s_dd, n = pick(S_ed), pick(S_dd), pick(N)
    p_ed, p_dd, p_n = S_ed.sum(-1), S_dd.sum(-1), N.sum(-1)
    m = np.where(p_n > 0, p_dd / np.maximum(p_n, 1), sd_ref ** 2)
    rho_pair = p_ed / (p_dd + K_PAIR * m)
    den = s_dd + K_CELL * m
    rho_cell = (s_ed + K_CELL * m * rho_pair) / den
    s2 = pick(S_e2) - 2 * rho_cell * pick(S_edd) + rho_cell ** 2 * pick(S_d4)
    se = np.sqrt(np.maximum(s2, 0)) / den
    with np.errstate(all="ignore"):
        raw = np.where(s_dd > 0, s_ed / np.maximum(s_dd, 1e-30), 0.)
    return dict(rho_cell=rho_cell, rho_pair=rho_pair, n_cell=n, se_cell=se, rho_raw=raw)
