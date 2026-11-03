"""Expanding-window OOS gain records (section 2) and placebo candidates (section 5.1).

Train rows (10..3154) are split into 5 time-ordered blocks.
  own predictor O_k: fit on blocks 1..k (last 5 rows purged); predicts block k+1 (OOF) and, for k = 5, block C.
  messages fold f (f = 1, 2, 3): fit on residuals z - b of blocks 1..f+1 (last 5 rows purged); records on block f+2.
     b on block 1 is the in-sample O_1 prediction (no earlier model exists); blocks 2.. use the OOF O_k.
  final messages: blocks 1..5 (last 5 rows purged), applied to C.
Every record date t is therefore produced by an own model and messages that never saw the target of t.
Timeline = blocks 3, 4, 5 then C (contiguous rows 1268..4202). Test rows are never touched.
"""
from __future__ import annotations

import numpy as np
import torch

from cpgg.train import TD, fit_own, fit_message, ploss, predict_message, predict_own

from . import message_linear as ML

PURGE = 5


def blocks(d):
    return np.array_split(np.asarray(d.rows["train"]), 5)


def manifest(d):
    bl = blocks(d); dt = lambda r: str(np.datetime_as_string(d.dates[r], "D"))
    out = {f"block{k + 1}": dict(first_row=int(b[0]), last_row=int(b[-1]), n=len(b), start=dt(b[0]), end=dt(b[-1])) for k, b in enumerate(bl)}
    for f in range(3):
        fit = np.concatenate(bl[:f + 2])[:-PURGE]
        out[f"fold{f + 1}"] = dict(fit_rows=[int(fit[0]), int(fit[-1])], fit_dates=[dt(fit[0]), dt(fit[-1])], purge=PURGE,
                                   record_block=f + 3, own_model=f"O_{f + 2}")
    fit = np.concatenate(bl)[:-PURGE]
    out["final"] = dict(fit_rows=[int(fit[0]), int(fit[-1])], fit_dates=[dt(fit[0]), dt(fit[-1])], applies_to="C", own_model="O_5")
    C = d.rows["C"]; out["C"] = dict(first_row=int(C[0]), last_row=int(C[-1]), n=len(C), start=dt(C[0]), end=dt(C[-1]))
    out["roles"] = dict(selector_train="records of blocks 3-4 (folds 1-2)", calibration="records of block 5 (fold 3)",
                        threshold="placebo candidates of blocks 3-5", evaluation="C")
    return out


def pair_shifts(lo, hi, seed):
    """One random circular offset (>= 20 rows) per (i, j): placebo sender inputs come from rows lo..hi-1 (never test)."""
    rng = np.random.default_rng(seed)
    return rng.integers(20, hi - lo - 20, size=(10, 11))


def shift_index(rows, shifts, lo, hi):
    rows = np.asarray(rows)
    return lo + (rows[None, None, :] - lo + shifts[:, :, None]) % (hi - lo)


def huber_bce_gain(b, d, z, y):
    """D = L(b) - L(b + d), elementwise (n, 10, 11)."""
    bt, zt, yt = (torch.as_tensor(v).float() for v in (b, z, y))
    dt = torch.as_tensor(d).float()
    L0 = ploss(bt, zt, yt, reduce=False)[..., None]
    L1 = ploss(bt[..., None].expand_as(dt) + dt, zt[..., None].expand_as(dt), yt[..., None].expand_as(dt), reduce=False)
    return (L0 - L1).numpy()


def build(d, seed, regime_kind="main", message="v2", log=print, own=None, test=None):
    """Returns a dict with the OOS timeline arrays (records + C) and fold artefacts.

    test=None (default) never touches test rows. test="frozen": O_5 and the final messages are applied to test (fold 4).
    test="refit": O_6 and messages are refit on train + C (last 5 rows purged) and applied to test (fold 4).
    """
    td = TD(d)
    bl = blocks(d)
    T = len(d.y)
    z = (d.y[:, 1:] / d.vol)
    reg, R = ML.regimes(d, regime_kind)
    P, ch = ML.sender_features(d, np.concatenate(bl[:2]))
    # ---- own predictors O_1..O_5
    b_all = np.full((T, 10), np.nan); own_info = []
    O = []
    for k in (range(1, 6) if own is None else []):
        fit = np.concatenate(bl[:k])[:-PURGE]
        m, info = fit_own(td, fit, seed * 100 + k); O.append(m); own_info.append(info)
        tgt = bl[k] if k < 5 else d.rows["C"]
        b_all[tgt] = predict_own([m], td, tgt)[0].numpy()
    if own is None:
        b_all[bl[0]] = predict_own([O[0]], td, bl[0])[0].numpy()                     # in-sample (documented)
        if test == "frozen":
            b_all[d.rows["test"]] = predict_own([O[4]], td, d.rows["test"])[0].numpy()
        elif test == "refit":
            fit6 = np.concatenate(bl + [d.rows["C"]])[:-PURGE]
            m6, info6 = fit_own(td, fit6, seed * 100 + 6); own_info.append(info6)
            b_all[d.rows["test"]] = predict_own([m6], td, d.rows["test"])[0].numpy()
    else:
        b_all, own_info = own
    E = z - b_all
    # ---- messages per fold
    lo, hi = 10, int(d.rows["C"][-1]) + 1
    shifts = pair_shifts(lo, hi, seed + 7)
    fold_rows = [bl[2], bl[3], bl[4], d.rows["C"]] + ([d.rows["test"]] if test else [])
    W_list, chosen_list, v1_models = [], [], []
    tl = dict(rows=[], fold=[], d=[], part=[], d_fake=[], part_fake=[], sd_ref=[])
    fit_d_stats = []
    for f in range(len(fold_rows)):
        fit = np.concatenate(bl[:f + 2] if f < 3 else (bl + [d.rows["C"]] if f == 4 and test == "refit" else bl))[:-PURGE]
        rows = fold_rows[f]
        idx = shift_index(rows, shifts, lo, hi)
        if message == "v2":
            W, chosen = ML.fit(P, reg, R, E, fit)
            W_list.append(W); chosen_list.append(chosen)
            dd, pp = ML.messages(P, reg, R, W, ch, rows)
            df, pf = ML.messages(P, reg, R, W, ch, rows, shift_idx=idx)
            dfit, _ = ML.messages(P, reg, R, W, ch, fit)
        else:                                                                         # M-v1 neural messages (CPGG)
            m, info = fit_message(td, fit, torch.as_tensor(b_all[fit]).float(), seed * 10 + f)
            v1_models.append(info)
            o = predict_message(m, td, rows); dd = o["d"].numpy(); pp = o["part"].numpy()
            dfit = predict_message(m, td, fit)["d"].numpy()
            df, pf = v1_fake(m, td, rows, idx)
        sd_ref = dfit.std(0) + 1e-8                                                  # (10, 11), known at fold time
        fit_d_stats.append(dict(fit_rows=[int(fit[0]), int(fit[-1])], d_mean=float(dfit[:, ML.candidate_mask()].mean()),
                                d_sd=float(dfit[:, ML.candidate_mask()].std()),
                                rho_fit=float((E[fit][:, :, None] * dfit)[:, ML.candidate_mask()].sum() / (dfit[:, ML.candidate_mask()] ** 2).sum())))
        tl["rows"].append(rows); tl["fold"].append(np.full(len(rows), f)); tl["d"].append(dd); tl["part"].append(pp)
        tl["d_fake"].append(df); tl["part_fake"].append(pf); tl["sd_ref"].append(np.repeat(sd_ref[None], len(rows), 0))
        log(f"  fold {f + 1 if f < 3 else ('final' if f == 3 else 'test')}: fit rows {fit[0]}..{fit[-1]} -> {rows[0]}..{rows[-1]}")
    out = {k: np.concatenate(v) for k, v in tl.items()}
    r = out["rows"]
    out.update(b=b_all[r], z=z[r], y=d.y[r, 1:], e=E[r], reg=reg[r], R=R, logvol=np.log(d.vol[r]),
               own_info=own_info, fit_d_stats=fit_d_stats, W=W_list, chosen=chosen_list, ch=ch, P=P, shifts=shifts,
               b_all=b_all, E_all=E, reg_all=reg, message=message)
    rr = np.full((T, 10), np.nan); rr[2:] = d.y[:-2, 1:]
    a = np.abs(rr) / d.vol
    out["a1"] = np.nan_to_num(a)[r]; out["a5"] = np.nan_to_num(ML._roll_mean(a, 5))[r]
    out["D"] = huber_bce_gain(out["b"], out["d"], out["z"], out["y"])
    out["D_fake"] = huber_bce_gain(out["b"], out["d_fake"], out["z"], out["y"])
    out["ed"] = out["e"][:, :, None] * out["d"]; out["ed_fake"] = out["e"][:, :, None] * out["d_fake"]
    return out


@torch.no_grad()
def v1_fake(m, td, rows, idx, chunk=512):
    """Placebo for the neural messages: sender j's inputs (all channels) taken from the shifted rows, one forward pass per pair."""
    m.eval()
    n = len(rows)
    d = np.zeros((n, 10, 11)); part = np.zeros((n, 10, 11, 4))
    Xr, Gr = td.batch(rows)
    for i in range(10):
        for j in range(11):
            if j == i + 1:
                continue
            Xs, _ = td.batch(idx[i, j])
            X = {c: Xr[c].clone() for c in Xr}
            for c in X:
                X[c][:, j] = Xs[c][:, j]
            o = m(X, Gr)
            d[:, i, j] = o["d"][:, i, j].numpy(); part[:, i, j] = o["part"][:, i, j].numpy()
    return d, part
