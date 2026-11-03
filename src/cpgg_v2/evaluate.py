"""C-block evaluation for CPGG-v2: model predictions, primary paired loss differences (H1-H3), secondary metrics, selector diagnostics."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from cpgg.data import CCY, ROOT
from cpgg.evaluate import boot_mean, secondary, stress_masks

from . import selection as SL
from .message_linear import candidate_mask
from .pipeline import date_loss

CM = candidate_mask()
PAIRS = [(i, j) for i in range(10) for j in range(11) if CM[i, j]]


def v1_predictions(name, n):
    p = ROOT / "results/cpgg/runs/lag1/s0/C_predictions.csv.gz"
    if not p.exists():
        return None
    f = pd.read_csv(p)
    f = f[f.model == name]
    return f.pred_z.to_numpy().reshape(n, 10) if len(f) == 10 * n else None


def models(res, res_v1=None, seed=0):
    tl, pos, data, sc = res["tl"], res["pos"], res["data"], res["C"]
    C = data["C"]; b = tl["b"][pos["C"]]; n = len(b)
    main, dname = res["chosen_ridge"]["S"], res["chosen_ridge"]["D"]
    M = sc[main]["M"]
    out = {"Own-only": np.zeros((n, 100)), "Dense (M-v2)": np.ones((n, 100)),
           f"G-loss-S + placebo c0 (main, {main})": M,
           "G-loss-S + threshold 0": sc[main]["M0"],
           "G-loss-S + max(c0, 0)": (sc[main]["g"] > max(sc[main]["c0"], 0)).astype(float),
           f"G-loss-D + placebo c0 ({dname})": sc[dname]["M"],
           "Cell rule: rho_EB - 1.645 se > 0": SL.cell_rule(C, 0.), "Cell rule: rho_EB - 1.645 se > 0.5": SL.cell_rule(C, .5),
           "Random same K as main": SL.random_same_k(M, seed + 5),
           "Static (i, j, regime) cells": SL.static_cells(data["real_rec"], tl["reg"][pos["C"]], tl["R"])[0],
           "Aux: G-loss-S sequential greedy": sc[main]["Mg"], "Aux: G-loss-S shrink by rho_EB": SL.shrink(M, C)}
    for k in ("S", "D"):
        best = res["chosen"][k]
        if best != res["chosen_ridge"][k]:
            out[f"Aux: G-loss-{k} best incl. MLP ({best})"] = sc[best]["M"]
    preds = {k: (SL.predict(b, C["d"], A), A) for k, A in out.items()}
    if res_v1 is not None:
        v1 = res_v1["C"][res_v1["chosen_ridge"]["S"]]["M"]
        bv1 = res_v1["tl"]["b"][res_v1["pos"]["C"]]
        preds[f"M-v1 messages + G-loss-S ({res_v1['chosen_ridge']['S']})"] = (SL.predict(bv1, res_v1["data"]["C"]["d"], v1), v1)
        preds["Dense (M-v1)"] = (SL.predict(bv1, res_v1["data"]["C"]["d"], np.ones_like(v1)), np.ones_like(v1))
    for nm, lab in (("Selected residual (G-loss)", "CPGG v1 G-loss (checkpoint run)"), ("Selected residual (G-rel, main)", "CPGG v1 G-rel (checkpoint run)"),
                    ("Own-only (fold mean)", "CPGG v1 Own-only (checkpoint run)")):
        p = v1_predictions(nm, n)
        if p is not None:
            preds[lab] = (p, None)
    return preds


def primary(preds, tl, pos):
    L = {k: date_loss(p, tl, pos["C"]) for k, (p, _) in preds.items()}
    main = next(k for k in preds if "(main" in k)
    pairs = {"H1: Dense (M-v2) vs Own-only": ("Dense (M-v2)", "Own-only"), "H2: main vs Dense (M-v2)": (main, "Dense (M-v2)"),
             "H3: main vs Own-only": (main, "Own-only")}
    pairs.update({f"{k} vs Own-only": (k, "Own-only") for k in preds if k not in ("Own-only",) and not k.startswith("H")})
    out = {}
    for name, (a, b) in pairs.items():
        x = L[b] - L[a]                                                  # positive = a better
        out[name] = {f"block{bl}": boot_mean(x, block=bl) for bl in (10, 5, 20)}
    return out, L


def secondary_all(preds, tl, pos, d):
    c = SimpleNamespace(z=torch.as_tensor(tl["z"][pos["C"]]).float(), y=torch.as_tensor(tl["y"][pos["C"]]).float(),
                        logvol=torch.as_tensor(tl["logvol"][pos["C"]]).float())
    up = (d.y[d.rows["train"], 1:] > 0).mean(0).clip(.01, .99)
    st = stress_masks(d, tl["rows"][pos["C"]])
    out = {}
    L0 = date_loss(np.zeros_like(tl["b"][pos["C"]]) + tl["b"][pos["C"]], tl, pos["C"])
    for k, (p, A) in preds.items():
        zp = torch.as_tensor(p).float()
        m = secondary(zp, c, up)
        m["brier"] = float(((torch.sigmoid(zp) - (c.y > 0).float()) ** 2).mean())
        m["K_mean"] = float(SL.as3(A).sum(-1).mean()) if A is not None else np.nan
        Lk = date_loss(p, tl, pos["C"])
        m.update({f"gain_vs_own_{s}": float((L0 - Lk)[msk].mean()) for s, msk in st.items()})
        out[k] = m
    return out


def deciles(est, real, dates, k=10):
    q = np.quantile(est, np.linspace(0, 1, k + 1)); b = np.clip(np.digitize(est, q[1:-1]), 0, k - 1)
    rows = []
    for g in range(k):
        m = b == g
        per_date = pd.Series(real[m]).groupby(dates[m]).mean()
        se = per_date.std() / np.sqrt(max(len(per_date), 1))
        rows.append(dict(decile=g + 1, est_mean=float(est[m].mean()), realized_mean=float(real[m].mean()),
                         lo=float(real[m].mean() - 1.96 * se), hi=float(real[m].mean() + 1.96 * se), n=int(m.sum())))
    return rows


def selector_diag(res, name):
    tl, pos, data, sc = res["tl"], res["pos"], res["data"], res["C"]
    c = res["cal"][name]; C = data["C"]; r5 = data["rec5"]
    out = dict(config=name, c0=sc[name]["c0"], placebo_rate_C=sc[name]["fake_rate"], placebo_rate_records=sc[name]["fake_rate_rec"])
    for lab, dd in (("block5", r5), ("C", C)):
        raw = c.raw(dd); calv = c(dd).ravel(); D = dd["D"].ravel(); dates = dd["date"].ravel()
        out[f"calib_{lab}_raw"] = deciles(raw, D, dates); out[f"calib_{lab}_cal"] = deciles(calv, D, dates)
        out[f"corr_{lab}_raw"] = float(np.corrcoef(raw, D)[0, 1]); out[f"corr_{lab}_cal"] = float(np.corrcoef(calv, D)[0, 1])
    M = sc[name]["M"]; D = C["D"]
    sel_dates = M.sum(1) > 0
    per_date = np.where(sel_dates, (M * D).sum(1) / np.maximum(M.sum(1), 1), np.nan)
    out["selected_gain_per_date"] = boot_mean(per_date[sel_dates]) if sel_dates.sum() > 5 else None
    out["selected_gain_elementwise_mean"] = float(D[M > 0].mean()) if M.sum() else None
    out["n_selected"] = int(M.sum()); out["dates_with_selection"] = int(sel_dates.sum())
    K = SL.as3(M).sum(-1)
    out["K_hist"] = np.bincount(K.ravel().astype(int), minlength=11).tolist()
    out["K_mean"] = float(K.mean()); out["K0"] = float((K == 0).mean())
    reg = tl["reg"][pos["C"]]
    out["K_by_regime"] = {int(r): float(K[reg == r].mean()) for r in range(tl["R"])}
    A3 = SL.as3(M) > 0
    inter = (A3[1:] & A3[:-1]).sum(-1); uni = (A3[1:] | A3[:-1]).sum(-1)
    out["jaccard_adjacent"] = float((inter[uni > 0] / uni[uni > 0]).mean()) if (uni > 0).any() else None
    e = tl["e"][pos["C"]]; d3 = tl["d"][pos["C"]]
    ed = np.stack([e[:, i] * d3[:, i, j] for i, j in PAIRS], 1); dd = np.stack([d3[:, i, j] ** 2 for i, j in PAIRS], 1)
    out["by_regime"] = {int(r): dict(sel_rate=float(M[reg == r].mean()),
                                     rho_selected=float((ed * M)[reg == r].sum() / max((dd * M)[reg == r].sum(), 1e-30)) if (M[reg == r]).sum() else None,
                                     rho_all=float(ed[reg == r].sum() / dd[reg == r].sum())) for r in range(tl["R"])}
    freq = M.mean(0); top = np.argsort(-freq)[:10]
    out["top_pairs"] = [dict(pair=f"{CCY[PAIRS[k][0] + 1]}<-{CCY[PAIRS[k][1]]}", rate=float(freq[k]),
                             rho_selected=float((ed[:, k] * M[:, k]).sum() / max((dd[:, k] * M[:, k]).sum(), 1e-30)) if M[:, k].sum() else None,
                             rho_all=float(ed[:, k].sum() / dd[:, k].sum())) for k in top if freq[k] > 0]
    return out


def message_diag(res):
    tl = res["tl"]
    out = dict(fit=tl["fit_d_stats"], applied=[])
    for f in range(4):
        m = tl["fold"] == f
        e = tl["e"][m][:, :, None]; dd = tl["d"][m]
        out["applied"].append(dict(fold=f + 1 if f < 3 else "final(C)", d_mean=float(dd[:, CM].mean()), d_sd=float(dd[:, CM].std()),
                                   rho=float((e * dd)[:, CM].sum() / (dd[:, CM] ** 2).sum()),
                                   rho_by_regime={int(r): float((e * dd)[tl["reg"][m] == r][:, CM].sum() / (dd[tl["reg"][m] == r][:, CM] ** 2).sum())
                                                  for r in range(tl["R"]) if (tl["reg"][m] == r).any()}))
    if tl["W"]:
        Wf = tl["W"][-1].ravel()
        out["coef_corr_with_final"] = [float(np.corrcoef(W.ravel(), Wf)[0, 1]) for W in tl["W"][:3]]
        out["coef_corr_adjacent_folds"] = [float(np.corrcoef(tl["W"][k].ravel(), tl["W"][k + 1].ravel())[0, 1]) for k in range(3)]
        lam = tl["chosen"][-1][..., 0][CM.T] if False else None
    rec, C = tl["fold"] < 3, tl["fold"] == 3
    pr = lambda m: ((tl["e"][m][:, :, None] * tl["d"][m]).sum(0) / (tl["d"][m] ** 2).sum(0))[CM]
    out["pair_rho_corr_records_vs_C"] = float(np.corrcoef(pr(rec), pr(C))[0, 1])
    return out
