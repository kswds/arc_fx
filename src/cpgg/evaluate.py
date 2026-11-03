"""Evaluation on C (validation block): primary paired loss differences, secondary metrics, selector diagnostics.

Primary metric (fixed): date-wise mean over the 10 currencies of the prediction loss (Huber + eta BCE, vol units);
paired differences between two models with a circular moving-block bootstrap (same blocks for both models).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from . import gain_dataset as GD
from .data import CCY, CFG, GLOBAL
from .selection import predict as sel_predict
from .train import LOSS, ploss

EV = CFG["evaluation"]


def block_index(n, block, reps, seed):
    rng = np.random.default_rng(seed)
    st = rng.integers(0, n, (reps, math.ceil(n / block)))
    return ((st[:, :, None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]


def boot_mean(x, block=None, reps=None, seed=20261103):
    block, reps = block or EV["block"], reps or EV["bootstrap_reps"]
    x = np.asarray(x, float)
    ix = block_index(len(x), block, reps, seed)
    b = x[ix].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.quantile(b, .025)), hi=float(np.quantile(b, .975)))


def date_loss(zp, ctx):
    return ploss(zp, ctx.z, ctx.y, reduce=False).mean(1).numpy()


def secondary(zp, ctx, ctx_train_up):
    z, y, vol = ctx.z, ctx.y, ctx.vol if hasattr(ctx, "vol") else None
    yv = y.numpy(); zpn = zp.numpy()
    hub = torch.nn.functional.huber_loss(zp, z, delta=LOSS["huber_delta"], reduction="none")
    lab = (y > 0).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(zp / LOSS["direction_temperature"], lab, reduction="none")
    up = torch.as_tensor(ctx_train_up).float()
    const = -(lab * torch.log(up) + (1 - lab) * torch.log(1 - up))
    ret = zp * torch.exp(ctx.logvol)
    hit = (np.sign(zpn) == np.sign(yv))
    ic_usd, ic_x = [], []
    iu = np.triu_indices(10, 1)
    for t in range(len(yv)):
        if np.std(zpn[t]) > 0:
            ic_usd.append(spearmanr(ret[t].numpy(), yv[t]).correlation)
            s = ret[t].numpy(); ps = (s[:, None] - s[None])[iu]; ys = (yv[t][:, None] - yv[t][None])[iu]
            ic_x.append(spearmanr(ps, ys).correlation)
    return dict(pred_loss=float(ploss(zp, z, y)), return_loss=float(hub.mean()), rmse=float(torch.sqrt(((ret - y) ** 2).mean())),
                dir_logloss=float(bce.mean()), skill_vs_half=float(1 - bce.mean() / math.log(2)),
                skill_vs_const=float(1 - bce.mean() / const.mean()), mean_hit=float(hit.mean()),
                ic_usd=float(np.nanmean(ic_usd)) if ic_usd else np.nan, ic_cross=float(np.nanmean(ic_x)) if ic_x else np.nan,
                **{f"hit_{CCY[k + 1]}": float(hit[:, k].mean()) for k in range(10)},
                **{f"loss_{CCY[k + 1]}": float(ploss(zp[:, k], z[:, k], y[:, k])) for k in range(10)})


def stress_masks(d, rows):
    m = d.merged
    n_train = int(len(m) * .6)
    num = lambda c: pd.to_numeric(m[c], errors="coerce").values

    def tail(v):
        s = np.sort(v[:n_train][np.isfinite(v[:n_train])]); return np.searchsorted(s, v, side="right") / len(s) >= .85
    fxv = np.abs(d.y[:, 1:]).mean(1)
    return dict(vix_stress=tail(np.abs(num("Global_VIX_change")))[rows], yield_stress=tail(np.abs(num("Global_US10Y_change")))[rows],
                fx_move_expost=tail(fxv)[rows])


def model_predictions(res, kinds=("loss", "rel")):
    c = res["ctx"]["C"]
    cand = (~torch.eye(11, dtype=torch.bool)[1:]).unsqueeze(0).expand(c.N, -1, -1)
    P = {"Own-only (fold mean)": (c.b, torch.zeros_like(cand)), "Dense residual": (sel_predict(c, cand), cand)}
    main = res["main_kind"]
    for k in kinds:
        g = res["graphs"][res["chosen"][k]]
        tag = "G-loss" if k == "loss" else "G-rel"
        lab = f"Selected residual ({tag}{', main' if k == main else ''})"
        P[lab] = (sel_predict(c, g["M"]), g["M"])
        if "aux" in res:
            P[f"Selected residual aux rule ({tag})"] = (sel_predict(c, res["aux"][k]["M"]), res["aux"][k]["M"])
    tk = {k: float(ploss(sel_predict(c, M), c.z, c.y)) for k, M in res["topk"].items()}
    kbest = min(tk, key=tk.get)
    P[f"Fixed Top-K |d| (K={kbest}, chosen on C)"] = (sel_predict(c, res["topk"][kbest]), res["topk"][kbest])
    Ms = res["static_edges"].unsqueeze(0).expand(c.N, -1, -1)
    P["Static selected residual"] = (sel_predict(c, Ms), Ms)
    if res.get("full"):
        P["Own-only full-train (A+B)"] = (res["full"]["own"], torch.zeros_like(cand))
        P["All-currency MLP full-train (A+B)"] = (res["full"]["allccy"], torch.ones_like(cand))
    return P, dict(topk_loss=tk, topk_chosen=kbest)


def main_label(P):
    return next(k for k in P if k.endswith("main)"))


def compare(P, c, base, other, blocks=None):
    a, b = date_loss(P[other][0], c), date_loss(P[base][0], c)
    out = {}
    for bl in [EV["block"]] + list(blocks or EV["block_sensitivity"]):
        r = boot_mean(b - a, bl)
        out[f"block{bl}"] = r
    return out


def graph_stats(M, d, rows, ctx):
    K = M.sum(-1).float()
    Kn = K.numpy()
    G = ctx.G.numpy()
    ro = G[:, GLOBAL.index("Regime_RiskOff")] > 0
    hv = G[:, GLOBAL.index("Regime_HighVol")] > 0
    inter = (M[1:] & M[:-1]).sum(-1).float(); uni = (M[1:] | M[:-1]).sum(-1).float()
    jac = torch.where(uni > 0, inter / uni.clamp_min(1), torch.ones_like(uni))
    hist = np.bincount(Kn.astype(int).ravel(), minlength=11) / Kn.size
    return dict(K_mean=float(Kn.mean()), K_sd=float(Kn.std()), K0=float((Kn == 0).mean()), K10=float((Kn == 10).mean()),
                K_hist=hist.round(4).tolist(), K_by_ccy={CCY[i + 1]: float(Kn[:, i].mean()) for i in range(10)},
                K_riskoff=float(Kn[ro].mean()) if ro.any() else np.nan, K_not_riskoff=float(Kn[~ro].mean()),
                K_highvol=float(Kn[hv].mean()) if hv.any() else np.nan, K_not_highvol=float(Kn[~hv].mean()),
                jaccard_adjacent=float(jac.mean()),
                sender_share={CCY[j]: float(M[..., j].float().mean()) for j in range(11)})


def reliability(ctx, groups):
    """Realised reliability sum(e d) / sum(d^2) at the empty set (e = z - b, d = single candidate message)."""
    e = (ctx.z - ctx.b).unsqueeze(-1).expand(-1, -1, 11)
    d = ctx.d
    cand = (~torch.eye(11, dtype=torch.bool)[1:]).unsqueeze(0).expand(ctx.N, -1, -1)
    out = {"all": float((e * d)[cand].sum() / (d ** 2)[cand].sum())}
    for name, mask in groups.items():
        m = mask & cand
        out[name] = float((e * d)[m].sum() / (d ** 2)[m].sum().clamp_min(1e-12)) if m.any() else np.nan
    return out


def rel_groups(ctx):
    N = ctx.N
    G = ctx.G.numpy()
    gs = {}
    for r in ("Regime_RiskOff", "Regime_HighVol", "Regime_LowVol"):
        v = torch.as_tensor(G[:, GLOBAL.index(r)] > 0).view(N, 1, 1).expand(-1, 10, 11)
        gs[r] = v; gs["not_" + r] = ~v
    for i in range(10):
        m = torch.zeros(N, 10, 11, dtype=torch.bool); m[:, i] = True; gs[f"recv_{CCY[i + 1]}"] = m
    for j in range(11):
        m = torch.zeros(N, 10, 11, dtype=torch.bool); m[:, :, j] = True; gs[f"send_{CCY[j]}"] = m
    a = ctx.d.abs()
    qs = torch.quantile(a.flatten()[a.flatten() > 0], torch.tensor([.2, .4, .6, .8]))
    edges = [0.] + qs.tolist() + [float("inf")]
    for q in range(5):
        gs[f"absd_q{q + 1}"] = (a > edges[q]) & (a <= edges[q + 1])
    return gs


def calibration(res, kind, reps=2000):
    """Estimated vs realised gain on C random-subset records (same generator recipe as B), by estimated-gain decile."""
    c = res["ctx"]["C"]
    sel = res["selectors"][res["chosen"][kind]]
    rec = GD.build(c, LOSS, CFG["gain_dataset"]["subsets_per_date_receiver"], torch.Generator().manual_seed(999))
    est = sel.gain(rec["x"], rec["d"]).numpy()
    D = rec["D"].numpy(); n = rec["n"].numpy(); w = rec["w"].numpy()
    q = np.quantile(est, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(q, est, side="right") - 1, 0, 9)
    N = c.N
    ix = block_index(N, EV["block"], reps, 4242)
    rows = []
    for k in range(10):
        m = dec == k
        # per-date sums so that the bootstrap resamples dates
        sD = np.bincount(n[m], weights=D[m] * w[m], minlength=N); sw = np.bincount(n[m], weights=w[m], minlength=N)
        bs = sD[ix].sum(1) / np.maximum(sw[ix].sum(1), 1e-12)
        rows.append(dict(decile=k + 1, est_mean=float(np.average(est[m], weights=w[m])), realized_mean=float(sD.sum() / sw.sum()),
                         lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=int(m.sum())))
    out = dict(deciles=rows, corr_est_realized=float(np.corrcoef(est, D)[0, 1]),
               share_est_positive=float((est > 0).mean()), realized_mean_when_est_pos=float(D[est > 0].mean()) if (est > 0).any() else np.nan,
               realized_mean_when_est_neg=float(D[est <= 0].mean()) if (est <= 0).any() else np.nan)
    if kind == "rel":
        rho = sel.rho(rec["x"]).numpy(); e = rec["e"].numpy(); d = rec["d"].numpy()
        qq = np.quantile(rho, np.linspace(0, 1, 6))
        bins = np.clip(np.searchsorted(qq, rho, side="right") - 1, 0, 4)
        out["rho_quintiles"] = [dict(q=k + 1, rho_est=float(rho[bins == k].mean()),
                                     rho_realized=float((e * d)[bins == k].sum() / max((d * d)[bins == k].sum(), 1e-12))) for k in range(5)]
    return out


def path_gains(res, kind):
    """Realised gain of the 1st, 2nd, ... edge added along the greedy path on C."""
    c = res["ctx"]["C"]
    g = res["graphs"][res["chosen"][kind]]
    order = g["order"]
    cum = c.b.clone()
    out = []
    for s in range(10):
        j = order[:, :, s]
        act = j >= 0
        if not act.any():
            break
        n, i = act.nonzero(as_tuple=True)
        dd = c.d[n, i, j[n, i]]
        gain = GD.realized_gain(c, n, i, cum[n, i], dd, LOSS)
        cum[n, i] = cum[n, i] + dd
        out.append(dict(step=s + 1, n=int(act.sum()), est_mean=float(g["est"][:, :, s][act].mean()), realized_mean=float(gain.mean()),
                        realized_ci=boot_mean(np.bincount(n.numpy(), weights=gain.numpy(), minlength=c.N) /
                                              np.maximum(np.bincount(n.numpy(), minlength=c.N), 1), reps=2000)))
    return out


def random_same_k(res, M, reps=20, seed=5):
    c = res["ctx"]["C"]
    cand = (~torch.eye(11, dtype=torch.bool)[1:]).unsqueeze(0).expand(c.N, -1, -1)
    K = M.sum(-1, keepdim=True)
    g = torch.Generator().manual_seed(seed)
    ls = []
    for _ in range(reps):
        sc = torch.rand(c.N, 10, 11, generator=g).masked_fill(~cand, 2.)
        R = (sc.argsort(-1).argsort(-1) < K) & cand
        ls.append(date_loss(sel_predict(c, R), c))
    return np.mean(ls, 0)


def pair_check(res, kind):
    """Pairs (j, k) with estimated single gains <= 0 at the empty set: realised gain of adding both together."""
    c = res["ctx"]["C"]
    est0 = res["graphs"][res["chosen"][kind]]["first_est"]
    cand = ~torch.eye(11, dtype=torch.bool)[1:]
    gains, single_both = [], []
    for j in range(11):
        for k in range(j + 1, 11):
            for i in range(10):
                if not (cand[i, j] and cand[i, k]):
                    continue
                m = (est0[:, i, j] <= 0) & (est0[:, i, k] <= 0)
                if m.sum() < 20:
                    continue
                n = m.nonzero().squeeze(-1); ii = torch.full_like(n, i)
                gp = GD.realized_gain(c, n, ii, c.b[n, i], c.d[n, i, j] + c.d[n, i, k], LOSS)
                gains.append(dict(recv=CCY[i + 1], j=CCY[j], k=CCY[k], n=int(len(n)), realized_pair=float(gp.mean())))
    df = pd.DataFrame(gains)
    return dict(n_pairs=len(df), share_pairs_positive=float((df.realized_pair > 0).mean()) if len(df) else np.nan,
                top=df.sort_values("realized_pair", ascending=False).head(5).to_dict("records") if len(df) else [])


def consistency(res):
    """Exact decomposition and bilateral antisymmetry / triangle checks on C (return units)."""
    c = res["ctx"]["C"]
    comps = res["comps"]["C"]
    g = res["graphs"][res["chosen"][res["main_kind"]]]
    vol = torch.exp(c.logvol)
    pred = sel_predict(c, g["M"]) * vol
    recon = (comps.sum(-1) + (c.part * g["M"].unsqueeze(-1)).sum((2, 3))) * vol
    s = torch.cat([torch.zeros(c.N, 1), pred], 1)
    r = s.unsqueeze(2) - s.unsqueeze(1)
    anti = (r + r.transpose(1, 2)).abs().max()
    tri = (r.unsqueeze(3) + r.unsqueeze(1) - r.unsqueeze(2)).abs().max()   # r_ij + r_jk - r_ik
    return dict(decomposition_max_abs_err=float((pred - recon).abs().max()), antisymmetry_max=float(anti), triangle_max=float(tri))


def evaluate_run(res, kinds=("loss", "rel")):
    c = res["ctx"]["C"]
    d = res["td"].d
    up = (d.y[np.concatenate([d.rows["A"], d.rows["B"]]), 1:] > 0).mean(0)
    P, tk = model_predictions(res, kinds)
    main = main_label(P)
    tab = {name: secondary(p, c, up) for name, (p, M) in P.items()}
    st = stress_masks(d, c.rows)
    for name, (p, M) in P.items():
        dl = date_loss(p, c)
        for k, m in st.items():
            tab[name][f"loss_{k}"] = float(dl[m].mean())
        tab[name]["K_mean"] = float(M.sum(-1).float().mean())
    prim = {"H1: Dense residual vs Own-only": compare(P, c, "Own-only (fold mean)", "Dense residual"),
            "H2: Selected (main) vs Dense residual": compare(P, c, "Dense residual", main)}
    others = {}
    for name in P:
        if name != "Own-only (fold mean)":
            others[f"{name} vs Own-only"] = compare(P, c, "Own-only (fold mean)", name, blocks=[])
        if name not in ("Dense residual",):
            others[f"{name} vs Dense residual"] = compare(P, c, "Dense residual", name, blocks=[])
    rnd = random_same_k(res, P[main][1])
    others["Selected (main) vs random same-K"] = dict(block10=boot_mean(rnd - date_loss(P[main][0], c)))
    sel = {}
    for k in kinds:
        g = res["graphs"][res["chosen"][k]]
        sel[k] = dict(config=res["chosen"][k], graph=graph_stats(g["M"], d, c.rows, c), calibration=calibration(res, k),
                      path=path_gains(res, k), pairs=pair_check(res, k))
        if "aux" in res:
            sel[k]["aux_graph"] = graph_stats(res["aux"][k]["M"], d, c.rows, c)
    rel = {s: reliability(res["ctx"][s], rel_groups(res["ctx"][s])) for s in ("A", "B", "C")}
    # residual alignment (6.1): message-training residual (A OOF) vs applied residual (B, C)
    align = {}
    for s in ("A", "B", "C"):
        cx = res["ctx"][s]
        e = (cx.z - cx.b)
        align[s] = dict(mean=float(e.mean()), sd=float(e.std()), corr_b_z=float(np.corrcoef(cx.b.flatten(), cx.z.flatten())[0, 1]),
                        b_sd=float(cx.b.std()))
    static = res["static_edges"]
    return dict(table=tab, primary=prim, others=others, selectors=sel, reliability=rel, residual_alignment=align,
                topk=tk, main_label=main, static_edges=int(static.sum()), consistency=consistency(res),
                lossC_by_config=res["lossC_by_config"], chosen=res["chosen"], main_kind=res["main_kind"])
