"""Section 12 synthetic checks (pass rules fixed in configs/cpgg.json before running).

Source: an independent N(0,1) series s_t (seeded) placed in the explicit FX-return lag slots of the source currency (CHF):
lag-k slot at row t holds s_{t-k}. Injection into the receiver (JPY) target, in vol units:
    z_JPY,t <- z_JPY,t + c * s_{t-1} [* regime_t],   c = sqrt(SNR * var_A(z_JPY)),   y = z * vol.
Horizon: the usual 1-day target of row t. CHF's own slow FX features stay real; only the explicit slots are replaced.
Checks: A (forced edge vs oracle, SNR grid), B (regime-only signal), C (noise source SEK, duplicate source NOK), D (lags, gradients, scale, linear recovery, CPU/MPS).

python -m cpgg.synthetic_checks [--only A,B,C,D] [--tag attempt0]
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import gain_dataset as GD
from .data import CCY, CFG, ROOT, build
from .evaluate import block_index, boot_mean
from .train import LOSS, TD, ploss, run

S = CFG["synthetic"]
RECV, SRC, NOISE, DUP = (CCY.index(S[k]) for k in ("receiver", "source", "noise_source", "duplicate_source"))
OUT = ROOT / "results/cpgg/synthetic"


def make(d, snr, regime=False, extras=False, seed=11):
    d = copy.deepcopy(d)
    rng = np.random.default_rng(seed)
    T = len(d.y)
    s = rng.standard_normal(T)
    fx = d.X["fx"]
    lag_slots = [d.names["fx"].index(f"FX_ret_1d_lag{k}") for k in range(CFG["n_lags"])]
    miss_slots = [d.names["fx"].index(f"missing_FX_ret_1d_lag{k}") for k in range(CFG["n_lags"])]

    def place(j, series):
        for k, (sl, ms) in enumerate(zip(lag_slots, miss_slots)):
            v = np.roll(series, k); v[:k] = 0.
            fx[:, j, sl] = v; fx[:, j, ms] = 0.
    place(SRC, s)
    if extras:
        place(NOISE, rng.standard_normal(T))
        place(DUP, s + S["duplicate_noise_sd"] * rng.standard_normal(T))
    z = d.y[:, 1:] / d.vol
    c = float(np.sqrt(snr * z[d.rows["A"], RECV - 1].var()))
    reg = (np.nan_to_num(np.asarray(d.merged["Global_VIX_change"], float)) > 0) if regime else np.ones(T, bool)
    sig = np.zeros(T); sig[1:] = c * s[:-1]
    sig = sig * reg
    z[:, RECV - 1] += sig
    d.y = d.y.copy(); d.y[:, 1:] = z * d.vol
    d.meta["synthetic"] = dict(snr=snr, coef=c, regime=bool(regime), extras=bool(extras), source=CCY[SRC], receiver=CCY[RECV],
                               lag=1, seed=seed)
    return d, sig, reg, s


def jpy_losses(ctx, zp_jpy):
    i = RECV - 1
    return ploss(zp_jpy, ctx.z[:, i], ctx.y[:, i], reduce=False).numpy()


def forced(ctx):
    return ctx.b[:, RECV - 1] + ctx.d[:, RECV - 1, SRC]


def selection_rates(res, rows, reg):
    out = {}
    for k, name in res["chosen"].items():
        M = res["graphs"][name]["M"][:, RECV - 1]
        out[k] = dict(config=name, src=float(M[:, SRC].float().mean()), noise=float(M[:, NOISE].float().mean()),
                      dup=float(M[:, DUP].float().mean()), K=float(M.sum(-1).float().mean()))
        if reg is not None:
            sel = M[:, SRC].float().numpy(); g = reg[rows]
            ix = block_index(len(sel), CFG["evaluation"]["block"], 10000, 77)
            xs, gs = sel[ix], g[ix]
            diff = (xs * gs).sum(1) / np.maximum(gs.sum(1), 1) - (xs * ~gs).sum(1) / np.maximum((~gs).sum(1), 1)
            out[k].update(src_regime=float(sel[g].mean()), src_non_regime=float(sel[~g].mean()),
                          diff=float(sel[g].mean() - sel[~g].mean()), lo=float(np.quantile(diff, .025)), hi=float(np.quantile(diff, .975)))
    return out


def est_gain_given(res, kind, j, S_list):
    """Estimated gain of adding j for receiver JPY on C dates given the fixed set S_list."""
    c = res["ctx"]["C"]
    sel = res["selectors"][res["chosen"][kind]]
    N = c.N
    n = torch.arange(N); i = torch.full((N,), RECV - 1); jj = torch.full((N,), j)
    if S_list:
        cum = sum(c.part[:, RECV - 1, k] for k in S_list); absum = sum(c.d[:, RECV - 1, k].abs() for k in S_list)
    else:
        cum = torch.zeros(N, 4); absum = torch.zeros(N)
    x = GD.features(c, n, i, jj, cum, absum, torch.full((N,), float(len(S_list))))
    return sel.gain(x, c.d[:, RECV - 1, j]).numpy()


def check_A(base, seed, log):
    rows = []
    for snr in S["snr"]:
        d, sig, reg, s = make(base, snr)
        res = run(d, seed, None, full_baselines=False, aux=False, log=log)
        c = res["ctx"]["C"]
        i = RECV - 1
        L_own = jpy_losses(c, c.b[:, i])
        L_f = jpy_losses(c, forced(c))
        L_o = jpy_losses(c, c.b[:, i] + torch.as_tensor(sig[c.rows]).float())
        gf, go = boot_mean(L_own - L_f), boot_mean(L_own - L_o)
        ratio = gf["mean"] / go["mean"] if go["mean"] > 0 else np.nan
        dC = c.d[:, i, SRC].numpy(); sC = s[c.rows - 1]
        slope = float(np.polyfit(sC, dC, 1)[0])
        resid = (res["ctx"]["A"].z - res["ctx"]["A"].b)[:, i].numpy(); sA = s[res["ctx"]["A"].rows - 1]
        row = dict(snr=snr, coef=d.meta["synthetic"]["coef"], forced_gain=gf, oracle_gain=go, ratio=ratio,
                   message_slope_on_source=slope, oof_residual_slope=float(np.polyfit(sA, resid, 1)[0]),
                   selection=selection_rates(res, c.rows, None),
                   passed=bool(ratio >= .5 and gf["lo"] > 0) if snr in (1.0, .3) else None)
        log(json.dumps(row, default=float))
        rows.append(row)
    return dict(rows=rows, passed=all(r["passed"] for r in rows if r["passed"] is not None))


def check_B(base, seed, log):
    d, sig, reg, s = make(base, S["regime_snr"], regime=True)
    res = run(d, seed, None, full_baselines=False, aux=False, log=log)
    c = res["ctx"]["C"]
    i = RECV - 1
    g = reg[c.rows]
    L_own = jpy_losses(c, c.b[:, i]); L_f = jpy_losses(c, forced(c))
    e = (c.z - c.b)[:, i].numpy(); dd = c.d[:, i, SRC].numpy()
    msg = {k: dict(forced_gain=boot_mean((L_own - L_f)[m], reps=2000), mean_abs_d=float(np.abs(dd[m]).mean()),
                   realized_rho=float((e[m] * dd[m]).sum() / (dd[m] ** 2).sum())) for k, m in (("regime", g), ("non_regime", ~g))}
    sel = selection_rates(res, c.rows, reg)
    contrib = {k: float(np.abs(dd * res["graphs"][res["chosen"][k]]["M"][:, i, SRC].numpy())[g].mean() -
                        np.abs(dd * res["graphs"][res["chosen"][k]]["M"][:, i, SRC].numpy())[~g].mean()) for k in res["chosen"]}
    out = dict(message=msg, selection=sel, contribution_abs_regime_minus_non=contrib,
               passed={k: bool(v["lo"] > 0) for k, v in sel.items()})
    log(json.dumps(out, default=float))
    return out


def check_C(base, seed, log):
    d, sig, reg, s = make(base, S["redundancy_snr"], extras=True)
    res = run(d, seed, None, full_baselines=False, aux=False, log=log)
    c = res["ctx"]["C"]
    sel = selection_rates(res, c.rows, None)
    out = dict(selection=sel)
    for k in res["chosen"]:
        g_src = est_gain_given(res, k, SRC, [])
        g_dup = est_gain_given(res, k, DUP, [SRC])
        g_dup0 = est_gain_given(res, k, DUP, [])
        out[k] = dict(est_src_alone=float(g_src.mean()), est_dup_given_src=float(g_dup.mean()), est_dup_alone=float(g_dup0.mean()),
                      noise_rate=sel[k]["noise"],
                      passed_noise=bool(sel[k]["noise"] <= .10), passed_dup=bool(g_dup.mean() <= .5 * g_src.mean()))
    out["passed"] = {k: bool(out[k]["passed_noise"] and out[k]["passed_dup"]) for k in res["chosen"]}
    log(json.dumps(out, default=float))
    return out


def check_D(base, seed, log):
    d, sig, reg, s = make(base, 1.0)
    td = TD(d)
    lag1 = d.names["fx"].index("FX_ret_1d_lag1")
    r = np.arange(100, 110)
    reach = float(np.abs(d.X["fx"][r, SRC, lag1] - s[r - 1]).max())
    from .message_model import MessageModel
    torch.manual_seed(0)
    m = MessageModel(td.dims, d.explicit, td.G.shape[1])
    X, G = td.batch(r)
    X = {k: v.clone().requires_grad_(k == "fx") for k, v in X.items()}
    out = m(X, G)
    loss = ploss(out["d"].sum(-1), td.z[r], td.y[r])
    loss.backward()
    gx = float(X["fx"].grad[:, SRC, lag1].abs().sum())
    # zero-initialised output layers: the first step's gradient reaches the output layer (weight.3) and beta / pair table,
    # not yet the hidden layers behind a zero output layer (by design)
    gparams = {n: float(p.grad.abs().sum()) for n, p in m.named_parameters()
               if p.grad is not None and (n.startswith("beta") or n.startswith("pair") or n.startswith("weight.3"))}
    floor = float((d.vol <= CFG["loss"]["vol_floor"] * 1.0001)[d.rows["A"]].mean())
    zs = (d.y[:, 1:] / d.vol)[d.rows["A"]]
    mps = None
    if torch.backends.mps.is_available():
        m.eval()
        with torch.no_grad():
            a = m(*td.batch(r))["d"]
            mm = copy.deepcopy(m).to("mps")
            Xm, Gm = td.batch(r)
            b = mm({k: v.to("mps") for k, v in Xm.items()}, Gm.to("mps"))["d"].cpu()
        mps = float((a - b).abs().max())
    out = dict(lag_reach_max_abs_err=reach, grad_input_source_lag1=gx, grad_params_nonzero={k: v > 0 for k, v in gparams.items()},
               vol_floor_share_A=floor, z_sd_A=float(zs.std()), z_abs_max_A=float(np.abs(zs).max()), cpu_mps_max_abs_diff=mps,
               passed=bool(reach < 1e-6 and gx > 0 and all(v > 0 for v in gparams.values()) and (mps is None or mps < 1e-4)))
    log(json.dumps(out, default=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="D,A,B,C")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="attempt0")
    a = ap.parse_args()
    torch.set_num_threads(CFG["threads"])
    OUT.mkdir(parents=True, exist_ok=True)
    logf = (OUT / f"{a.tag}.log").open("a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    base = build(CFG["fx_input_policy"])
    path = OUT / f"{a.tag}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for part in a.only.split(","):
        t0 = time.perf_counter()
        res[part] = dict(D=check_D, A=check_A, B=check_B, C=check_C)[part](base, a.seed, log)
        res[part]["seconds"] = time.perf_counter() - t0
        path.write_text(json.dumps(res, indent=1, default=float))
    log("done")


if __name__ == "__main__":
    main()
