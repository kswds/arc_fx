"""Section 8 synthetic checks for CPGG-v2 (M-v2 messages, G-loss-S / G-loss-D selectors), pass rules as in CPGG section 12 plus:
A extra: at SNR 0.1 the CHF->JPY selection rate must exceed the noise-candidate (SEK, independent N(0,1) placed in its FX lag slots)
selection rate. Main rule = calibrated gain > c0 (placebo 95th percentile), selector config chosen by C loss as on real data.

PYTHONPATH=. python -m cpgg_v2.synthetic_checks --only D,A,B,C --tag attempt0
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from cpgg.data import CCY, CFG, ROOT, build
from cpgg.evaluate import block_index, boot_mean
from cpgg.synthetic_checks import DUP, NOISE, RECV, SRC, S, make
from cpgg.train import ploss

from . import cell_reliability as CR
from . import pipeline as PL
from .message_linear import candidate_mask

OUT = ROOT / "results/cpgg_v2/synthetic"
I = RECV - 1


def cand(i, j):
    cm = candidate_mask()
    return int(np.cumsum(cm.ravel())[i * 11 + j] - 1)


def place_noise(d, seed=99):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(len(d.y))
    for k in range(5):
        sl = d.names["fx"].index(f"FX_ret_1d_lag{k}"); ms = d.names["fx"].index(f"missing_FX_ret_1d_lag{k}")
        x = np.roll(v, k); x[:k] = 0.
        d.X["fx"][:, NOISE, sl] = x; d.X["fx"][:, NOISE, ms] = 0.
    return d


def jl(tl, pos, zp):
    return ploss(torch.as_tensor(zp).float(), torch.as_tensor(tl["z"][pos, I]).float(), torch.as_tensor(tl["y"][pos, I]).float(), reduce=False).numpy()


def rates(res, reg_C=None):
    out = {}
    for kind in ("S", "D"):
        name = res["chosen"][kind]; M = res["C"][name]["M"]
        r = dict(config=name, c0=res["C"][name]["c0"], src=float(M[:, cand(I, SRC)].mean()), noise=float(M[:, cand(I, NOISE)].mean()),
                 dup=float(M[:, cand(I, DUP)].mean()), K_JPY=float(M.reshape(-1, 10, 10)[:, I].sum(-1).mean()),
                 placebo_rate_C=res["C"][name]["fake_rate"], placebo_rate_rec=res["C"][name]["fake_rate_rec"])
        if reg_C is not None:
            sel = M[:, cand(I, SRC)]; g = reg_C.astype(bool)
            ix = block_index(len(sel), 10, 10000, 77); xs, gs = sel[ix], g[ix]
            diff = (xs * gs).sum(1) / np.maximum(gs.sum(1), 1) - (xs * ~gs).sum(1) / np.maximum((~gs).sum(1), 1)
            r.update(src_regime=float(sel[g].mean()), src_non_regime=float(sel[~g].mean()), diff=float(sel[g].mean() - sel[~g].mean()),
                     lo=float(np.quantile(diff, .025)), hi=float(np.quantile(diff, .975)))
        out[kind] = r
    return out


def forced(res, sig):
    tl, pos = res["tl"], res["pos"]["C"]
    b = tl["b"][pos, I]; dd = tl["d"][pos, I, SRC]
    L0, Lf, Lo = jl(tl, pos, b), jl(tl, pos, b + dd), jl(tl, pos, b + sig[tl["rows"][pos]])
    gf, go = boot_mean(L0 - Lf), boot_mean(L0 - Lo)
    return gf, go, float(np.corrcoef(dd, sig[tl["rows"][pos]])[0, 1])


def check_A(base, seed, log):
    rows = []
    for snr in S["snr"]:
        d, sig, reg, s = make(base, snr); d = place_noise(d)
        res = PL.run(d, seed, mlp=False, log=log)
        gf, go, corr = forced(res, sig)
        ratio = gf["mean"] / go["mean"] if go["mean"] > 0 else np.nan
        sel = rates(res)
        row = dict(snr=snr, coef=d.meta["synthetic"]["coef"], forced_gain=gf, oracle_gain=go, ratio=ratio, corr_d_signal=corr, selection=sel)
        if snr in (1.0, .3):
            row["passed"] = bool(ratio >= .5 and gf["lo"] > 0)
        if snr == .1:
            row["passed"] = {k: bool(v["src"] > v["noise"]) for k, v in sel.items()}
        log(json.dumps(row, default=float)); rows.append(row)
    p_main = all(r["passed"] for r in rows if r["snr"] in (1.0, .3))
    p_extra = {k: rows[[r["snr"] for r in rows].index(.1)]["passed"][k] for k in ("S", "D")}
    return dict(rows=rows, passed_forced=p_main, passed_snr01=p_extra, passed={k: bool(p_main and p_extra[k]) for k in ("S", "D")})


def check_B(base, seed, log):
    out = {}
    for rk in ("vix", "main"):
        d, sig, reg, s = make(base, S["regime_snr"], regime=True); d = place_noise(d)
        res = PL.run(d, seed, regime_kind=rk, mlp=False, log=log)
        tl, pos = res["tl"], res["pos"]["C"]
        g = reg[tl["rows"][pos]]
        e = tl["e"][pos, I]; dd = tl["d"][pos, I, SRC]
        msg = {k: dict(mean_abs_d=float(np.abs(dd[m]).mean()), realized_rho=float((e[m] * dd[m]).sum() / (dd[m] ** 2).sum()))
               for k, m in (("regime", g), ("non_regime", ~g))}
        sel = rates(res, g)
        out[rk] = dict(message=msg, selection=sel, passed={k: bool(v["lo"] > 0) for k, v in sel.items()})
        log(json.dumps({rk: out[rk]}, default=float))
    out["passed"] = out["vix"]["passed"]
    out["note"] = "pass rule evaluated with the regime set = injected regime (VIX change > 0); 'main' = real 3-class regimes (information)"
    return out


def check_C(base, seed, log):
    d, sig, reg, s = make(base, S["redundancy_snr"], extras=True)
    res = PL.run(d, seed, mlp=False, log=log)
    sel = rates(res)
    C = res["data"]["C"]; tl, pos = res["tl"], res["pos"]["C"]
    sub = lambda k: {key: v[:, k] for key, v in C.items()}
    out = dict(selection=sel)
    name = res["chosen"]["S"]; c = res["cal"][name]
    g_src = c.raw(sub(cand(I, SRC)))
    g_dup0 = c.raw(sub(cand(I, DUP)))
    g_dup = c.raw(sub(cand(I, DUP)), cross=tl["d"][pos, I, SRC])
    Mg = res["C"][name]["Mg"]
    both = Mg[:, cand(I, SRC)] > 0
    out["S"] = dict(est_src_alone=float(g_src.mean()), est_dup_alone=float(g_dup0.mean()), est_dup_given_src=float(g_dup.mean()),
                    greedy_dup_rate_given_src=float(Mg[both, cand(I, DUP)].mean()) if both.any() else None,
                    greedy_src_rate=float(both.mean()), noise_rate=sel["S"]["noise"],
                    passed_noise=bool(sel["S"]["noise"] <= .10), passed_dup=bool(g_dup.mean() <= .5 * g_src.mean()))
    out["D"] = dict(noise_rate=sel["D"]["noise"], passed_noise=bool(sel["D"]["noise"] <= .10),
                    note="duplicate rule needs the structured conditional gain; checked for S only")
    out["passed"] = dict(S=bool(out["S"]["passed_noise"] and out["S"]["passed_dup"]), D=bool(out["D"]["passed_noise"]))
    log(json.dumps(out, default=float))
    return out


def leakage_check(n=300, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, 10)); d = rng.standard_normal((n, 10, 11)) * .1; reg = rng.integers(0, 3, n); sd = np.full((n, 10, 11), .1)
    f0 = CR.features(e, d, reg, 3, sd)
    t = 150
    e2, d2 = e.copy(), d.copy(); e2[t - 1:] = rng.standard_normal(e2[t - 1:].shape); d2[t - 1:] *= 5.
    f1 = CR.features(e2, d2, reg, 3, sd)
    keys = ("rho_cell", "rho_pair", "n_cell", "se_cell")
    return float(max(np.abs(f0[k][:t + 1] - f1[k][:t + 1]).max() for k in keys if k != "se_cell")), \
        float(np.abs(f0["se_cell"][:t + 1] - f1["se_cell"][:t + 1]).max())


def check_D(base, seed, log):
    d, sig, reg, s = make(base, 1.0)
    lag1 = d.names["fx"].index("FX_ret_1d_lag1")
    r = np.arange(100, 110)
    reach = float(np.abs(d.X["fx"][r, SRC, lag1] - s[r - 1]).max())
    res = PL.run(d, seed, mlp=False, log=log)
    tl = res["tl"]
    dec = float(np.abs(tl["part"].sum(-1) - tl["d"]).max())
    selfmax = float(max(np.abs(tl["d"][:, i, i + 1]).max() for i in range(10)))
    usd_fx = float(np.abs(tl["P"][:, 0, :8]).max())
    leak_rho, leak_se = leakage_check()
    name = res["chosen"]["S"]
    out = dict(lag_reach_max_abs_err=reach, decomposition_max_abs_err=dec, self_message_max=selfmax, usd_fx_inputs_max=usd_fx,
               cell_feature_leak_max_change=leak_rho, cell_se_leak_max_change=leak_se,
               corr_message_signal_C=forced(res, sig)[2], placebo_rate_records=res["C"][name]["fake_rate_rec"],
               cpu_mps="not applicable: M-v2 and ridge selectors run in float64 numpy on CPU")
    out["passed"] = bool(reach < 1e-6 and dec < 1e-9 and selfmax == 0 and usd_fx == 0 and leak_rho == 0 and leak_se == 0)
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
        print(msg, flush=True); logf.write(str(msg) + "\n"); logf.flush()
    base = build("lag1")
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
