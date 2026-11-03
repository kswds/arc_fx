"""Test-block model selection for CPGG-v2 (requested by the user: choose the configuration by test loss).

The test block (rows 4203..5251) is opened here for the first time. Every candidate below is scored on test and the
one with the lowest mean date loss is reported as the main model, so its test numbers are selection-optimistic:
no untouched hold-out remains.

Candidates = deployment x selector-training set x selector config x inference rule.
  deployment  frozen: O_5 and final messages (fit on train) applied to test.
              refit:  O_6 and messages refit on train + C (last 5 rows purged), applied to test.
  training    std:  selectors on records 3-4, isotonic on block 5, c0 = placebo 95% of blocks 3-5 (as for C).
              wide: selectors on records 3-5, isotonic on C, c0 = placebo 95% of blocks 3-5 + C.
  rules       independent at several thresholds, shrink by rho_EB, sequential greedy (S), cell rules, soft rho weights.

PYTHONPATH=. python -m cpgg_v2.test_select --seed 0
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np
import torch

from cpgg.data import ROOT, build
from cpgg.evaluate import boot_mean

from . import evaluate as EV
from . import gain_records as GR
from . import pipeline as PL
from . import selection as SL
from .selector import candidate_features, flat, stack

OUT = ROOT / "results/cpgg_v2/test_select"
FAKE_Q = (.90, .95, .975, .99)
ABS_THR = (0., 1e-4, 2e-4, 5e-4, 1e-3)


def prepare(tl):
    tl["X"], tl["rel"] = candidate_features(tl)
    tl["X_fake"], tl["rel_fake"] = candidate_features(tl, fake=True)
    f = tl["fold"]
    pos = dict(rec34=np.where(f <= 1)[0], rec5=np.where(f == 2)[0], rec=np.where(f <= 2)[0], C=np.where(f == 3)[0],
               rec5C=np.where((f >= 2) & (f <= 3))[0], recC=np.where(f <= 3)[0], rec345=np.where(f <= 2)[0], T=np.where(f == 4)[0])
    sets = dict(
        std=dict(train=stack(flat(tl, pos["rec34"], False), flat(tl, pos["rec34"], True)),
                 cal=stack(flat(tl, pos["rec5"], False), flat(tl, pos["rec5"], True)), fake_rec=stack(flat(tl, pos["rec"], True))),
        wide=dict(train=stack(flat(tl, pos["rec345"], False), flat(tl, pos["rec345"], True)),
                  cal=stack(flat(tl, pos["C"], False), flat(tl, pos["C"], True)), fake_rec=stack(flat(tl, pos["recC"], True))))
    T = flat(tl, pos["T"], False); T_fake = flat(tl, pos["T"], True)
    real_recC = stack(flat(tl, pos["recC"], False))
    return pos, sets, T, T_fake, real_recC


def rules(c, T, T_fake, n, fake_rec):
    """All inference rules of one calibrated selector on test -> {rule: mask (n, 100)}."""
    g = c(T).reshape(n, 100)
    fk = c(fake_rec).ravel()
    out = {}
    for q in FAKE_Q:
        thr = float(np.quantile(fk, q))
        out[f"placebo{q:g}"] = (g > thr).astype(float)
        out[f"max(placebo{q:g},0)"] = (g > max(thr, 0.)).astype(float)
    for a in ABS_THR:
        out[f"thr{a:g}"] = (g > a).astype(float)
    out["placebo0.95+shrink"] = SL.shrink(out["placebo0.95"], T)
    out["thr0+shrink"] = SL.shrink(out["thr0"], T)
    if c.sel.kind == "S":
        out["greedy_c0"] = SL.greedy(c, T, c.c0)
        out["greedy_0"] = SL.greedy(c, T, max(c.c0, 0.))
    fake_rate = {k: None for k in out}
    gf = c(T_fake).reshape(n, 100)
    fake_rate["placebo0.95"] = float((gf > c.c0).mean())
    return out, fake_rate


def selector_free(T, real_recC, reg_T, R):
    out = {}
    rho, se = T["X"][..., 0], T["X"][..., 3]
    for z in (0., 1., 1.645, 2.):
        for lv in (0., .25, .5, .75, 1.):
            out[f"cell: rho_EB - {z:g} se > {lv:g}"] = ((rho - z * se) > lv).astype(float)
    out["soft: clip(rho_EB, 0, 1)"] = np.clip(rho, 0, 1)
    out["soft: clip(rho_EB - 0.5, 0, 0.5)"] = np.clip(rho - .5, 0, .5)
    out["static cells (records 3-5 + C)"] = SL.static_cells(real_recC, reg_T, R)[0]
    return out


def score(tl, pos, b, d, A):
    return PL.date_loss(SL.predict(b, d, A), tl, pos["T"])


def run_deployment(d, seed, dep, log):
    t0 = time.perf_counter()
    tl = GR.build(d, seed, test=dep, log=log)
    pos, sets, T, T_fake, real_recC = prepare(tl)
    b = tl["b"][pos["T"]]; n = len(b)
    log(f"[{dep}] timeline {time.perf_counter() - t0:.0f}s, test n={n}")
    cands = {"Own-only": np.zeros((n, 100)), "Dense": np.ones((n, 100))}
    meta = {"Own-only": {}, "Dense": {}}
    for k, A in selector_free(T, real_recC, tl["reg"][pos["T"]], tl["R"]).items():
        cands[k] = A; meta[k] = {}
    for tr, data in sets.items():
        cal = PL.fit_selectors(data, tl["R"], True, seed, ("S", "D"), log)
        log(f"[{dep}/{tr}] selectors {time.perf_counter() - t0:.0f}s")
        for name, c in cal.items():
            rs, fr = rules(c, T, T_fake, n, data["fake_rec"])
            for rn, A in rs.items():
                key = f"{tr} | {name} | {rn}"
                cands[key] = A; meta[key] = dict(c0=c.c0, placebo_rate_test=fr[rn])
    L = {k: score(tl, pos, b, T["d"], A) for k, A in cands.items()}
    log(f"[{dep}] {len(cands)} candidates scored {time.perf_counter() - t0:.0f}s")
    return dict(tl=tl, pos=pos, T=T, b=b, cands=cands, meta=meta, L=L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    d = build("lag1")
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    res = {dep: run_deployment(d, a.seed, dep, log) for dep in ("frozen", "refit")}
    table = []
    for dep, r in res.items():
        for k, Lk in r["L"].items():
            A = r["cands"][k]
            table.append(dict(deployment=dep, model=k, test_loss=float(Lk.mean()),
                              gain_vs_own=float((r["L"]["Own-only"] - Lk).mean()), gain_vs_dense=float((r["L"]["Dense"] - Lk).mean()),
                              K_mean=float(SL.as3(A).sum(-1).mean()), **r["meta"][k]))
    table.sort(key=lambda x: x["test_loss"])
    best = next(t for t in table if t["model"] not in ("Own-only", "Dense"))
    r = res[best["deployment"]]
    Lm = r["L"][best["model"]]
    comp = {}
    for dep, rr in res.items():
        for base in ("Own-only", "Dense"):
            x = rr["L"][base] - Lm
            comp[f"main vs {base} ({dep})"] = {f"block{bl}": boot_mean(x, block=bl) for bl in (10, 5, 20)}
    comp["Dense vs Own-only (frozen)"] = {f"block{bl}": boot_mean(res["frozen"]["L"]["Own-only"] - res["frozen"]["L"]["Dense"], block=bl) for bl in (10, 5, 20)}
    comp["Dense vs Own-only (refit)"] = {f"block{bl}": boot_mean(res["refit"]["L"]["Own-only"] - res["refit"]["L"]["Dense"], block=bl) for bl in (10, 5, 20)}
    # secondary metrics for the main model and the baselines
    preds = {}
    for dep, rr in res.items():
        for base in ("Own-only", "Dense"):
            preds[f"{base} ({dep})"] = (SL.predict(rr["b"], rr["T"]["d"], rr["cands"][base]), rr["cands"][base])
    preds["main"] = (SL.predict(r["b"], r["T"]["d"], r["cands"][best["model"]]), r["cands"][best["model"]])
    sec = EV.secondary_all(preds, r["tl"], {"C": r["pos"]["T"]}, d)
    A = r["cands"][best["model"]]; Dm = r["tl"]["D"][r["pos"]["T"]][:, EV.CM]
    K = SL.as3(A).sum(-1)
    diag = dict(K_hist=np.bincount(np.ceil(K).ravel().astype(int), minlength=11).tolist(), K_mean=float(K.mean()),
                K0=float((K == 0).mean()), selected_gain_elementwise=float(Dm[A > 0].mean()) if (A > 0).any() else None,
                top_pairs=[dict(pair=f"{EV.CCY[EV.PAIRS[k][0] + 1]}<-{EV.CCY[EV.PAIRS[k][1]]}", rate=float((A[:, k] > 0).mean()))
                           for k in np.argsort(-(A > 0).mean(0))[:10] if (A[:, k] > 0).mean() > 0])
    ev = dict(main=best, comparisons=comp, secondary=sec, main_diag=diag, top30=table[:30], n_candidates=len(table),
              baselines=[t for t in table if t["model"] in ("Own-only", "Dense")],
              rank_by_deployment={dep: [t for t in table if t["deployment"] == dep][:10] for dep in res},
              seconds=time.perf_counter() - t0, max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
    (OUT / f"eval_s{a.seed}.json").write_text(json.dumps(ev, indent=1, default=float))
    with open(OUT / f"all_candidates_s{a.seed}.json", "w") as f:
        json.dump(table, f, indent=0, default=float)
    np.savez_compressed(OUT / f"test_predictions_s{a.seed}.npz", **{k.replace(" ", "_").replace("(", "").replace(")", ""): p for k, (p, _) in preds.items()},
                        rows=r["tl"]["rows"][r["pos"]["T"]])
    log(f"main: {best['deployment']} | {best['model']}  loss {best['test_loss']:.5f}  K {best['K_mean']:.2f}")
    for k, v in comp.items():
        v = v["block10"]; log(f"{k}: {v['mean'] * 1e3:+.2f} [{v['lo'] * 1e3:+.2f}, {v['hi'] * 1e3:+.2f}] x1e-3")
    log(f"done {ev['seconds']:.0f}s, {ev['max_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
