"""Mean-Hit-selected graphs where the edge count really varies by date (not only by regime).

Same shared b and d as graph_order (CPGG-v2, frozen or refit). Dynamic candidates here are
    cells(pair, regime) optimised on test Mean Hit *under* a per-date gate, where the gate is one of
      |d|/sd > q, selector score g > test quantile, rho_cell_EB > q, sign(d) == sign(b), or combinations.
A dynamic graph is admissible only if the edge count varies within each regime (no regime-constant graph) and is never constant over dates.
All structures/thresholds are chosen on test labels (user request) -> test numbers are in-sample.

PYTHONPATH=. python -m cpgg_v2.graph_order_hit --seed 0
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

from . import gain_records as GR
from . import pipeline as PL
from . import selection as SL
from .graph_order import Scorer, cells_to_mask, k_stats, local_search
from .selector import flat
from .test_select import prepare

OUT = ROOT / "results/cpgg_v2/graph_order"
ABSD = (0., .25, .5, .75, 1., 1.5)
QS = (.3, .5, .7, .8, .9, .95)
RHO = (-.5, 0., .25, .5)


def gates(T, tl, cal, pos):
    """Per-date, per-candidate boolean gates known at decision time. T['X'] columns: 0 rho_cell_EB, 4 |d|/sd."""
    n = len(T["d"]); b = tl["b"][pos["T"]]
    bsign = np.repeat(np.sign(b)[:, :, None], 10, 2).reshape(n, 100) if False else None
    out = {"none": np.ones((n, 100), bool)}
    for q in ABSD[1:]:
        out[f"|d|/sd>{q:g}"] = T["X"][..., 4] > q
    for r in RHO:
        out[f"rho_cell>{r:g}"] = T["X"][..., 0] > r
    # sign agreement between message and own prediction of the receiver (candidate order: receiver-major, 10 senders each)
    bb = np.repeat(b[:, :, None], 10, 2).reshape(n, 100)
    out["sign(d)=sign(b)"] = np.sign(T["d"]) == np.sign(bb)
    out["sign(d)!=sign(b)"] = np.sign(T["d"]) != np.sign(bb)
    for name, c in cal.items():
        g = c(T).reshape(n, 100)
        for q in QS:
            out[f"{name} g>q{q:g}"] = g > np.quantile(g, q)
    return out


def admissible(A, reg, R):
    K = A.sum(1)
    if K.min() == K.max() or K.min() == 0 or K.max() == 100:
        return False
    for r in range(R):
        m = reg == r
        if m.sum() > 20 and len(np.unique(K[m])) < 8:       # edge count must vary inside every regime
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    t0 = time.perf_counter(); d = build("lag1")
    log = lambda m: print(m, flush=True)
    final = {}
    for dep in ("frozen", "refit"):
        tl = GR.build(d, a.seed, test=dep, log=log)
        pos, sets, T, T_fake, real_recC = prepare(tl)
        C = flat(tl, pos["C"], False)
        cal = PL.fit_selectors(sets["std"], tl["R"], True, a.seed, ("S", "D"), log)
        R = tl["R"]; reg = tl["reg"][pos["T"]]; regC = tl["reg"][pos["C"]]
        scT, scC = Scorer(tl, pos["T"], T), Scorer(tl, pos["C"], C)
        gT, gC = gates(T, tl, cal, pos), gates(C, tl, cal, dict(T=pos["C"]))
        # C gates use thresholds from the test quantiles: recompute with test quantile values for the selector gates
        for name, c in cal.items():
            g_t = c(T).reshape(len(T["d"]), 100); g_c = c(C).reshape(len(C["d"]), 100)
            for q in QS:
                gC[f"{name} g>q{q:g}"] = g_c > np.quantile(g_t, q)
        mean_D = T["D"].mean(0)
        s_init = np.repeat((mean_D > 0)[:, None], R, 1)
        order = np.argsort(-mean_D)
        s_best, _ = local_search(s_init, lambda c: -scT.hit(cells_to_mask(c, reg)), tie=True, order=order)
        base = {"No graph": (np.zeros_like(T["D"]), np.zeros((len(C["d"]), 100))), "Dense": (np.ones_like(T["D"]), np.ones((len(C["d"]), 100))),
                "Static": (cells_to_mask(s_best, reg), cells_to_mask(s_best, regC))}
        cand = {}
        for gname in gT:
            if gname == "none":
                continue
            for start_name, start in (("static set", s_best), ("all pairs", np.ones((100, R), bool))):
                f = lambda c: -scT.hit(cells_to_mask(c, reg) * gT[gname])
                cell, v = local_search(start, f, tie=False, passes=4)
                A = cells_to_mask(cell, reg) * gT[gname]
                if admissible(A, reg, R):
                    cand[f"cells({start_name}) & {gname}"] = (A, cells_to_mask(cell, regC) * gC[gname])
        log(f"[{dep}] {len(cand)} admissible dynamic graphs {time.perf_counter() - t0:.0f}s")
        ranked = sorted(cand, key=lambda k: -scT.hit(cand[k][0]))
        best = ranked[0]
        chosen = dict(base, Dynamic=cand[best])
        Ls = {k: scT.date_loss(v[0]) for k, v in chosen.items()}
        Hs = {k: (np.sign(scT.pred(v[0])) == np.sign(scT.y)).mean(1) for k, v in chosen.items()}
        HsC = {k: (np.sign(scC.pred(v[1])) == np.sign(scC.y)).mean(1) for k, v in chosen.items()}
        comp = {f"{x} vs {y}": dict(test_hit_gain=boot_mean(Hs[x] - Hs[y]), C_hit_gain=boot_mean(HsC[x] - HsC[y]),
                                    test_loss_gain=boot_mean(Ls[y] - Ls[x]))
                for x, y in (("Static", "No graph"), ("Dynamic", "Static"), ("Dynamic", "No graph"), ("Dynamic", "Dense"))}
        K = chosen["Dynamic"][0].sum(1).astype(int)
        byreg = {int(r): dict(n=int((reg == r).sum()), K_mean=float(K[reg == r].mean()), K_min=int(K[reg == r].min()), K_max=int(K[reg == r].max()),
                              n_distinct=int(len(np.unique(K[reg == r])))) for r in range(R)}
        final[dep] = dict(dynamic=best, static_edges=int(s_best[:, 0].sum()), n_admissible=len(cand),
                          table={k: dict(test_loss=scT.loss(v[0]), test_hit=scT.hit(v[0]), C_loss=scC.loss(v[1]), C_hit=scC.hit(v[1]),
                                         test_K=k_stats(v[0])) for k, v in chosen.items()},
                          comparisons=comp, K_by_regime=byreg, K_hist_dates=np.bincount(K, minlength=101).tolist(),
                          top10=[dict(name=k, test_hit=scT.hit(cand[k][0]), test_loss=scT.loss(cand[k][0]), C_hit=scC.hit(cand[k][1]),
                                      K_mean=float(cand[k][0].sum(1).mean())) for k in ranked[:10]])
        log(f"[{dep}] dynamic = {best}")
        for k, t in final[dep]["table"].items():
            log(f"   {k:9s} test hit {t['test_hit']:.4f} loss {t['test_loss']:.5f} K {t['test_K']['K_date_mean']:.1f}±{t['test_K']['K_date_std']:.1f} "
                f"[{t['test_K']['K_date_min']},{t['test_K']['K_date_max']}] | C hit {t['C_hit']:.4f}")
        log(f"   K by regime: {byreg}")
        np.savez_compressed(OUT / f"masks_hitdyn_{dep}_s{a.seed}.npz", **{f"{k}_test": v[0] for k, v in chosen.items()}, **{f"{k}_C": v[1] for k, v in chosen.items()})
        del tl
    final["seconds"] = time.perf_counter() - t0; final["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    (OUT / f"eval_hitdyn_s{a.seed}.json").write_text(json.dumps(final, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    log(f"done {final['seconds']:.0f}s, {final['max_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
