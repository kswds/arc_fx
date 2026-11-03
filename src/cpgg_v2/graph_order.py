"""Test-selected graph comparison (user request: No graph < Static < Dynamic, structures may be chosen on test).

All three families share the same own predictions b and the same pairwise messages d (CPGG-v2, frozen or refit
deployment). Only the adjacency A (n dates x 100 candidates) differs:
  No graph : A = 0.
  Dense    : A = 1.
  Static   : one fixed set of (receiver, sender) pairs, identical on every date. Chosen on test.
  Dynamic  : edges switch per date from information known at the decision date (regime, selector score, |d|).
             The cell layout / thresholds are chosen on test. The number of edges varies across dates.
Structures and thresholds are fitted to test labels, so every number here is in-sample on test.
The same structures are also applied to C (before test) as a reference.

PYTHONPATH=. python -m cpgg_v2.graph_order --seed 0
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
from .selector import flat
from .test_select import prepare

OUT = ROOT / "results/cpgg_v2/graph_order"
THR_Q = (.5, .6, .7, .8, .9, .95, .975, .99)
ABSD_Q = (0., .25, .5, .75, 1., 1.5)


class Scorer:
    def __init__(self, tl, rows, data):
        self.tl, self.rows, self.b, self.d = tl, rows, tl["b"][rows], data["d"]
        self.z, self.y = tl["z"][rows], tl["y"][rows]

    def pred(self, A):
        return SL.predict(self.b, self.d, A)

    def date_loss(self, A):
        return PL.date_loss(self.pred(A), self.tl, self.rows)

    def loss(self, A):
        return float(self.date_loss(A).mean())

    def hit(self, A):
        return float((np.sign(self.pred(A)) == np.sign(self.y)).mean())

    def objective(self, A, obj):
        return self.loss(A) if obj == "loss" else -self.hit(A)


def cells_to_mask(cell, reg):
    """cell (100, R) bool -> (n, 100) mask using each date's regime."""
    return cell[:, reg].T.astype(float)


def local_search(cell0, f, tie=False, passes=6, order=None):
    """Toggle one unit at a time (a pair across all regimes if tie, else a (pair, regime) cell); keep if f decreases."""
    cell = cell0.copy(); best = f(cell)
    P, R = cell.shape
    units = [(p, None) for p in range(P)] if tie else [(p, r) for p in range(P) for r in range(R)]
    if order is not None:
        units = [units[k] for k in order]
    for _ in range(passes):
        changed = False
        for p, r in units:
            c = cell.copy()
            if tie:
                c[p, :] = ~c[p, 0]
            else:
                c[p, r] = ~c[p, r]
            v = f(c)
            if v < best - 1e-9:
                cell, best, changed = c, v, True
        if not changed:
            break
    return cell, best


def k_stats(A):
    K = SL.as3(A > 0).sum(-1); Kd = K.sum(1)
    return dict(K_date_mean=float(Kd.mean()), K_date_std=float(Kd.std()), K_date_min=int(Kd.min()), K_date_max=int(Kd.max()),
                share_dates_K0=float((Kd == 0).mean()), share_dates_K100=float((Kd == 100).mean()),
                constant_K=bool(Kd.min() == Kd.max()), K_mean=float(K.mean()), K_std=float(K.std()), K_min=int(K.min()), K_max=int(K.max()),
                K_hist=np.bincount(K.ravel(), minlength=11).tolist())


def search(sc, reg, R, cal, D_test, obj):
    """Returns dict name -> (family, description, mask builder on (data, reg, cal) for re-application to C)."""
    f_mask = lambda A: sc.objective(A, obj)
    out = {}
    # ---- static: pairs with positive mean realised gain on test, then toggle pairs by the test objective
    mean_D = D_test.mean(0)
    s_init = np.repeat((mean_D > 0)[:, None], R, 1)
    out["Static: test mean D > 0"] = ("static", dict(cell=s_init))
    order = np.argsort(-mean_D)
    s_best, _ = local_search(s_init, lambda c: f_mask(cells_to_mask(c, reg)), tie=True, order=order)
    out["Static: test-optimised pair set"] = ("static", dict(cell=s_best))
    # ---- dynamic 1: (pair, regime) cells, starting from the best static set
    cr = np.zeros((100, R), bool)
    for r in range(R):
        m = reg == r
        if m.any():
            cr[:, r] = D_test[m].mean(0) > 0
    out["Dynamic: (pair, regime) cells, test mean D > 0"] = ("dynamic", dict(cell=cr))
    d_best, _ = local_search(s_best, lambda c: f_mask(cells_to_mask(c, reg)), tie=False)
    out["Dynamic: (pair, regime) cells, test-optimised"] = ("dynamic", dict(cell=d_best))
    # ---- dynamic 2: optimised cells gated by |d| / sd (message size known at t)
    for base_name, base in (("static set", s_best), ("regime cells", d_best)):
        for q in ABSD_Q:
            out[f"Dynamic: {base_name} x |d|/sd > {q:g}"] = ("dynamic", dict(cell=base, absd=q))
    # ---- dynamic 3: selector score gates (calibrated G-loss), alone or on top of the cell sets
    for name, c in cal.items():
        for q in THR_Q:
            out[f"Dynamic: {name} g > test q{q:g}"] = ("dynamic", dict(sel=name, q=q))
            out[f"Dynamic: static set & {name} g > test q{q:g}"] = ("dynamic", dict(cell=s_best, sel=name, q=q))
            out[f"Dynamic: regime cells & {name} g > test q{q:g}"] = ("dynamic", dict(cell=d_best, sel=name, q=q))
            out[f"Dynamic: static set | {name} g > test q{q:g}"] = ("dynamic", dict(cell=s_best, sel=name, q=q, union=True))
    return out


def realise(spec, data, reg, cal, thr_cache):
    """Turn a spec into a mask for the dates of data (test or C). Selector thresholds are the test quantiles."""
    n = len(data["d"])
    A = np.ones((n, 100))
    if "cell" in spec:
        A = cells_to_mask(spec["cell"], reg)
    if "absd" in spec:
        A = A * (data["X"][..., 4] > spec["absd"])
    if "sel" in spec:
        key = (spec["sel"], id(data))
        if key not in thr_cache["memo"]:
            thr_cache["memo"][key] = thr_cache["g"][spec["sel"]](data)
        g = thr_cache["memo"][key]
        G = (g > thr_cache["thr"][(spec["sel"], spec["q"])]).astype(float)
        A = np.maximum(A, G) if spec.get("union") else (A * G if "cell" in spec else G)
    return A


def run_deployment(d, seed, dep, log):
    t0 = time.perf_counter()
    tl = GR.build(d, seed, test=dep, log=log)
    pos, sets, T, T_fake, real_recC = prepare(tl)
    C = flat(tl, pos["C"], False)
    cal = PL.fit_selectors(sets["std"], tl["R"], True, seed, ("S", "D"), log)
    log(f"[{dep}] timeline + selectors {time.perf_counter() - t0:.0f}s")
    R = tl["R"]; reg_T = tl["reg"][pos["T"]]; reg_C = tl["reg"][pos["C"]]
    sc_T, sc_C = Scorer(tl, pos["T"], T), Scorer(tl, pos["C"], C)
    gfun = {name: (lambda data, c=c: c(data).reshape(len(data["d"]), 100)) for name, c in cal.items()}
    gT = {name: f(T) for name, f in gfun.items()}
    thr = {(name, q): float(np.quantile(gT[name], q)) for name in cal for q in THR_Q}
    cache = dict(g=gfun, thr=thr, memo={(n, id(T)): v for n, v in gT.items()})
    res = {}
    for obj in ("loss", "hit"):
        specs = search(sc_T, reg_T, R, cal, T["D"], obj)
        rows = {}
        for name, (fam, spec) in specs.items():
            A = realise(spec, T, reg_T, cal, cache)
            rows[name] = dict(family=fam, spec=spec, A=A)
        log(f"[{dep}/{obj}] {len(rows)} graphs {time.perf_counter() - t0:.0f}s")
        res[obj] = rows
    return dict(tl=tl, pos=pos, T=T, C=C, cal=cal, cache=cache, sc_T=sc_T, sc_C=sc_C, reg_T=reg_T, reg_C=reg_C, res=res)


def summary_row(r, A_T, A_C):
    return dict(test_loss=r["sc_T"].loss(A_T), test_hit=r["sc_T"].hit(A_T), C_loss=r["sc_C"].loss(A_C), C_hit=r["sc_C"].hit(A_C),
                test_K=k_stats(A_T), C_K=k_stats(A_C))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    d = build("lag1")
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    final = {}
    for dep in ("frozen", "refit"):
        r = run_deployment(d, a.seed, dep, log)
        nT, nC = len(r["T"]["d"]), len(r["C"]["d"])
        base = {"No graph": (np.zeros((nT, 100)), np.zeros((nC, 100))), "Dense": (np.ones((nT, 100)), np.ones((nC, 100)))}
        for obj, rows in r["res"].items():
            table = {k: dict(family=("none" if k == "No graph" else "dense"), **summary_row(r, *v)) for k, v in base.items()}
            masks = {k: v for k, v in base.items()}
            for name, row in rows.items():
                A_T = row["A"]; A_C = realise(row["spec"], r["C"], r["reg_C"], r["cal"], r["cache"])
                table[name] = dict(family=row["family"], **summary_row(r, A_T, A_C)); masks[name] = (A_T, A_C)
            key = "test_loss" if obj == "loss" else "test_hit"
            better = (lambda x, y: x < y) if obj == "loss" else (lambda x, y: x > y)
            ok = lambda v: v["family"] != "dynamic" or not (v["test_K"]["constant_K"] or v["test_K"]["share_dates_K0"] == 1 or v["test_K"]["share_dates_K100"] == 1)
            pick = lambda fam: (min if obj == "loss" else max)((k for k, v in table.items() if v["family"] == fam and ok(v)), key=lambda k: table[k][key])
            best_static, best_dyn = pick("static"), pick("dynamic")
            chosen = {"No graph": "No graph", "Dense": "Dense", "Static": best_static, "Dynamic": best_dyn}
            Ls = {lab: r["sc_T"].date_loss(masks[k][0]) for lab, k in chosen.items()}
            LsC = {lab: r["sc_C"].date_loss(masks[k][1]) for lab, k in chosen.items()}
            hits = lambda sc, A: (np.sign(sc.pred(A)) == np.sign(sc.y)).mean(1)
            Hs = {lab: hits(r["sc_T"], masks[k][0]) for lab, k in chosen.items()}
            comp = {}
            for x, y in (("Static", "No graph"), ("Dynamic", "Static"), ("Dynamic", "No graph"), ("Dense", "No graph"), ("Dynamic", "Dense")):
                comp[f"{x} vs {y}"] = dict(test_loss_gain=boot_mean(Ls[y] - Ls[x]), test_hit_gain=boot_mean(Hs[x] - Hs[y]),
                                          C_loss_gain=boot_mean(LsC[y] - LsC[x]))
            order_ok = better(table[best_static][key], table["No graph"][key]) and better(table[best_dyn][key], table[best_static][key])
            final[f"{dep}/{obj}"] = dict(chosen=chosen, order_holds=bool(order_ok), comparisons=comp,
                                         chosen_rows={lab: table[k] for lab, k in chosen.items()},
                                         static_edges=int(r["res"][obj][best_static]["spec"]["cell"][:, 0].sum()) if best_static in r["res"][obj] else None,
                                         top_dynamic=sorted(({"name": k, **{m: table[k][m] for m in ("test_loss", "test_hit", "C_loss", "C_hit")}, "K_mean": table[k]["test_K"]["K_mean"]}
                                                             for k in table if table[k]["family"] == "dynamic"), key=lambda t: t[key], reverse=obj == "hit")[:10],
                                         n_graphs=len(table))
            log(f"[{dep}/{obj}] static={best_static} | dynamic={best_dyn} | order {order_ok}")
            for lab, k in chosen.items():
                t = table[k]; log(f"   {lab:9s} test loss {t['test_loss']:.5f} hit {t['test_hit']:.4f} K {t['test_K']['K_mean']:.2f}±{t['test_K']['K_std']:.2f} | C loss {t['C_loss']:.5f} hit {t['C_hit']:.4f}")
            np.savez_compressed(OUT / f"masks_{dep}_{obj}_s{a.seed}.npz", **{f"{lab}_test": masks[k][0] for lab, k in chosen.items()},
                                **{f"{lab}_C": masks[k][1] for lab, k in chosen.items()})
        del r
    final["seconds"] = time.perf_counter() - t0
    final["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    (OUT / f"eval_s{a.seed}.json").write_text(json.dumps(final, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    log(f"done {final['seconds']:.0f}s, {final['max_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
