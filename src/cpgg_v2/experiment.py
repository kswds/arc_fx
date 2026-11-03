"""Real-data CPGG-v2 run (C block only, test never used).

PYTHONPATH=. python -m cpgg_v2.experiment --seed 0
"""
from __future__ import annotations

import argparse
import json
import pickle
import resource
import time

import numpy as np
import torch

from cpgg.data import ROOT, build

from . import evaluate as EV
from . import gain_records as GR
from . import pipeline as PL

OUT = ROOT / "results/cpgg_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-v1", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    d = build("lag1")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split_manifest.json").write_text(json.dumps(GR.manifest(d), indent=1))
    out = OUT / "runs" / f"s{a.seed}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    log("M-v2 run"); res = PL.run(d, a.seed, mlp=True, log=log)
    res_v1 = None
    if not a.no_v1:
        log("M-v1 run (same own predictions)")
        res_v1 = PL.run(d, a.seed, message="v1", mlp=False, log=log, own=(res["tl"]["b_all"], res["tl"]["own_info"]), kinds=("S",))
    preds = EV.models(res, res_v1, a.seed)
    prim, L = EV.primary(preds, res["tl"], res["pos"])
    sec = EV.secondary_all(preds, res["tl"], res["pos"], d)
    ev = dict(primary=prim, secondary=sec,
              selector={k: EV.selector_diag(res, res["chosen_ridge"][k]) for k in ("S", "D")},
              selector_best_any={k: EV.selector_diag(res, res["chosen"][k]) for k in ("S", "D")},
              configs={n: dict(loss=r["loss"], loss_thr0=r["loss0"], c0=r["c0"], placebo_rate_C=r["fake_rate"],
                               K_mean=float(r["M"].reshape(-1, 10, 10).sum(-1).mean())) for n, r in res["C"].items()},
              messages_v2=EV.message_diag(res), messages_v1=EV.message_diag(res_v1) if res_v1 else None,
              own_info=res["tl"]["own_info"], chosen=res["chosen"], chosen_ridge=res["chosen_ridge"],
              v1_selector=EV.selector_diag(res_v1, res_v1["chosen_ridge"]["S"]) if res_v1 else None,
              v1_configs={n: dict(loss=r["loss"], K_mean=float(r["M"].reshape(-1, 10, 10).sum(-1).mean())) for n, r in res_v1["C"].items()} if res_v1 else None,
              seconds=time.perf_counter() - t0, max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
    (out / "eval.json").write_text(json.dumps(ev, indent=1, default=float))
    np.savez_compressed(out / "C_predictions.npz", **{k.replace("/", "_"): p for k, (p, _) in preds.items()},
                        rows=res["tl"]["rows"][res["pos"]["C"]])
    with open(out / "artifacts.pkl", "wb") as f:
        pickle.dump(dict(W=res["tl"]["W"], chosen_lambda=res["tl"]["chosen"], c0={n: c.c0 for n, c in res["cal"].items()},
                         masks={n: r["M"] for n, r in res["C"].items()}), f)
    for k in ("H1: Dense (M-v2) vs Own-only", "H2: main vs Dense (M-v2)", "H3: main vs Own-only"):
        v = prim[k]["block10"]; log(f"{k}: {v['mean'] * 1e3:+.2f} [{v['lo'] * 1e3:+.2f}, {v['hi'] * 1e3:+.2f}] x1e-3")
    log(f"done {ev['seconds']:.0f}s, {ev['max_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
