"""Real-data run + evaluation for one seed and FX policy (C block only; test never used).

python -m cpgg.experiment --seed 0 --policy lag1
Writes results/cpgg/runs/<policy>/s<seed>/{predictors.pt, selectors.pt, gain_records_B.pt, C_outputs.pt, run.json, eval.json,
C_predictions.csv.gz, C_topology.csv.gz, calibration_<kind>.png}
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import CCY, CFG, ROOT, build, split_manifest
from .evaluate import evaluate_run, model_predictions
from .train import run


def tables(res, out):
    c = res["ctx"]["C"]
    dates = pd.to_datetime(res["td"].d.dates[c.rows]).strftime("%Y-%m-%d")
    P, _ = model_predictions(res)
    vol = torch.exp(c.logvol).numpy()
    rows = []
    for name, (p, M) in P.items():
        f = pd.DataFrame(dict(date=np.repeat(dates, 10), currency=np.tile(CCY[1:], len(dates)), model=name,
                              pred_return=(p.numpy() * vol).ravel(), pred_z=p.numpy().ravel(), target=c.y.numpy().ravel(),
                              K=M.sum(-1).numpy().ravel()))
        rows.append(f)
    pd.concat(rows).to_csv(out / "C_predictions.csv.gz", index=False)
    g = res["graphs"][res["chosen"][res["main_kind"]]]
    n, i, j = g["M"].nonzero(as_tuple=True)
    step = (g["order"][n, i] == j.unsqueeze(-1)).float().argmax(-1)
    topo = pd.DataFrame(dict(date=dates[n.numpy()], receiver=[CCY[k + 1] for k in i.tolist()], sender=[CCY[k] for k in j.tolist()],
                             step=(step + 1).numpy(), est_gain=g["est"][n, i, step].numpy(), a=c.a[n, i, j].numpy(),
                             d=c.d[n, i, j].numpy(), **{f"part_{ch}": c.part[n, i, j, k].numpy() for k, ch in enumerate(["fx", "rate", "equity", "macro"])}))
    topo.to_csv(out / "C_topology.csv.gz", index=False)


def figures(ev, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for k, s in ev["selectors"].items():
        dec = pd.DataFrame(s["calibration"]["deciles"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.errorbar(dec.est_mean, dec.realized_mean, yerr=[dec.realized_mean - dec.lo, dec.hi - dec.realized_mean], fmt="o-", capsize=3)
        lim = [min(dec.est_mean.min(), dec.lo.min()), max(dec.est_mean.max(), dec.hi.max())]
        ax.plot(lim, lim, "--", color="grey", lw=1); ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
        ax.set_xlabel("estimated gain (decile mean)"); ax.set_ylabel("realised gain (C, 10-day block CI)")
        ax.set_title(f"{'G-loss' if k == 'loss' else 'G-rel'} calibration ({s['config']})", fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"calibration_{k}.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default=CFG["fx_input_policy"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(CFG["threads"])
    t0 = time.perf_counter()
    d = build(a.policy)
    mode = os.environ.get("CPGG_LOSS_MODE", "huber_bce")
    if mode == "arcfx_dir":
        from . import own_model
        zA = np.abs(d.y[d.rows["A"]][:, -10:] / d.vol[d.rows["A"]])
        own_model.LOSS_MODE.update(mode=mode, tau=float(np.quantile(zA, 0.40)))
        print("loss mode arcfx_dir, tau(Q0.40 |z|, A rows) =", own_model.LOSS_MODE["tau"])
    base = ROOT / "results/cpgg"
    base.mkdir(parents=True, exist_ok=True)
    if not (base / "split_manifest.json").exists():
        (base / "split_manifest.json").write_text(json.dumps(split_manifest(d), indent=1))
    out = base / "runs" / f"{a.policy}{a.tag}" / f"s{a.seed}"
    res = run(d, a.seed, out)
    ev = evaluate_run(res)
    ev["seconds_total"] = time.perf_counter() - t0
    ev["max_rss_mb"] = res["max_rss_mb"]
    (out / "eval.json").write_text(json.dumps(ev, indent=1, default=float))
    tables(res, out)
    figures(ev, out)
    p = ev["primary"]
    print(json.dumps({k: v["block10"] for k, v in p.items()}, indent=1))


if __name__ == "__main__":
    main()
