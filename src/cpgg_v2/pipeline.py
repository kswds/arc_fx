"""One CPGG-v2 run: timeline -> selector features -> selectors (train blocks 3-4, isotonic on block 5, c0 from placebo 3-5) -> C rules."""
from __future__ import annotations

import time

import numpy as np
import torch

from cpgg.train import ploss

from . import gain_records as GR
from . import selection as SL
from .selector import MLP, RIDGE, Calibrated, MLPSel, RidgeSel, candidate_features, flat, stack


def date_loss(zp, tl, pos):
    return ploss(torch.as_tensor(zp).float(), torch.as_tensor(tl["z"][pos]).float(), torch.as_tensor(tl["y"][pos]).float(),
                 reduce=False).mean(1).numpy()


def prepare(tl):
    tl["X"], tl["rel"] = candidate_features(tl)
    tl["X_fake"], tl["rel_fake"] = candidate_features(tl, fake=True)
    f = tl["fold"]
    pos = dict(rec34=np.where(f <= 1)[0], rec5=np.where(f == 2)[0], rec=np.where(f <= 2)[0], C=np.where(f == 3)[0])
    data = dict(train=stack(flat(tl, pos["rec34"], False), flat(tl, pos["rec34"], True)),
                cal=stack(flat(tl, pos["rec5"], False), flat(tl, pos["rec5"], True)),
                fake_rec=stack(flat(tl, pos["rec"], True)), real_rec=stack(flat(tl, pos["rec"], False)),
                C=flat(tl, pos["C"], False), C_fake=flat(tl, pos["C"], True), rec5=flat(tl, pos["rec5"], False))
    return pos, data


def fit_selectors(data, R, mlp=True, seed=0, kinds=("S", "D"), log=print):
    sels = {}
    for kind in kinds:
        for a in RIDGE:
            s = RidgeSel(kind, a).fit(data["train"]); sels[s.name] = s
        if mlp:
            for h, wd in MLP:
                t0 = time.perf_counter()
                s = MLPSel(kind, h, wd, data["train"]["X"].shape[-1], R, seed).fit(data["train"]); sels[s.name] = s
                log(f"    {s.name} {time.perf_counter() - t0:.0f}s")
    cal = {}
    for name, s in sels.items():
        c = Calibrated(s).fit_iso(data["cal"]).set_threshold(data["fake_rec"])
        cal[name] = c
    return cal


def apply_C(tl, pos, data, cal, seed):
    """Returns per-config dict of masks on C and the chosen configs (by C loss of the main rule; allowed by the user)."""
    C = data["C"]; b = tl["b"][pos["C"]]
    out = {}
    for name, c in cal.items():
        g = c(C).reshape(len(b), 100)
        M = (g > c.c0).astype(float)
        M0 = (g > 0).astype(float)
        r = dict(g=g, M=M, M0=M0, c0=c.c0,
                 loss=float(date_loss(SL.predict(b, C["d"], M), tl, pos["C"]).mean()),
                 loss0=float(date_loss(SL.predict(b, C["d"], M0), tl, pos["C"]).mean()),
                 fake_rate=float((c(data["C_fake"]).reshape(len(b), 100) > c.c0).mean()),
                 fake_rate_rec=float((c(data["fake_rec"]) > c.c0).mean()))
        if name.startswith("S_"):
            r["Mg"] = SL.greedy(c, C, c.c0)
        out[name] = r
    chosen = {k: min((n for n in out if n.startswith(k + "_")), key=lambda n: out[n]["loss"]) for k in ("S", "D") if any(n.startswith(k + "_") for n in out)}
    chosen_ridge = {k: min((n for n in out if n.startswith(k + "_ridge")), key=lambda n: out[n]["loss"]) for k in ("S", "D") if any(n.startswith(k + "_ridge") for n in out)}
    return out, chosen, chosen_ridge


def run(d, seed, regime_kind="main", message="v2", mlp=True, log=print, own=None, kinds=("S", "D")):
    t0 = time.perf_counter()
    tl = GR.build(d, seed, regime_kind, message, log, own=own)
    log(f"  timeline built {time.perf_counter() - t0:.0f}s")
    pos, data = prepare(tl)
    cal = fit_selectors(data, tl["R"], mlp, seed, kinds, log)
    log(f"  selectors {time.perf_counter() - t0:.0f}s")
    sc, chosen, chosen_ridge = apply_C(tl, pos, data, cal, seed)
    return dict(tl=tl, pos=pos, data=data, cal=cal, C=sc, chosen=chosen, chosen_ridge=chosen_ridge, seconds=time.perf_counter() - t0)
