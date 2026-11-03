"""Resubstitution on block C (fit rows == evaluation rows; NOT an independent estimate).

ARC_FX-style recipe: directional active-set softplus loss (tau = Q0.40 |z| on A rows), checkpoint = max sign-hit on the evaluated rows
(ties: lower loss), primary metric = Mean Hit. Models: Own-only, All-currency MLP, Own + dense messages (end-to-end).

python -m cpgg.resub_c --policy lag1 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn

from . import own_model
from .data import CFG, ROOT, build
from .message_model import MessageModel
from .own_model import OwnModel, pred_loss
from .train import AllCurrency, TD

EP = 80


class OwnDense(nn.Module):
    def __init__(self, td):
        super().__init__()
        c = CFG["own_model"]; m = CFG["message_model"]
        self.own = OwnModel(td.dims, td.d.explicit, td.G.shape[1], c["hidden"], c["emb"], c["dropout"])
        self.msg = MessageModel(td.dims, td.d.explicit, td.G.shape[1], m["hidden"], m["emb"], m["enc"], m["dropout"])

    def forward(self, X, G):
        b, _ = self.own(X, G)
        return b + self.msg(X, G)["d"].sum(-1), None


def fit_eval(make, td, rows, seed):
    torch.manual_seed(seed)
    model = make()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.RandomState(seed)
    rows = np.asarray(rows)
    y, z = td.y[rows], td.z[rows]
    best, hist = None, []
    for ep in range(EP):
        model.train()
        order = rng.permutation(rows)
        for k in range(0, len(order), 32):
            ix = order[k:k + 32]
            X, G = td.batch(ix)
            zp, _ = model(X, G)
            loss = pred_loss(zp, td.z[ix], td.y[ix])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        model.eval()
        with torch.no_grad():
            X, G = td.batch(rows)
            zp, _ = model(X, G)
            hit = float((torch.sign(zp) == torch.sign(y)).float().mean())
            l = float(pred_loss(zp, z, y))
        hist.append(hit)
        if best is None or (hit, -l) > (best[0], -best[1]):
            best = (hit, l, ep + 1)
    return dict(best_hit=best[0], best_loss=best[1], best_epoch=best[2], last_hit=hist[-1], mean_hit_over_epochs=float(np.mean(hist)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lag1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    torch.set_num_threads(CFG["threads"])
    d = build(a.policy)
    zA = np.abs(d.y[d.rows["A"]][:, -10:] / d.vol[d.rows["A"]])
    own_model.LOSS_MODE.update(mode="arcfx_dir", tau=float(np.quantile(zA, 0.40)))
    td = TD(d)
    C = d.rows["C"]
    makers = {"Own-only": lambda: OwnModel(td.dims, td.d.explicit, td.G.shape[1], CFG["own_model"]["hidden"], CFG["own_model"]["emb"], CFG["own_model"]["dropout"]),
              "All-currency MLP": lambda: AllCurrency(td.dims, td.G.shape[1], 64, CFG["own_model"]["dropout"]),
              "Own + dense messages": lambda: OwnDense(td)}
    res = {}
    out = ROOT / "results/cpgg/resub_C"; out.mkdir(parents=True, exist_ok=True)
    for name, mk in makers.items():
        res[name] = []
        for s in a.seeds:
            t0 = time.perf_counter()
            r = fit_eval(mk, td, C, s); r["seed"] = s; res[name].append(r)
            print(f"{name} seed {s}: best-epoch hit {r['best_hit']:.4f} (ep {r['best_epoch']}), last-epoch hit {r['last_hit']:.4f}, "
                  f"mean over epochs {r['mean_hit_over_epochs']:.4f}  {time.perf_counter() - t0:.0f}s", flush=True)
        (out / f"resub_{a.policy}.json").write_text(json.dumps(res, indent=1))
    print("\nsummary (mean +- sd over seeds)")
    for name, rs in res.items():
        f = lambda k: f"{np.mean([r[k] for r in rs]):.4f} +- {np.std([r[k] for r in rs]):.4f}"
        print(f"{name:24s} best-epoch {f('best_hit')} | last-epoch {f('last_hit')} | epoch-mean {f('mean_hit_over_epochs')}")


if __name__ == "__main__":
    main()
