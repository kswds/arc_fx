"""CPGG pipeline: A (own OOF + messages) -> B (gain records, selectors) -> C (graphs, predictions). Test is never touched.

python -m cpgg.train --seed 0 [--policy lag1|same_day] [--out results/cpgg/runs/...]
"""
from __future__ import annotations

import argparse
import copy
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import gain_dataset as GD
from .data import CFG, CHANNELS, ROOT, build, split_manifest
from .gain_model import Selector
from .message_model import MessageModel, candidate_mask
from .own_model import OwnModel, mlp, pred_loss
from .selection import greedy, topk_by_magnitude, predict as sel_predict
from .subset_sampling import sample_masks

LC = CFG["loss"]
LOSS = dict(huber_delta=LC["huber_delta"], eta=LC["eta"], direction_temperature=LC["direction_temperature"])


def ploss(zp, z, y, reduce=True):
    return pred_loss(zp, z, y, LC["huber_delta"], LC["eta"], LC["direction_temperature"], reduce)


class TD:
    """Torch view of a cpgg.data.Data (CPU)."""

    def __init__(self, d):
        self.d = d
        self.X = {c: torch.from_numpy(d.X[c]) for c in CHANNELS}
        self.G = torch.from_numpy(d.G)
        self.y = torch.from_numpy(d.y[:, 1:]).float()
        self.vol = torch.from_numpy(d.vol).float()
        self.z = self.y / self.vol
        self.dims = {c: d.X[c].shape[-1] for c in CHANNELS}

    def batch(self, rows):
        rows = torch.as_tensor(rows)
        return {c: self.X[c][rows] for c in CHANNELS}, self.G[rows]


def split_inner(rows, frac):
    k = int(round(len(rows) * (1 - frac)))
    return rows[:k], rows[k:]


def _train_loop(model, td, rows, step_loss, cfg, seed):
    tr, va = split_inner(np.asarray(rows), cfg["inner_val_fraction"])
    pair = [p for n, p in model.named_parameters() if n.startswith("pair.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("pair.")]
    groups = [dict(params=rest, lr=cfg["lr"], weight_decay=cfg["weight_decay"])]
    if pair:
        groups.append(dict(params=pair, lr=cfg["lr"] * cfg.get("pair_lr_mult", 1.), weight_decay=cfg.get("pair_weight_decay", cfg["weight_decay"])))
    opt = torch.optim.AdamW(groups)
    rng = np.random.RandomState(seed)
    best, state, stale, hist = np.inf, None, 0, []
    for ep in range(cfg["max_epochs"]):
        model.train()
        order = rng.permutation(tr)
        for k in range(0, len(order), cfg["batch_size"]):
            ix = order[k:k + cfg["batch_size"]]
            loss = step_loss(model, ix, train=True)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(step_loss(model, va, train=False))
        hist.append(vl)
        if vl < best - 1e-7:
            best, state, stale = vl, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(state)
    return dict(best_inner_val=best, epochs=len(hist), best_epoch=int(np.argmin(hist)) + 1)


# ----------------------------------------------------------------------------- own predictor
def fit_own(td, rows, seed, cfg=None):
    cfg = cfg or CFG["own_model"]
    torch.manual_seed(seed)
    model = OwnModel(td.dims, td.d.explicit, td.G.shape[1], cfg["hidden"], cfg["emb"], cfg["dropout"])

    def step(m, ix, train):
        X, G = td.batch(ix)
        zp, _ = m(X, G)
        return ploss(zp, td.z[ix], td.y[ix])
    info = _train_loop(model, td, rows, step, cfg, seed)
    return model, info


@torch.no_grad()
def predict_own(models, td, rows):
    zs, cs = [], []
    for m in models:
        m.eval()
        X, G = td.batch(rows)
        z, c = m(X, G)
        zs.append(z); cs.append(c)
    return torch.stack(zs).mean(0), torch.stack(cs).mean(0)


# ----------------------------------------------------------------------------- messages
def fit_message(td, rows, b_rows, seed, cfg=None, **kw):
    """b_rows: own predictions (vol units) aligned with ``rows`` (frozen); masks sampled per step."""
    cfg = cfg or CFG["message_model"]
    torch.manual_seed(seed)
    model = MessageModel(td.dims, td.d.explicit, td.G.shape[1], cfg["hidden"], cfg["emb"], cfg["enc"], cfg["dropout"], **kw)
    pos = {int(r): k for k, r in enumerate(rows)}
    gen = torch.Generator().manual_seed(seed + 101)
    mix = tuple(cfg["mask_mix"][k] for k in ("singleton", "full", "random"))
    tr, va = split_inner(np.asarray(rows), cfg["inner_val_fraction"])
    val_masks = sample_masks(len(va), torch.Generator().manual_seed(seed + 202), mix)

    def step(m, ix, train):
        X, G = td.batch(ix)
        out = m(X, G)
        b = b_rows[[pos[int(r)] for r in ix]]
        if not train and cfg.get("checkpoint", "selectable_gain") == "selectable_gain":
            # fix attempt 2: checkpoint = minus the selectable gain on the inner validation rows,
            # sum over (receiver, sender) of max(0, mean single-add loss gain at the empty set)
            z, y = td.z[ix].unsqueeze(-1).expand(-1, -1, 11), td.y[ix].unsqueeze(-1).expand(-1, -1, 11)
            L0 = ploss(b.unsqueeze(-1).expand(-1, -1, 11), z, y, reduce=False)
            L1 = ploss(b.unsqueeze(-1) + out["d"], z, y, reduce=False)
            gain = (L0 - L1).mean(0)[candidate_mask()]
            return -gain.clamp_min(0).sum()
        M = sample_masks(len(ix), gen, mix) if train else val_masks
        zp = b + (out["d"] * M).sum(-1)
        return ploss(zp, td.z[ix], td.y[ix])
    info = _train_loop(model, td, rows, step, cfg, seed)
    return model, info


@torch.no_grad()
def predict_message(model, td, rows, chunk=256):
    model.eval()
    acc = {k: [] for k in ("d", "part", "a", "delta")}
    for s in range(0, len(rows), chunk):
        X, G = td.batch(rows[s:s + chunk])
        out = model(X, G)
        for k in acc:
            acc[k].append(out[k])
    return {k: torch.cat(v) for k, v in acc.items()}


# ----------------------------------------------------------------------------- all-currency dense baseline
class AllCurrency(nn.Module):
    def __init__(self, dims, n_global, hidden=64, dropout=.1):
        super().__init__()
        self.net = mlp(11 * sum(dims.values()) + n_global, hidden, 10, dropout)

    def forward(self, X, G):
        x = torch.cat([X[c].flatten(1) for c in CHANNELS] + [G], -1)
        return self.net(x), None


def fit_allccy(td, rows, seed):
    cfg = dict(CFG["own_model"], hidden=64)
    torch.manual_seed(seed)
    model = AllCurrency(td.dims, td.G.shape[1], 64, cfg["dropout"])

    def step(m, ix, train):
        X, G = td.batch(ix)
        zp, _ = m(X, G)
        return ploss(zp, td.z[ix], td.y[ix])
    return model, _train_loop(model, td, rows, step, cfg, seed)


# ----------------------------------------------------------------------------- selectors
def fit_selectors(rec, n_dates_rows, seed, kinds=("loss", "rel"), grid=None):
    """Fit the pre-defined grid (<= 6 configs per selector kind). Inner validation for MLPs = last 20% of B dates."""
    grid = grid or CFG["selector_grid"]
    cut = int(n_dates_rows * .8)
    val_mask = rec["n"] >= cut
    out = {}
    for kind in kinds:
        for p in grid["ridge"]:
            out[f"{kind}_ridge_a{p['alpha']:g}"] = Selector(kind, "ridge", p, seed).fit(rec)
        for p in grid["mlp"]:
            out[f"{kind}_mlp_h{p['hidden']}_wd{p['weight_decay']:g}"] = Selector(kind, "mlp", p, seed).fit(rec, val_mask, grid["mlp_train"])
    return out


def subset_rec(rec, keep):
    return {k: v[keep] for k, v in rec.items()}


# ----------------------------------------------------------------------------- pipeline
def run(d, seed, out=None, full_baselines=True, aux=True, kinds=("loss", "rel"), log=print, message_kw=None,
        own_override=None):
    """Runs A -> B -> C for one seed on Data ``d`` (possibly synthetic). Returns a dict of artifacts (tensors)."""
    t0 = time.perf_counter()
    td = TD(d)
    R = d.rows
    A_oof, B, C = R["A_oof"], R["B"], R["C"]
    timing = {}
    # A: out-of-fold own predictions (expanding folds), fold models are kept for B and C
    own_models, own_info, oof_z, oof_c = [], [], [], []
    for f, fold in enumerate(R["folds"]):
        m, info = fit_own(td, fold["fit"], seed * 100 + f)
        own_models.append(m); own_info.append(info)
        z, c = predict_own([m], td, fold["predict"])
        oof_z.append(z); oof_c.append(c)
    oof_z = torch.cat(oof_z)
    timing["own_oof"] = time.perf_counter() - t0
    if own_override is not None:                                       # 6.1 alternative (ablation)
        own_models, oof_z = own_override(td, seed)
    # A: messages on OOF residuals (own predictor frozen)
    msg, msg_info = fit_message(td, A_oof, oof_z, seed * 100 + 50, **(message_kw or {}))
    timing["message"] = time.perf_counter() - t0
    log(f"[seed {seed}] own folds {own_info} | message {msg_info} | {timing['message']:.0f}s")

    def context(rows):
        b, comps = predict_own(own_models, td, rows)
        outm = predict_message(msg, td, rows)
        X, G = td.batch(rows)
        return GD.Ctx(b, outm, G, td.vol[torch.as_tensor(rows)], X, td.z[rows], td.y[rows], rows), comps

    XA, GA = td.batch(A_oof)
    ctxA = GD.Ctx(oof_z, predict_message(msg, td, A_oof), GA, td.vol[A_oof], XA, td.z[A_oof], td.y[A_oof], A_oof)
    ctxB, compsB = context(B)
    ctxC, compsC = context(C)
    # B: out-of-sample gain records
    gen = torch.Generator().manual_seed(seed + 7)
    rec = GD.build(ctxB, LOSS, CFG["gain_dataset"]["subsets_per_date_receiver"], gen)
    sels = fit_selectors(rec, len(B), seed, kinds)
    timing["selectors"] = time.perf_counter() - t0
    # C: graphs for every selector config
    graphs = {name: greedy(ctxC, [s]) for name, s in sels.items()}
    lossC = {name: float(ploss(sel_predict(ctxC, g["M"]), ctxC.z, ctxC.y)) for name, g in graphs.items()}
    chosen = {k: min((n for n in sels if n.startswith(k + "_")), key=lossC.get) for k in kinds}
    main_kind = min(kinds, key=lambda k: lossC[chosen[k]])
    log(f"[seed {seed}] C loss per selector config: {json.dumps({k: round(v, 6) for k, v in lossC.items()})}")
    res = dict(seed=seed, policy=d.policy, td=td, ctx=dict(A=ctxA, B=ctxB, C=ctxC), comps=dict(B=compsB, C=compsC),
               selectors=sels, rec=rec, graphs=graphs, lossC_by_config=lossC, chosen=chosen, main_kind=main_kind,
               own_models=own_models, msg=msg, info=dict(own=own_info, message=msg_info))
    # aux rule: same chosen config, three time blocks of B
    if aux:
        res["aux"] = {}
        for kind in kinds:
            name = chosen[kind]
            blocks = np.array_split(np.arange(len(B)), 3)
            subs = []
            for bl in blocks:
                keep = (rec["n"] >= bl[0]) & (rec["n"] <= bl[-1])
                r = subset_rec(rec, keep)
                r["n"] = r["n"] - bl[0]
                s = sels[name]
                subs.append(Selector(s.kind, s.model, s.params, seed).fit(r, r["n"] >= int(len(bl) * .8), CFG["selector_grid"]["mlp_train"]))
            res["aux"][kind] = greedy(ctxC, subs, rule="aux")
    # static graph: edges with positive mean realised single-add gain on B
    DB = GD.singles(ctxB, LOSS)
    res["static_edges"] = torch.nan_to_num(DB.mean(0), nan=-1.) > 0
    res["singles"] = dict(B=DB, C=GD.singles(ctxC, LOSS))
    # top-K by message magnitude
    res["topk"] = {k: topk_by_magnitude(ctxC, k) for k in CFG["topk_grid"]}
    if full_baselines:
        AB = np.concatenate([R["A"], B])
        own_full, i1 = fit_own(td, AB, seed * 100 + 90)
        allc, i2 = fit_allccy(td, AB, seed * 100 + 91)
        with torch.no_grad():
            X, G = td.batch(C)
            res["full"] = dict(own=own_full(X, G)[0], allccy=allc(X, G)[0], info=dict(own=i1, allccy=i2))
    timing["total"] = time.perf_counter() - t0
    res["timing"] = timing
    res["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    if out is not None:
        save(res, Path(out))
    return res


def _xg(td, rows):
    X, G = td.batch(rows)
    return X, G


def save(res, out):
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(own=[m.state_dict() for m in res["own_models"]], message=res["msg"].state_dict()), out / "predictors.pt")
    torch.save({k: v for k, v in res["selectors"].items()}, out / "selectors.pt")
    torch.save(dict(rec={k: v for k, v in res["rec"].items()}, provenance=dict(
        rows="B", own="mean of the A expanding-fold own models (never trained on B)", message="trained on A OOF residuals",
        subsets="empty + random size 1..9 per (date, receiver)")), out / "gain_records_B.pt")
    c = res["ctx"]["C"]
    keep = dict(b=c.b, d=c.d, part=c.part, a=c.a, comps=res["comps"]["C"], rows=torch.as_tensor(c.rows),
                graphs={k: dict(M=g["M"], order=g["order"], est=g["est"], first_est=g["first_est"]) for k, g in res["graphs"].items()},
                aux={k: dict(M=g["M"], order=g["order"], est=g["est"]) for k, g in res.get("aux", {}).items()},
                static=res["static_edges"], topk=res["topk"], full=res.get("full"))
    torch.save(keep, out / "C_outputs.pt")
    (out / "run.json").write_text(json.dumps(dict(seed=res["seed"], policy=res["policy"], chosen=res["chosen"], main_kind=res["main_kind"],
                                                  lossC_by_config=res["lossC_by_config"], info=res["info"], timing=res["timing"],
                                                  max_rss_mb=res["max_rss_mb"]), indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default=CFG["fx_input_policy"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(CFG["threads"])
    d = build(a.policy)
    out = Path(a.out or ROOT / f"results/cpgg/runs/{a.policy}/s{a.seed}")
    (ROOT / "results/cpgg").mkdir(parents=True, exist_ok=True)
    man = ROOT / "results/cpgg/split_manifest.json"
    if not man.exists():
        man.write_text(json.dumps(split_manifest(d), indent=1))
    run(d, a.seed, out)


if __name__ == "__main__":
    main()
