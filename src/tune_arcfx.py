"""Validation-only random search for ARC_FX hyperparameters.

Protocol (so that tuning cannot leak test information):
  * The data is truncated at the last validation target date, so the test period is never loaded.
  * Candidates are ranked by the mean *validation* hit ratio over several seeds (the same seeds as the final runs).
  * Trial 0 is the current configuration, evaluated with the identical procedure, as the reference.
  * Every trial is appended to <out>/trials.csv; nothing is discarded.
Only ARC_FX is trained here. Baselines are untouched.

Usage:
  python src/tune_arcfx.py --config configs/pit_main.yaml --trials 30 --seeds 21 42 456
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import training_core as tc                                    # noqa: E402
from data_pipeline import (MultiBlockSequenceDataset, collate_batch, create_splits,  # noqa: E402
                           prepare_data, set_seed)
from train import as_namespace, load_config, resolve_data_path  # noqa: E402

TRAIN_SPACE = {
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3, 1e-2],
    "dropout": [0.1, 0.25, 0.4],
    "edge_dropout": [0.0, 0.05, 0.1, 0.2],
    "small_return_quantile": [0.0, 0.2, 0.4, 0.6],
    "lambda_component": [0.0, 1e-4, 1e-3],
    "graph_reg_scale": [0.1, 1.0, 10.0],     # multiplies lambda_smooth/static/sparse/spectral
    "hit_alpha": [0.0, 0.01, 0.05],
    "batch_size": [64, 128, 256],
}
ARCH_SPACE = {"hidden": [32, 48, 64, 96], "top_k": [2, 3, 4], "graph_rank": [4, 8, 16]}   # changes the paper's architecture settings
SPLIT_TUNE = [0.75, 0.25, 0.0]     # train/val inside the truncated data == 60/20 of the full sample


def current_params(cfg: dict, tune_arch: bool) -> dict:
    t, g = cfg["training"], cfg["graph"]
    p = {"lr": t["lr"], "weight_decay": t["weight_decay"], "dropout": t["dropout"], "edge_dropout": g["edge_dropout"],
         "small_return_quantile": t["small_return_quantile"], "lambda_component": t["lambda_component"],
         "graph_reg_scale": 1.0, "hit_alpha": t["hit_alpha"], "batch_size": t["batch_size"]}
    if tune_arch:
        p.update({"hidden": t["hidden"], "top_k": g["top_k"], "graph_rank": g["graph_rank"]})
    return p


def sample_params(rng: np.random.Generator, tune_arch: bool) -> dict:
    space = dict(TRAIN_SPACE, **(ARCH_SPACE if tune_arch else {}))
    return {k: v[int(rng.integers(len(v)))] for k, v in space.items()}


def apply_params(args, params: dict, base_cfg: dict) -> None:
    for k, v in params.items():
        if k == "graph_reg_scale":
            for lam in ("lambda_smooth", "lambda_static", "lambda_sparse", "lambda_spectral"):
                setattr(args, lam, float(base_cfg["training"].get(lam, getattr(args, lam))) * v)
        else:
            setattr(args, k, v)


def fit_validation_only(prepared, args, device: str) -> dict:
    """Same training loop as training_core.train_relational_model, without any test-period evaluation."""
    dataset = MultiBlockSequenceDataset(prepared, args.lookback)
    train_ds, val_ds, _ = create_splits(dataset, args.split)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    non_usd_np = np.array([c != "USD" for c in prepared.currency_names], dtype=bool)
    non_usd_t = torch.tensor(non_usd_np, dtype=torch.bool, device=device)
    config = {"n_ccy": len(prepared.currency_names), "currency_names": prepared.currency_names, "usd_idx": 0,
              "dims": prepared.dims, "hidden": args.hidden, "top_k": args.top_k, "dropout": args.dropout,
              "graph_rank": args.graph_rank, "edge_dropout": args.edge_dropout, "spectral_bound": args.spectral_bound,
              "component_gate_type": args.component_gate_type}
    model = tc.create_relational_model("oursmain", config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    threshold = tc.compute_small_return_threshold(prepared, args.split, args.small_return_quantile)
    best_score, best_metrics, best_epoch, no_improve, prev_adj = float("inf"), None, 0, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch, target, _ in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if prev_adj is not None:
                batch["A_prev"] = prev_adj
            out = model(batch)
            loss = tc.compute_relational_loss("oursmain", out, target["y_norm"].to(device), non_usd_t, prepared, args, threshold)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            prev_adj = out["adj"].detach()
        val_loss, m = tc.evaluate_relational(model, val_loader, device, prepared, non_usd_t, args, threshold, "oursmain")
        sched.step(val_loss)
        score = tc.build_selection_score(args.selection_metric, val_loss, m, args.hit_alpha)
        if score < best_score - 1e-8:
            best_score, best_metrics, best_epoch, no_improve = score, dict(m), epoch, 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break
    best_metrics["best_epoch"] = best_epoch
    return best_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pit_main.yaml"))
    ap.add_argument("--trials", type=int, default=30, help="number of random candidates (trial 0 = current config, extra)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[21, 42, 456])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "results" / "pit_tune"))
    ap.add_argument("--tune-arch", action="store_true", help="also tune hidden/top_k/graph_rank (changes the paper's stated architecture)")
    ap.add_argument("--rng-seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(Path(a.config))
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    currencies = list(cfg["experiment"]["currencies"])
    lookback, split = int(cfg["experiment"]["lookback"]), list(cfg["experiment"]["split"])
    fx = resolve_data_path(ROOT, cfg["data"]["fx_data_path"], None)
    nonfx = resolve_data_path(ROOT, cfg["data"]["nonfx_data_path"], None)
    common = dict(start_date=cfg["data"].get("start_date"), drop_equity_ccys=cfg["data"].get("drop_equity_ccys", []))

    # last validation target date of the *full* split; everything after it is dropped before any training
    full = prepare_data(fx, nonfx, currencies, lookback, split, include_regime_onehot=True, **common)
    n = len(full.merged) - lookback
    val_end = int(n * split[0]) + int(n * split[1])
    cutoff = full.merged["Date"].iloc[val_end + lookback - 1]
    test_start = full.merged["Date"].iloc[val_end + lookback]
    del full
    prepared = prepare_data(fx, nonfx, currencies, lookback, SPLIT_TUNE, include_regime_onehot=True, end_date=str(cutoff.date()), **common)
    assert prepared.merged["Date"].max() <= cutoff < test_start
    print(f"tuning data: {prepared.merged.Date.min().date()} .. {prepared.merged.Date.max().date()} ({len(prepared.merged)} rows); test period starts {test_start.date()} and is not loaded", flush=True)

    rng = np.random.default_rng(a.rng_seed)
    candidates = [("default", current_params(cfg, a.tune_arch))] + [(f"t{i:03d}", sample_params(rng, a.tune_arch)) for i in range(1, a.trials + 1)]
    log_path = out_dir / "trials.csv"
    done = set(pd.read_csv(log_path)["trial"]) if log_path.exists() else set()

    for name, params in candidates:
        if name in done:
            continue
        t0, rows = time.time(), []
        for seed in a.seeds:
            args = as_namespace(cfg, seed, "oursmain", a.device, None, None)
            args.split = SPLIT_TUNE
            apply_params(args, params, cfg)
            set_seed(seed)
            m = fit_validation_only(prepared, args, a.device)
            rows.append(m)
        rec = {"trial": name, **params, "val_hit_mean": float(np.mean([r["hit_ratio"] for r in rows])),
               "val_hit_std": float(np.std([r["hit_ratio"] for r in rows])), "val_ic_mean": float(np.mean([r["ic"] for r in rows])),
               "val_rmse_mean": float(np.mean([r["rmse"] for r in rows])), "best_epoch_mean": float(np.mean([r["best_epoch"] for r in rows])),
               "per_seed_hit": json.dumps([round(r["hit_ratio"], 4) for r in rows]), "minutes": round((time.time() - t0) / 60, 2)}
        pd.DataFrame([rec]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        print(f"{name}: val_hit {rec['val_hit_mean']:.4f} ±{rec['val_hit_std']:.4f}  ic {rec['val_ic_mean']:.4f}  {rec['minutes']} min  {params}", flush=True)

    res = pd.read_csv(log_path)
    ref = res[res.trial == "default"].iloc[0]
    cand = res[res.trial != "default"].sort_values("val_hit_mean", ascending=False)
    best = cand.iloc[0] if len(cand) else ref
    # winner's-curse warning: spread of validation hit among candidates vs. noise level
    summary = [f"# ARC_FX tuning summary (validation only)", "",
               f"- tuning data ends {cutoff.date()}; test period (from {test_start.date()}) never loaded",
               f"- candidates: {len(cand)}, seeds per candidate: {len(a.seeds)}",
               f"- default config: val hit {ref.val_hit_mean:.4f} (seed std {ref.val_hit_std:.4f})",
               f"- best candidate `{best.trial}`: val hit {best.val_hit_mean:.4f} (seed std {best.val_hit_std:.4f})",
               f"- val hit across candidates: median {cand.val_hit_mean.median():.4f}, max {cand.val_hit_mean.max():.4f}" if len(cand) else "",
               "", "Top 5:", "", cand.head(5).drop(columns=["per_seed_hit"]).to_string(index=False) if len(cand) else ""]
    (out_dir / "tuning_summary.md").write_text("\n".join(summary), encoding="utf-8")

    keep = ["lr", "weight_decay", "dropout", "edge_dropout", "small_return_quantile", "lambda_component", "graph_reg_scale",
            "hit_alpha", "batch_size"] + (["hidden", "top_k", "graph_rank"] if a.tune_arch else [])
    bp = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in keep}
    (out_dir / "best_params.json").write_text(json.dumps({"trial": str(best.trial), **bp}, indent=2), encoding="utf-8")
    tuned = copy.deepcopy(cfg)
    tr, gr = tuned["training"], tuned["graph"]
    for k in ("lr", "weight_decay", "dropout", "small_return_quantile", "lambda_component", "hit_alpha", "batch_size", "hidden"):
        if k in bp: tr[k] = bp[k]
    for k in ("edge_dropout", "top_k", "graph_rank"):
        if k in bp: gr[k] = bp[k]
    for lam in ("lambda_smooth", "lambda_static", "lambda_sparse", "lambda_spectral"):
        tr[lam] = float(cfg["training"].get(lam)) * float(bp["graph_reg_scale"])
    tr["batch_size"], tr["hit_alpha"] = int(tr["batch_size"]), float(tr["hit_alpha"])
    tuned["outputs"] = {"prediction_root": "results/pit_tuned/runs", "report_dir": "results/pit_tuned/report"}
    yaml.safe_dump(tuned, open(ROOT / "configs" / "pit_main_tuned.yaml", "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    print("\n".join(summary[:8]))
    print("wrote configs/pit_main_tuned.yaml")


if __name__ == "__main__":
    main()
