"""CPGG data: channel arrays with explicit lags, availability flags, time splits.

Row t is a decision date; the target is the noon-to-noon log return of row t (H.10 noon t -> next H.10 date).
FX input policy (config ``fx_input_policy``):
  lag1     (primary)     FX-derived inputs use rows <= t-1: the 12:00 rate of day t is not an input.
  same_day (sensitivity) FX-derived inputs use row t (noon rate observed through an assumed real-time feed).
Non-FX features are taken from row t as supplied (as-of 11:30 New York, see data/README.md).
Scalers use the A rows only; the regime thresholds come from the legacy pipeline (first 60% of rows = train block).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import data_pipeline as dp

ROOT = Path(__file__).resolve().parents[2]
CCY = ["USD", "EUR", "JPY", "GBP", "CAD", "AUD", "KRW", "CHF", "NZD", "SEK", "NOK"]
CFG = json.loads((ROOT / "configs/cpgg.json").read_text())
CHANNELS = ["fx", "rate", "equity", "macro"]

# (slow features, fast features with lags, single-lag fast features, flag columns); {c} = currency
SCHEMA = {
    "fx": dict(slow=["FX_mom_5", "FX_mom_20", "FX_vol_5", "FX_vol_20", "FX_zscore_20"], fast=["FX_ret_1d"], fast1=[], flags=[]),
    "rate": dict(slow=["Yield10Y", "Yield10Y_minus_US10Y", "RealRate10Y"], fast=["Yield10Y_change"], fast1=[], flags=["Yield10Y_Available"]),
    "equity": dict(slow=["Stock_mom_5", "Stock_mom_20", "Stock_vol_20"], fast=["Stock_ret"], fast1=["Stock_ret_minus_SP500_ret"],
                   flags=["Stock_Available"]),
    "macro": dict(slow=["GDP_growth", "CPIInfl", "GDP_growth_minus_USD", "CPIInfl_minus_USD"], fast=[], fast1=["macro_changed"], flags=[]),
}
FX_COLS = {"FX_ret_1d", "FX_mom_5", "FX_mom_20", "FX_vol_5", "FX_vol_20", "FX_zscore_20"}
GLOBAL = ["Global_VIX", "Global_VIX_change", "Global_DXY_ret", "Global_SP500_ret", "Global_Oil_ret", "Global_Gold_ret",
          "Global_Copper_ret", "Global_US2Y_change", "Global_US10Y_change", "Global_US10Y_minus_US2Y",
          "Global_US1Y", "Global_US2Y", "Global_US5Y", "Global_US10Y",
          "Regime_LowVol", "Regime_MidVol", "Regime_HighVol", "Regime_RiskOff", "Regime_DollarShock", "Regime_CommodityShock"]
FX_REGIMES = {"Regime_LowVol", "Regime_MidVol", "Regime_HighVol"}          # built from FX_ret_1d of row t
REGIME_COLS = [g for g in GLOBAL if g.startswith("Regime_")] + ["Global_VIX", "Global_VIX_change"]


@dataclass
class Data:
    merged: pd.DataFrame
    policy: str
    X: dict                     # channel -> (T, 11, F) float32, layout [slow | fast lags | single fast | flags | missing flags]
    explicit: dict              # channel -> indices of the explicit (linear) path features inside X[channel]
    names: dict                 # channel -> feature names
    G: np.ndarray               # (T, n_global) standardised global inputs
    y: np.ndarray               # (T, 11) raw log returns, USD column = 0
    vol: np.ndarray             # (T, 10) past FX volatility (policy-consistent), floor applied
    rows: dict                  # A, B, C, test, train, A_oof_folds
    dates: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def z(self):
        return self.y[:, 1:] / self.vol


def _num(m, col):
    return pd.to_numeric(m[col], errors="coerce") if col in m.columns else pd.Series(np.nan, index=m.index)


def splits(n_rows):
    n = n_rows - 10
    nt, nv = int(n * .6), int(n * .2)
    train = np.arange(10, 10 + nt)
    C = np.arange(10 + nt, 10 + nt + nv)
    test = np.arange(10 + nt + nv, n_rows)
    nA = int(len(train) * CFG["split"]["A_fraction_of_train"])
    A, B = train[:nA], train[nA:]
    w = int(len(A) * CFG["split"]["oof"]["warmup_fraction_of_A"])
    k = CFG["split"]["oof"]["n_folds"]
    edges = np.linspace(w, len(A), k + 1).astype(int)
    folds = [dict(fit=A[:edges[f]], predict=A[edges[f]:edges[f + 1]]) for f in range(k)]
    return dict(train=train, A=A, B=B, C=C, test=test, A_oof=A[w:], folds=folds)


def load_merged():
    p = dp.prepare_data(str(ROOT / "data/processed/fx_panel_pit.csv"), str(ROOT / "data/processed/nonfx_features_pit.csv"),
                        CCY, 10, [.6, .2, .2], True)
    return p.merged.copy(), p.y_raw.astype(np.float64)


def build(policy=None, merged=None, y=None):
    policy = policy or CFG["fx_input_policy"]
    assert policy in ("lag1", "same_day")
    if merged is None:
        merged, y = load_merged()
    L = 1 if policy == "lag1" else 0
    T, nl = len(merged), CFG["n_lags"]
    rows = splits(T)
    sc = rows["A"]
    m = merged.copy()
    for c in CCY:                                    # macro value changed vs previous row (a release became available)
        ch = (_num(m, f"{c}_CPIInfl").diff().abs() > 0) | (_num(m, f"{c}_GDP_growth").diff().abs() > 0)
        m[f"{c}_macro_changed"] = ch.astype(float)

    def series(c, f, lag):
        s = _num(m, f"{c}_{f}")
        if f in FX_COLS:
            if c == "USD":
                return pd.Series(0., index=m.index)
            lag += L
        return s.shift(lag)

    X, explicit, names = {}, {}, {}
    for ch, s in SCHEMA.items():
        cols, nm = [], []
        for f in s["slow"]:
            cols.append([series(c, f, 0) for c in CCY]); nm.append(f)
        fast_start = len(cols)
        for f in s["fast"]:
            for k in range(nl):
                cols.append([series(c, f, k) for c in CCY]); nm.append(f"{f}_lag{k}")
        for f in s["fast1"]:
            cols.append([series(c, f, 0) for c in CCY]); nm.append(f"{f}_lag0")
        fast_end = len(cols)
        raw = np.stack([np.stack([np.asarray(v, float) for v in col], 1) for col in cols], -1)   # (T, 11, F)
        if ch == "macro":                                # USD differences are identically 0
            raw[:, 0, 2:4] = 0.
        miss = ~np.isfinite(raw[:, :, fast_start:fast_end])
        ref = raw[sc][:, 1:] if ch == "fx" else raw[sc]
        mu, sd = np.nanmean(ref, (0, 1)), np.nanstd(ref, (0, 1))
        sd = np.where(sd > 1e-12, sd, 1.)
        zed = np.clip((raw - mu) / sd, -8, 8)
        if ch == "fx":
            zed[:, 0] = 0.
        zed = np.nan_to_num(zed)
        flags = [np.stack([np.asarray(series(c, f, 0), float) for c in CCY], 1)[..., None] for f in s["flags"]]
        parts = [zed] + [np.nan_to_num(f) for f in flags] + [miss.astype(float)]
        X[ch] = np.concatenate(parts, -1).astype(np.float32)
        nm += s["flags"] + [f"missing_{n}" for n in nm[fast_start:fast_end]]
        names[ch] = nm
        explicit[ch] = np.arange(fast_start, X[ch].shape[-1])
    g = []
    for col in GLOBAL:
        v = _num(m, col)
        if col in FX_REGIMES:
            v = v.shift(L)
        g.append(np.asarray(v, float))
    g = np.stack(g, 1)
    gm, gs = np.nanmean(g[sc], 0), np.nanstd(g[sc], 0)
    gs = np.where(gs > 1e-12, gs, 1.)
    G = np.nan_to_num(np.clip((g - gm) / gs, -8, 8)).astype(np.float32)
    lc = CFG["loss"]
    vol = np.stack([_num(m, f"{c}_FX_ret_1d").shift(L).rolling(lc["vol_window"], min_periods=2).std().fillna(lc["vol_floor"])
                    .clip(lower=lc["vol_floor"]).to_numpy() for c in CCY[1:]], -1)
    dates = pd.to_datetime(m.Date).values
    meta = dict(policy=policy, fx_lag=L, scaler_rows="A", regime_thresholds="legacy pipeline: first 60% of merged rows",
                n_lags=nl)
    return Data(merged=m, policy=policy, X=X, explicit=explicit, names=names, G=G, y=y, vol=vol, rows=rows, dates=dates, meta=meta)


def split_manifest(d: Data):
    def span(ix):
        ix = np.asarray(ix)
        return dict(n_dates=int(len(ix)), first_row=int(ix[0]), last_row=int(ix[-1]),
                    decision_start=str(pd.Timestamp(d.dates[ix[0]]).date()), decision_end=str(pd.Timestamp(d.dates[ix[-1]]).date()),
                    target_end_start=str(d.merged.Target_Date.iloc[ix[0]]), target_end_end=str(d.merged.Target_Date.iloc[ix[-1]]))
    r = d.rows
    out = {k: span(r[k]) for k in ("train", "A", "B", "C", "test")}
    out["A_oof_residual_rows"] = span(r["A_oof"])
    out["oof_folds"] = [dict(fit=span(f["fit"]), predict=span(f["predict"])) for f in r["folds"]]
    out["notes"] = ("A/B are the first 70% / last 30% of the legacy train block; C is the legacy validation block "
                    "(pit_original); the test block is not used by any automatic step.")
    out["fx_input_policy"] = d.policy
    return out


def global_index(name):
    return GLOBAL.index(name)
