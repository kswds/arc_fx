"""Data and timing audit for CPGG -> results/cpgg/DATA_AUDIT.md (+ audit.json). Uses train-block rows only for statistics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import CCY, CFG, ROOT, build, load_merged

P = ROOT / "data/processed"
INV = {"JPY", "CAD", "CHF", "SEK", "NOK", "KRW"}


def main():
    out = {}
    fx = pd.read_csv(P / "fx_panel_pit.csv", parse_dates=["Date"])
    nf = pd.read_csv(P / "nonfx_features_pit.csv", parse_dates=["Date"])
    out["rows"] = dict(fx=len(fx), nonfx=len(nf), dates_equal=bool((fx.Date.values == nf.Date.values).all()),
                       sorted=bool(fx.Date.is_monotonic_increasing), duplicates=int(fx.Date.duplicated().sum()))
    td = pd.to_datetime(fx.Target_Date)
    out["target_date"] = dict(equals_next_row_date=float((td.iloc[:-1].values == fx.Date.iloc[1:].values).mean()),
                              gap_days_max=int((td - fx.Date).dt.days.max()), gap_days_min=int((td - fx.Date).dt.days.min()))
    err = {}
    for c in CCY[1:]:
        lv = lambda s: -np.log(s) if c in INV else np.log(s)
        rec = lv(fx[f"Target_{c}_FX"]) - lv(fx[f"{c}_FX"])
        err[c] = float(np.nanmax(np.abs(rec - fx[f"TARGET_{c}_FX_RET_FWD1"])))
        nxt = lv(fx[f"{c}_FX"]).shift(-1) - lv(fx[f"{c}_FX"])
        err[c + "_vs_next_row"] = float(np.nanmax(np.abs(nxt.iloc[:-1] - fx[f"TARGET_{c}_FX_RET_FWD1"].iloc[:-1])))
    out["target_recompute_max_abs_err"] = err
    av = pd.read_csv(P / "availability_audit.csv")
    out["availability_audit"] = dict(series=len(av), violations_avail_after_cutoff=int(av.violations_avail_after_cutoff.sum()),
                                     violations_obs_after_t=int(av.violations_obs_date_after_t.sum()),
                                     fx_series_in_audit=int(av.series.str.contains("_FX").sum()))
    ms = pd.read_csv(P / "missingness_by_split.csv")
    out["missing_train_over_10pct"] = ms[ms.train > .10][["column", "train", "val", "test"]].round(3).to_dict("records")
    d = build("lag1")
    out["feature_names"] = d.names
    out["target_in_features"] = any("TARGET" in n.upper() for v in d.names.values() for n in v)
    m, y = load_merged()
    tr = d.rows["train"]
    r = np.stack([pd.to_numeric(m[f"{c}_FX_ret_1d"], errors="coerce").values for c in CCY[1:]], 1)
    yy = y[:, 1:]
    cc = np.corrcoef(r[tr].T); iu = np.triu_indices(10, 1)
    lead = np.array([[np.corrcoef(r[tr, j], yy[tr, i])[0, 1] for j in range(10)] for i in range(10)])
    same_row = np.array([np.corrcoef(r[tr, i], yy[tr, i])[0, 1] for i in range(10)])
    out["train_stats"] = dict(median_abs_contemporaneous_corr=float(np.median(np.abs(cc[iu]))),
                              median_abs_corr_other_ret_t_vs_target=float(np.median(np.abs(lead[~np.eye(10, dtype=bool)]))),
                              median_abs_corr_own_ret_t_vs_target=float(np.median(np.abs(same_row))))
    out["vol_floor_share_train"] = float((d.vol[tr] <= CFG["loss"]["vol_floor"] * 1.0001).mean())
    out["timing"] = dict(
        decision_row="t (H.10 noon New York of date t)", target="log return noon t -> noon of next H.10 date (TARGET_*_FWD1 of row t)",
        nonfx_cutoff="11:30 New York on t (as supplied; build asserts 0 violations)",
        fx_policy_primary="lag1: FX-derived inputs use rows <= t-1; latest FX return input = noon t-2 -> noon t-1",
        fx_policy_sensitivity="same_day: FX return noon t-1 -> noon t is an input; requires a real-time feed observing the 12:00 rate at decision time",
        h10_publication="H.10 rates are published by the Federal Reserve with a delay (weekly release); the panel does not record "
                        "an available_at for FX, so neither policy is verified against FRED publication times",
        observation_age="not reconstructable for FX from the supplied CSVs; non-FX lags are in availability_audit.csv (lag_days_mean/max)",
        scalers="CPGG: A rows only", regime_thresholds="legacy pipeline: first 60% of merged rows (inside the train block)")
    (ROOT / "results/cpgg").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/cpgg/audit.json").write_text(json.dumps(out, indent=1))
    lines = ["# CPGG 데이터·시점 감사", "",
             f"- 행: FX {out['rows']['fx']}, non-FX {out['rows']['nonfx']}; 날짜 일치 {out['rows']['dates_equal']}, 정렬 {out['rows']['sorted']}, 중복 {out['rows']['duplicates']}",
             f"- Target_Date = 다음 행 날짜 비율 {out['target_date']['equals_next_row_date']:.4f}; 간격 {out['target_date']['gap_days_min']}~{out['target_date']['gap_days_max']}일",
             f"- forward 로그수익률 재계산 최대 오차: {max(v for k, v in err.items() if not k.endswith('next_row')):.2e} (호가 방향 반영), 다음 행 수준과의 최대 오차 {max(v for k, v in err.items() if k.endswith('next_row')):.2e}",
             f"- availability_audit: {out['availability_audit']['series']}개 계열, cutoff 이후 공개 위반 {out['availability_audit']['violations_avail_after_cutoff']}, 관측일>t 위반 {out['availability_audit']['violations_obs_after_t']}, FX 계열 포함 수 {out['availability_audit']['fx_series_in_audit']} (FX는 audit에 없음)",
             f"- feature에 TARGET 열 포함: {out['target_in_features']}",
             f"- train 구간: 동시 수익률 |상관| 중앙값 {out['train_stats']['median_abs_contemporaneous_corr']:.3f}, 다른 통화 t 수익률과 대상 target |상관| 중앙값 {out['train_stats']['median_abs_corr_other_ret_t_vs_target']:.3f}",
             f"- 변동성 floor에 걸린 비율(train): {out['vol_floor_share_train']:.4f}", "",
             "## 시점 가정", ""] + [f"- **{k}**: {v}" for k, v in out["timing"].items()] + [
             "", "## train 결측 10% 초과 열", "", "| 열 | train | val | test |", "|---|---|---|---|"] + [
             f"| {r['column']} | {r['train']} | {r['val']} | {r['test']} |" for r in out["missing_train_over_10pct"]] + [
             "", "주 결과는 FX 입력을 t-1까지로 제한(lag1)해 11:30 cutoff와 12:00 FX 관측의 충돌을 없앤다. same_day는 실시간 피드 가정의 민감도 분석이다.",
             "공식 FRED/H.10 공개 시점까지 확인된 입력이라고 주장하지 않는다."]
    (ROOT / "results/cpgg/DATA_AUDIT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
