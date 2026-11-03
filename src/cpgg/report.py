"""Build results/cpgg/CHECKPOINT_seed0.md (and tables) from synthetic/*.json and runs/<policy>/s<seed>/eval.json.

python -m cpgg.report --seed 0 --synthetic attempt1
"""
from __future__ import annotations

import argparse
import json

from .data import ROOT

OUT = ROOT / "results/cpgg"


def f(x, k=4):
    return "nan" if x is None else f"{x:.{k}f}"


def ci(r, s=1e3, k=2):
    return f"{r['mean'] * s:+.{k}f} [{r['lo'] * s:+.{k}f}, {r['hi'] * s:+.{k}f}]"


def verdict(r):
    return "지지 (CI가 0 미포함, 개선)" if r["lo"] > 0 else ("반대 방향 유의 (악화)" if r["hi"] < 0 else "C 구간에서 확인되지 않음")


def synth_section(sy):
    L = ["## 1. 합성 신호 점검 (12절, 사전 고정 기준)", ""]
    if "D" in sy:
        D = sy["D"]
        L += [f"- **D (시차·gradient·스케일)**: 주입 lag 도달 오차 {D['lag_reach_max_abs_err']:.1e}, 입력 gradient {D['grad_input_source_lag1']:.2e}, "
              f"출력층·beta·쌍 계수 gradient 모두 0 아님 = {all(D['grad_params_nonzero'].values())}, 변동성 floor 비율 {D['vol_floor_share_A']:.4f}, "
              f"CPU–MPS 최대 차이 {D['cpu_mps_max_abs_diff']}, **통과 {D['passed']}**"]
    if "A" in sy:
        L += ["", "**A. 독립 합성 source, 강제 edge (CHF→JPY, lag 1)** — 통과 기준: SNR 1.0·0.3에서 강제 edge 개선 ≥ oracle 개선의 50%, CI 하한 > 0", "",
              "| SNR | 계수 | 강제 edge 개선 (×10³) [95% CI] | oracle 개선 (×10³) | 비율 | 메시지 기울기 | G-loss 선택률 CHF / SEK | G-rel 선택률 CHF | 판정 |", "|---|---|---|---|---|---|---|---|---|"]
        for r in sy["A"]["rows"]:
            sl, sr = r["selection"].get("loss", {}), r["selection"].get("rel", {})
            L.append(f"| {r['snr']} | {r['coef']:.3f} | {ci(r['forced_gain'])} | {r['oracle_gain']['mean'] * 1e3:.2f} | {r['ratio']:.2f} | "
                     f"{r['message_slope_on_source']:.3f} | {sl.get('src', float('nan')):.2f} / {sl.get('noise', float('nan')):.2f} | {sr.get('src', float('nan')):.2f} | "
                     f"{'-' if r['passed'] is None else ('통과' if r['passed'] else '실패')} |")
        L.append(f"\nA 전체 판정: **{'통과' if sy['A']['passed'] else '실패'}**")
    if "B" in sy:
        B = sy["B"]
        L += ["", "**B. regime 조건부 신호 (VIX 변화 > 0인 날만)** — 통과 기준: CHF→JPY 선택률이 regime 날에 유의하게 높음", "",
              "| 구분 | 강제 edge 개선 (×10³) | 평균 |d| | 실현 ρ |", "|---|---|---|---|"]
        for k, v in B["message"].items():
            L.append(f"| {k} | {ci(v['forced_gain'])} | {v['mean_abs_d']:.3f} | {v['realized_rho']:.2f} |")
        L += ["", "| 선택기 | 설정 | 선택률 regime | 선택률 비regime | 차이 [95% CI] | 판정 |", "|---|---|---|---|---|---|"]
        for k, v in B["selection"].items():
            L.append(f"| {'G-loss' if k == 'loss' else 'G-rel'} | {v['config']} | {v['src_regime']:.3f} | {v['src_non_regime']:.3f} | "
                     f"{v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] | {'통과' if B['passed'][k] else '실패'} |")
    if "C" in sy:
        C = sy["C"]
        L += ["", "**C. 무신호(SEK)·중복(NOK = CHF 신호 + 잡음 0.1) source** — 기준: 무신호 선택률 ≤ 10%, CHF 선택 후 중복의 추정 추가 gain ≤ CHF 단독 gain의 절반", "",
              "| 선택기 | SEK 선택률 | CHF 단독 추정 gain | NOK 단독 | NOK | {CHF} | 판정 |", "|---|---|---|---|---|---|"]
        for k in ("loss", "rel"):
            if k in C:
                v = C[k]
                L.append(f"| {'G-loss' if k == 'loss' else 'G-rel'} | {v['noise_rate']:.3f} | {v['est_src_alone']:.2e} | {v['est_dup_alone']:.2e} | "
                         f"{v['est_dup_given_src']:.2e} | {'통과' if C['passed'][k] else '실패'} |")
    return L


def run_section(ev, title):
    L = [f"## {title}", ""]
    p = ev["primary"]
    L += ["**주 비교 (날짜별 10개 통화 평균 prediction loss의 paired 차이, 양수 = 개선, ×10³, 10일 블록 bootstrap 10,000회)**", "",
          "| 비교 | block 10 | block 5 | block 20 | 판정 (block 10) |", "|---|---|---|---|---|"]
    for k, v in p.items():
        L.append(f"| {k} | {ci(v['block10'])} | {ci(v['block5'])} | {ci(v['block20'])} | {verdict(v['block10'])} |")
    L += ["", "**모델 비교 (C 구간 1,048일)**", "",
          "| 모델 | pred loss | return loss | dir log-loss | skill vs 0.5 % | skill vs 통화별 상수 % | Mean Hit | IC USD | IC cross | RMSE ×10³ | K |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in ev["table"].items():
        L.append(f"| {name} | {m['pred_loss']:.5f} | {m['return_loss']:.5f} | {m['dir_logloss']:.5f} | {100 * m['skill_vs_half']:+.2f} | "
                 f"{100 * m['skill_vs_const']:+.2f} | {m['mean_hit']:.4f} | {m['ic_usd']:+.4f} | {m['ic_cross']:+.4f} | {1e3 * m['rmse']:.3f} | {m['K_mean']:.2f} |")
    L += ["", "**Own-only / Dense 대비 paired 차이 (×10³, block 10)**", "", "| 비교 | 차이 [95% CI] |", "|---|---|"]
    for k, v in ev["others"].items():
        L.append(f"| {k} | {ci(v['block10'])} |")
    L += ["", f"Top-K 후보 C loss: {json.dumps({k: round(v, 5) for k, v in ev['topk']['topk_loss'].items()})} → K={ev['topk']['topk_chosen']}; "
          f"static edge 수 {ev['static_edges']}/100", ""]
    L += ["**선택기 (C)**", "", "| 선택기 | 설정 | K 평균 | K=0 비율 | K=10 비율 | 인접일 Jaccard | 추정–실현 상관 | 추정>0 비율 | 실현 gain (추정>0) | 실현 gain (추정≤0) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in ev["selectors"].items():
        g, c = s["graph"], s["calibration"]
        L.append(f"| {'G-loss' if k == 'loss' else 'G-rel'} | {s['config']} | {g['K_mean']:.2f} | {g['K0']:.3f} | {g['K10']:.3f} | {g['jaccard_adjacent']:.3f} | "
                 f"{c['corr_est_realized']:+.3f} | {c['share_est_positive']:.3f} | {c['realized_mean_when_est_pos']:+.2e} | {c['realized_mean_when_est_neg']:+.2e} |")
    for k, s in ev["selectors"].items():
        tag = "G-loss" if k == "loss" else "G-rel"
        L += ["", f"*{tag} calibration (추정 gain 10분위, 실현 gain 95% CI)*", "", "| 분위 | 추정 평균 | 실현 평균 | CI |", "|---|---|---|---|"]
        for r in s["calibration"]["deciles"]:
            L.append(f"| {r['decile']} | {r['est_mean']:+.2e} | {r['realized_mean']:+.2e} | [{r['lo']:+.2e}, {r['hi']:+.2e}] |")
        if "rho_quintiles" in s["calibration"]:
            L += ["", "| ρ̂ 5분위 | 추정 ρ | 실현 ρ |", "|---|---|---|"] + [
                f"| {r['q']} | {r['rho_est']:.3f} | {r['rho_realized']:.3f} |" for r in s["calibration"]["rho_quintiles"]]
        L += ["", f"*{tag} 순차 선택 경로 (C)*: " + "; ".join(
            f"{r['step']}번째 연결 n={r['n']} 추정 {r['est_mean']:+.2e} 실현 {r['realized_mean']:+.2e} [{r['realized_ci']['lo']:+.2e}, {r['realized_ci']['hi']:+.2e}]"
            for r in s["path"][:4])]
        L += [f"*K 분포 (0..10)*: {s['graph']['K_hist']}; RiskOff {s['graph']['K_riskoff']:.2f} vs 아님 {s['graph']['K_not_riskoff']:.2f}; "
              f"HighVol {s['graph']['K_highvol']:.2f} vs 아님 {s['graph']['K_not_highvol']:.2f}",
              f"*쌍 점검*: 단독 추정 gain ≤ 0인 후보 쌍 {s['pairs']['n_pairs']}개 중 함께 추가 시 실현 gain > 0 비율 {s['pairs']['share_pairs_positive']:.3f}"]
    rel = ev["reliability"]
    keys = ["all", "Regime_RiskOff", "not_Regime_RiskOff", "Regime_HighVol", "not_Regime_HighVol", "absd_q1", "absd_q3", "absd_q5"]
    L += ["", "**실현 신뢰도 ρ = Σe·d / Σd² (빈 집합 기준; 0.5 초과일 때만 추가가 이득)**", "",
          "| 구간 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for s in ("A", "B", "C"):
        L.append(f"| {s} | " + " | ".join(f"{rel[s].get(k, float('nan')):.3f}" for k in keys) + " |")
    al = ev["residual_alignment"]
    L += ["", "**잔차 정합 (6.1, e = z − b)**: " + "; ".join(f"{s}: 평균 {v['mean']:+.3f}, sd {v['sd']:.3f}, b sd {v['b_sd']:.3f}, corr(b,z) {v['corr_b_z']:+.3f}" for s, v in al.items()),
          "", f"**구조 검사**: 분해 최대 오차 {ev['consistency']['decomposition_max_abs_err']:.2e}, 반대칭 {ev['consistency']['antisymmetry_max']:.1e}, 삼각 {ev['consistency']['triangle_max']:.1e}",
          f"\n실행 시간 {ev['seconds_total']:.0f}초, 최대 메모리 {ev['max_rss_mb']:.0f} MB"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", default="attempt1")
    a = ap.parse_args()
    sy = json.loads((OUT / f"synthetic/{a.synthetic}.json").read_text())
    L = [f"# CPGG 체크포인트 (seed {a.seed}, 개발 구간 C만 사용, test 미사용)", ""]
    L += synth_section(sy) + [""]
    for pol, title in (("lag1", "2. 실제 데이터, 주 설정 (FX 입력 t−1까지, lag1)"), ("same_day", "3. 민감도: 당일 12:00 FX 입력 (실시간 피드 가정)")):
        p = OUT / f"runs/{pol}/s{a.seed}/eval.json"
        if p.exists():
            L += run_section(json.loads(p.read_text()), title) + [""]
    (OUT / f"CHECKPOINT_seed{a.seed}_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
