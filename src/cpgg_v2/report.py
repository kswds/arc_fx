"""Tables for results/cpgg_v2/CHECKPOINT_seed0_tables.md from synthetic/<tag>.json and runs/s<seed>/eval.json.
PYTHONPATH=. python -m cpgg_v2.report --seed 0 --synthetic attempt0
"""
from __future__ import annotations

import argparse
import json

from cpgg.data import ROOT

OUT = ROOT / "results/cpgg_v2"
REG = {"0": "평시", "1": "RiskOff", "2": "HighVol"}
ci = lambda r, s=1e3: f"{r['mean'] * s:+.2f} [{r['lo'] * s:+.2f}, {r['hi'] * s:+.2f}]"
verdict = lambda r: "개선 (유의)" if r["lo"] > 0 else ("악화 (유의)" if r["hi"] < 0 else "C 구간에서 확인되지 않음")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--synthetic", default="attempt0")
    a = ap.parse_args()
    sy = json.loads((OUT / f"synthetic/{a.synthetic}.json").read_text())
    ev = json.loads((OUT / f"runs/s{a.seed}/eval.json").read_text())
    L = [f"# CPGG-v2 체크포인트 표 (seed {a.seed}, C 구간, test 미사용)", "", "## 합성 점검", ""]
    D = sy["D"]
    L.append(f"- D: 시차 도달 오차 {D['lag_reach_max_abs_err']:.1e}, 분해 오차 {D['decomposition_max_abs_err']:.1e}, 자기쌍 메시지 최대 {D['self_message_max']}, "
             f"USD FX 입력 최대 {D['usd_fx_inputs_max']}, 칸 특징 누수 변화 {D['cell_feature_leak_max_change']}, 메시지–주입 신호 상관(C) {D['corr_message_signal_C']:.3f}, "
             f"기록 구간 placebo 선택률 {D['placebo_rate_records']:.3f} → **{'통과' if D['passed'] else '실패'}**")
    L += ["", "| SNR | 강제 edge 개선 ×10³ [CI] | oracle ×10³ | 비율 | d–신호 상관 | S: CHF / SEK 선택률 | D: CHF / SEK 선택률 | 판정 |", "|---|---|---|---|---|---|---|---|"]
    for r in sy["A"]["rows"]:
        s, d = r["selection"]["S"], r["selection"]["D"]
        p = r.get("passed"); p = "-" if p is None else (("통과" if p else "실패") if isinstance(p, bool) else ", ".join(f"{k} {'통과' if v else '실패'}" for k, v in p.items()))
        L.append(f"| {r['snr']} | {ci(r['forced_gain'])} | {r['oracle_gain']['mean'] * 1e3:.2f} | {r['ratio']:.2f} | {r['corr_d_signal']:.3f} | "
                 f"{s['src']:.3f} / {s['noise']:.3f} | {d['src']:.3f} / {d['noise']:.3f} | {p} |")
    L += ["", "| B regime 집합 | 선택기 | regime 날 선택률 | 아닌 날 | 차이 [CI] | 메시지 |d| regime / 아닌 날 | 메시지 ρ regime / 아닌 날 | 판정 |", "|---|---|---|---|---|---|---|---|"]
    for rk in ("vix", "main"):
        b = sy["B"][rk]
        for k, v in b["selection"].items():
            L.append(f"| {'주입 regime (VIX 변화>0)' if rk == 'vix' else '실제 3-regime (참고)'} | {k} | {v['src_regime']:.3f} | {v['src_non_regime']:.3f} | "
                     f"{v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] | {b['message']['regime']['mean_abs_d']:.3f} / {b['message']['non_regime']['mean_abs_d']:.3f} | "
                     f"{b['message']['regime']['realized_rho']:.2f} / {b['message']['non_regime']['realized_rho']:.2f} | {'통과' if b['passed'][k] else '실패'} |")
    C = sy["C"]
    L += ["", f"- C: SEK(무신호) 선택률 S {C['S']['noise_rate']:.3f}, D {C['D']['noise_rate']:.3f}; CHF 단독 추정 gain {C['S']['est_src_alone']:.3f}, "
          f"NOK 단독 {C['S']['est_dup_alone']:.3f}, CHF 선택 후 NOK {C['S']['est_dup_given_src']:.3f}; 순차 greedy에서 CHF 선택 시 NOK 추가 비율 {C['S']['greedy_dup_rate_given_src']:.3f} "
          f"→ S {'통과' if C['passed']['S'] else '실패'}, D(무신호만) {'통과' if C['passed']['D'] else '실패'}"]
    L += ["", "## 실제 데이터: 주 지표 (날짜별 10개 통화 평균 prediction loss paired 차이 ×10³, 양수 = 앞 모델이 좋음)", "",
          "| 비교 | block 10 | block 5 | block 20 | 판정 (block 10) |", "|---|---|---|---|---|"]
    for k, v in ev["primary"].items():
        L.append(f"| {k} | {ci(v['block10'])} | {ci(v['block5'])} | {ci(v['block20'])} | {verdict(v['block10'])} |")
    L += ["", "## 보조 지표 (C 구간 1,048일)", "", "| 모델 | pred loss | dir log-loss | skill vs 상수 % | Mean Hit | Brier | IC USD | IC cross | return loss | RMSE ×10³ | K | VIX stress gain ×10³ | Yield stress gain ×10³ |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in ev["secondary"].items():
        L.append(f"| {k} | {m['pred_loss']:.5f} | {m['dir_logloss']:.5f} | {100 * m['skill_vs_const']:+.2f} | {m['mean_hit']:.4f} | {m['brier']:.4f} | {m['ic_usd']:+.4f} | "
                 f"{m['ic_cross']:+.4f} | {m['return_loss']:.5f} | {1e3 * m['rmse']:.3f} | {m['K_mean']:.2f} | {1e3 * m['gain_vs_own_vix_stress']:+.2f} | {1e3 * m['gain_vs_own_yield_stress']:+.2f} |")
    L += ["", "## 선택기 설정별 C 결과 (주 규칙: 보정 gain > c0)", "", "| 설정 | C loss | C loss (임계값 0) | c0 | C placebo 선택률 | K |", "|---|---|---|---|---|---|"]
    for n, r in ev["configs"].items():
        L.append(f"| {n} | {r['loss']:.5f} | {r['loss_thr0']:.5f} | {r['c0']:+.2e} | {r['placebo_rate_C']:.3f} | {r['K_mean']:.2f} |")
    for k in ("S", "D"):
        s = ev["selector"][k]
        g = s["selected_gain_per_date"]
        L += ["", f"## 선택기 진단: G-loss-{k} ({s['config']})", "",
              f"- c0 = {s['c0']:+.2e}, placebo 선택률 C {s['placebo_rate_C']:.3f} (기록 구간 {s['placebo_rate_records']:.3f})",
              f"- 선택된 연결의 실현 gain (날짜별 평균, 10일 블록 CI) ×10³: {ci(g) if g else '-'}; 선택 수 {s['n_selected']}, 선택이 있는 날 {s['dates_with_selection']}",
              f"- K 평균 {s['K_mean']:.2f}, K=0 비율 {s['K0']:.3f}, K 분포(0..10) {s['K_hist']}, regime별 K " + ", ".join(f"{REG.get(r, r)} {v:.2f}" for r, v in s["K_by_regime"].items()) +
              f", 인접일 Jaccard {s['jaccard_adjacent']}",
              f"- 추정–실현 상관: block 5 raw {s['corr_block5_raw']:+.3f} / 보정 {s['corr_block5_cal']:+.3f}; C raw {s['corr_C_raw']:+.3f} / 보정 {s['corr_C_cal']:+.3f}",
              "", "| regime | 선택률 | 선택된 연결의 실현 ρ | 전체 메시지 실현 ρ |", "|---|---|---|---|"]
        for r, v in s["by_regime"].items():
            rs = "-" if v["rho_selected"] is None else f"{v['rho_selected']:.2f}"
            L.append(f"| {REG.get(r, r)} | {v['sel_rate']:.3f} | {rs} | {v['rho_all']:.2f} |")
        L += ["", "| 자주 선택된 쌍 | 선택률 | 선택 시 실현 ρ | 전체 실현 ρ |", "|---|---|---|---|"]
        for t in s["top_pairs"][:8]:
            L.append(f"| {t['pair']} | {t['rate']:.3f} | {t['rho_selected']:.2f} | {t['rho_all']:.2f} |")
        for lab in ("block5", "C"):
            L += ["", f"*calibration {lab}: 추정 gain 10분위 (raw → 실현 / 보정 → 실현), ×10³*", "", "| 분위 | raw 추정 | 실현 [CI] | 보정 추정 |", "|---|---|---|---|"]
            for rr, rc in zip(s[f"calib_{lab}_raw"], s[f"calib_{lab}_cal"]):
                L.append(f"| {rr['decile']} | {rr['est_mean'] * 1e3:+.3f} | {rr['realized_mean'] * 1e3:+.3f} [{rr['lo'] * 1e3:+.3f}, {rr['hi'] * 1e3:+.3f}] | {rc['est_mean'] * 1e3:+.3f} |")
    for lab, m in (("M-v2", ev["messages_v2"]), ("M-v1", ev["messages_v1"])):
        if not m:
            continue
        L += ["", f"## 메시지 {lab}: 실현 ρ와 d 분포", "", "| 구간 | d 평균 | d sd | 실현 ρ | regime별 ρ (평시/RiskOff/HighVol) |", "|---|---|---|---|---|"]
        for k, f in enumerate(m["fit"]):
            L.append(f"| fold {k + 1 if k < 3 else 'final'} 학습 구간 (in-sample) {f['fit_rows']} | {f['d_mean']:+.4f} | {f['d_sd']:.4f} | {f['rho_fit']:.2f} | - |")
        for f in m["applied"]:
            L.append(f"| {'gain 기록 블록 ' + str(f['fold'] + 2) if f['fold'] != 'final(C)' else 'C (최종 메시지)'} | {f['d_mean']:+.4f} | {f['d_sd']:.4f} | {f['rho']:.2f} | "
                     + " / ".join(f"{v:.2f}" for v in f["rho_by_regime"].values()) + " |")
        if "coef_corr_with_final" in m:
            L.append(f"\n계수 상관: fold 1~3 vs 최종 {[round(x, 3) for x in m['coef_corr_with_final']]}, 인접 fold {[round(x, 3) for x in m['coef_corr_adjacent_folds']]}")
        L.append(f"쌍별 실현 ρ의 기록 구간–C 상관: {m['pair_rho_corr_records_vs_C']:+.3f}")
    L.append(f"\n실행 {ev['seconds']:.0f}초, 최대 메모리 {ev['max_rss_mb']:.0f} MB (CPU 스레드 2개, 프로세스 1개)")
    (OUT / f"CHECKPOINT_seed{a.seed}_tables.md").write_text("\n".join(L) + "\n")
    print("written", len(L), "lines")


if __name__ == "__main__":
    main()
