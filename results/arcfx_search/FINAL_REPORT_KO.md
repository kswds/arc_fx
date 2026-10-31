# ARC_FX 탐색 결과

결과를 꾸미거나 유리한 seed·기간으로 전체 성능을 대체하지 않았다. 아래 유리한 사례는 사후 탐색 결과로 별도 표시한다.

Development 선택 모델: **dense_signed**. 전체 보고 구간 Mean Hit: **0.50545**, seed 표준편차 **0.00231**, 5 seeds.
전체 평균 기준 0.52 목표: **미달**. 범위 0.50229–0.50753.

평가 return 종료일: 2021-10-07–2025-12-18, 1,049일 × 10개 통화. 대응 decision date는 2021-10-06–2025-12-17다.

## 1. 진단과 해석

- 기존 PIT 두 설정의 checkpoint와 예측은 다르다. 부호 불일치는 seed별 0.87–1.85%; 같은 0.4997은 반올림된 집계 결과다.
- 평균을 뺀 target과 원수익률의 이진 label 불일치는 약 0.80%, 3-way sign 불일치는 약 1.06%다. 그러나 Q40 이상 기존 active 표본에서는 0%다. 따라서 label 문제만으로 기존 성능을 설명할 수 없다.
- 기존 입력은 t−1에서 끝나지만 label은 t→t+1이다. 신규 입력은 t까지 사용하고 실제 return 종료일을 별도 저장했다. 기존 방식보다 정보시점이 하루 신선해졌다는 점을 비교 해석에 포함해야 한다.
- 기존 smoothness는 셔플된 배치 간 비교이며 실제 시간 인접성 규제가 아니다. 기존 relational branch는 메시지가 없어도 자기 상태에서 출력이 생긴다.
- 16일 subset 과적합은 sign-hit 0.99375로 통과했다. 0 target 처리 때문에 binary-perfect와 sign-hit-perfect는 다르다. 전체 모델의 일반화가 입증된 것은 아니다.
- 공급된 availability audit의 시각 위반은 0건이다. 이는 코드의 가용시각 가정 아래 결과이며, 실제 역사적 공표시각·H.10 실시간 가용성을 완전히 보증하지 않는다.

## 2. 전체 결과: 유리한 seed를 선택하지 않음

| name            | fold   | split   |    mean |     std |     min |     max |   count |
|:----------------|:-------|:--------|--------:|--------:|--------:|--------:|--------:|
| dense_signed    | final  | report  | 0.50545 | 0.00231 | 0.50229 | 0.50753 |       5 |
| original_legacy | final  | report  | 0.49507 | 0.00461 | 0.49123 | 0.50019 |       3 |
| soft_threshold  | final  | report  | 0.50207 | 0.00279 | 0.49886 | 0.50391 |       3 |
| winner_no_graph | final  | report  | 0.50423 | 0.00033 | 0.50391 | 0.50458 |       3 |

## 3. Development 결과

| name              | fold   | split      |    mean |       std |     min |     max |   count |
|:------------------|:-------|:-----------|--------:|----------:|--------:|--------:|--------:|
| aligned_active    | dev1   | validation | 0.49124 | nan       | 0.49124 | 0.49124 |       1 |
| aligned_active    | dev2   | validation | 0.49810 | nan       | 0.49810 | 0.49810 |       1 |
| all_bce           | dev1   | validation | 0.49543 | nan       | 0.49543 | 0.49543 |       1 |
| all_bce           | dev2   | validation | 0.49790 | nan       | 0.49790 | 0.49790 |       1 |
| compact_gru       | dev1   | validation | 0.49352 | nan       | 0.49352 | 0.49352 |       1 |
| compact_gru       | dev2   | validation | 0.49771 | nan       | 0.49771 | 0.49771 |       1 |
| dense_signed      | dev1   | validation | 0.50507 |   0.00853 | 0.49638 | 0.51410 |       5 |
| dense_signed      | dev2   | validation | 0.50255 |   0.00569 | 0.49619 | 0.51162 |       5 |
| dual_003          | dev1   | validation | 0.49467 | nan       | 0.49467 | 0.49467 |       1 |
| dual_003          | dev2   | validation | 0.49657 | nan       | 0.49657 | 0.49657 |       1 |
| dual_010          | dev1   | validation | 0.49448 | nan       | 0.49448 | 0.49448 |       1 |
| dual_010          | dev2   | validation | 0.49676 | nan       | 0.49676 | 0.49676 |       1 |
| lowdim_message    | dev1   | validation | 0.49638 | nan       | 0.49638 | 0.49638 |       1 |
| lowdim_message    | dev2   | validation | 0.50838 | nan       | 0.50838 | 0.50838 |       1 |
| original_legacy   | dev1   | validation | 0.49460 |   0.00418 | 0.48990 | 0.49790 |       3 |
| original_legacy   | dev2   | validation | 0.50095 |   0.00468 | 0.49562 | 0.50438 |       3 |
| original_residual | dev1   | validation | 0.49295 | nan       | 0.49295 | 0.49295 |       1 |
| original_residual | dev2   | validation | 0.50438 | nan       | 0.50438 | 0.50438 |       1 |
| soft_threshold    | dev1   | validation | 0.50908 |   0.01051 | 0.49695 | 0.51543 |       3 |
| soft_threshold    | dev2   | validation | 0.50362 |   0.00629 | 0.49752 | 0.51010 |       3 |
| static_qk         | dev1   | validation | 0.51219 | nan       | 0.51219 | 0.51219 |       1 |
| static_qk         | dev2   | validation | 0.49543 | nan       | 0.49543 | 0.49543 |       1 |
| temporal_mlp      | dev1   | validation | 0.51276 | nan       | 0.51276 | 0.51276 |       1 |
| temporal_mlp      | dev2   | validation | 0.49524 | nan       | 0.49524 | 0.49524 |       1 |

각 fold는 과거 데이터만으로 scaler와 threshold를 계산했다. 후보 선택에 사용된 구간이므로 이 표의 최고값은 선택 편향을 포함한다. 최종 보고 기간은 이미 본 기간이며 untouched test가 아니다. 선택은 seeds 0,1,2 기준이며 dense_signed의 seeds 3,4는 선택을 고정한 뒤 추가 검증했다. 따라서 위 표의 5-seed 평균은 원래 선택 순위와 다를 수 있다. 1-seed 행의 std=nan은 추정 불가를 의미한다.

## 4. 선택 모델의 seed·연도 안정성

|    seed |   mean_hit |     bce |   pred_positive |   actual_positive |   logit_std |   zero_graph_hit |
|--------:|-----------:|--------:|----------------:|------------------:|------------:|-----------------:|
| 0.00000 |    0.50229 | 0.69515 |         0.43251 |           0.48008 |     0.16845 |          0.50257 |
| 1.00000 |    0.50381 | 0.69302 |         0.31077 |           0.48008 |     0.08517 |          0.50572 |
| 2.00000 |    0.50734 | 0.69307 |         0.21668 |           0.48008 |     0.09203 |          0.50820 |
| 3.00000 |    0.50753 | 0.69623 |         0.40543 |           0.48008 |     0.21643 |          0.51459 |
| 4.00000 |    0.50629 | 0.69659 |         0.44118 |           0.48008 |     0.19522 |          0.50133 |

|       year |   hit_mean |   hit_std |   samples_per_seed |
|-----------:|-----------:|----------:|-------------------:|
| 2021.00000 |    0.49368 |   0.01168 |          570.00000 |
| 2022.00000 |    0.50488 |   0.01293 |         2500.00000 |
| 2023.00000 |    0.50482 |   0.00657 |         2490.00000 |
| 2024.00000 |    0.51052 |   0.01020 |         2510.00000 |
| 2025.00000 |    0.50421 |   0.01419 |         2420.00000 |

## 5. 사용자 요청: 유리한 seed·기간 (POST HOC)

선택 모델의 전체 보고 기간에서 가장 유리한 seed는 **3**, Mean Hit **0.50753**다.
선택 모델의 seed×달력연도 조합 최고는 **seed 2, 2024년**, Hit **0.52510**, 2510 통화-날짜 표본이다. 정확한 return 종료일 범위는 2024-01-02–2024-12-31다.
전체 보고 모델을 통틀어 가장 높은 연간 단일 seed 값은 original_legacy, seed 0, 2024년의 0.53625다. 이 값을 선택 모델의 성능으로 섞지 않는다.
이 값은 다중 비교 후 최대값이며 대표 성능이나 재현성의 근거가 아니다. 전체 후보의 모든 보고 seed×연도를 좋은 순서부터 나쁜 순서까지 `favorable_seed_period_POSTHOC.csv`에 저장했다. 임의 이동창이나 유리한 시작·종료일은 추가 탐색하지 않았다.

같은 날짜 baseline 대비 유리한 차이와 불리한 차이는 `favorable_relative_POSTHOC.csv`에 모두 저장했다. 이것도 seed·연도·baseline을 반복 비교한 사후 통계로, 독립 검증이 아니다.

## 6. 그래프 기여

선택 모델의 graph를 추론 시 제거한 Mean Hit 평균은 0.50648; 원래 출력 대비 차이는 -0.00103다.
이는 고정된 모델의 의존성 검사다. 재학습 no-graph 결과와 paired bootstrap을 함께 봐야 실제 성능 이득을 주장할 수 있다. 원래 backbone에서는 adjacency=0이 전체 relational branch 제거와 다르다.
선택된 dense graph의 off-diagonal edge는 모두 활성이고 gate는 포화되지 않았으며 상태에 따라 adjacency가 변한다. Graph gradient도 0이 아니다. 그러나 전체 예측 개선에 대한 독립적 기여는 확인되지 않았다.
`graph_diagnostics.csv`는 초기/선택 checkpoint의 실제 메시지 RMS, gate 분위수·포화, off-diagonal 활성률, 상태별 adjacency 변화, encoder/graph/head gradient와 손실 항을 담는다. 초기 비교는 동일한 validation 128일, gradient는 고정된 과거 256일을 사용한다. Epoch 로그의 gradient/regularization은 마지막 minibatch 진단값이다.

## 7. Baseline 차이와 불확실성

| comparator                     |   n_dates |   baseline_mean_hit |   difference |   ci_low |   ci_high |
|:-------------------------------|----------:|--------------------:|-------------:|---------:|----------:|
| majority                       |      1049 |             0.50982 |     -0.00437 | -0.01686 |   0.00881 |
| logistic                       |      1049 |             0.50257 |      0.00288 | -0.00938 |   0.01583 |
| original_legacy                |      1049 |             0.49507 |      0.01038 | -0.00244 |   0.02348 |
| winner_no_graph                |      1049 |             0.50423 |      0.00123 | -0.00400 |   0.00627 |
| archived_mlp                   |      1049 |             0.51045 |     -0.00500 | -0.01339 |   0.00353 |
| archived_transformer           |      1049 |             0.50226 |      0.00320 | -0.00761 |   0.01378 |
| archived_gnn                   |      1049 |             0.50165 |      0.00380 | -0.00563 |   0.01285 |
| archived_corrlstmgat           |      1049 |             0.50064 |      0.00482 | -0.00741 |   0.01700 |
| archived_fxrp                  |      1049 |             0.49873 |      0.00672 | -0.00897 |   0.02128 |
| archived_pit_main              |      1049 |             0.49975 |      0.00571 | -0.00677 |   0.01930 |
| archived_pit_arcfx_active_core |      1049 |             0.49965 |      0.00580 | -0.00669 |   0.01887 |
| original_legacy_matched_seeds  |      1049 |             0.49507 |      0.00941 | -0.00540 |   0.02466 |
| winner_no_graph_matched_seeds  |      1049 |             0.50423 |      0.00025 | -0.00448 |   0.00493 |

기본 paired circular block bootstrap은 20일 블록, 2,000회다. 5/60일 민감도는 CSV에 있다. 같은 날짜 통화들을 함께 뽑고 seed별 correctness를 먼저 평균했다. 따라서 CI는 고정된 seed 집합 조건부 시간 불확실성이고 seed 표준편차와 별개다. 선택 효과·기존 test 반복 열람을 보정하지 않았다. Archived baseline은 학습 종료일과 입력 정렬이 달라 인과적 공정 비교가 아니다.

## 8. 추천 구조와 손실

```json
{
  "name": "dense_signed",
  "encoder": "mlp",
  "loss": "bce",
  "graph": "dense"
}
```

Direction BCE: L_dir = mean[BCEWithLogits(u_i−u_USD, 1[r_i>0])], T=1. Dual 후보는 L=L_dir+λ mean[Huber(v_i−v_USD,r_i/σ_train;δ=1)], λ∈{0.03,0.1}. 수익률 예측은 σ_train(v_i−v_USD), σ_train은 모든 통화의 학습수익률 pooled std다. Direction logit은 수익률 크기가 아니다. 기존 loss가 선택되었다면 JSON의 legacy/active를 따른다(프로토콜 참조).
추천은 이번 제한된 development 탐색의 결과다. 유의한 전체 우위가 없으면 기존 모델을 대체하는 확정 개선으로 추천할 수 없다. 구조/손실 변경, 입력 정렬, 과거 데이터량 차이를 분리한 인과 추론은 이번 실험으로 불가능하다.

선택 모델이 BCE 단독이면 방향 연구 후보이며 수익률 크기를 출력하는 완성된 점예측 모델을 대체하지 않는다. 해당 후보의 logit에 RMSE를 붙이지 않았다. 수익률 점예측이 필수인 경우 dual-head 후보의 별도 RMSE와 방향 지표를 사용해야 하며, 좋은 방향 지표와 다른 모델의 좋은 RMSE를 결합해서는 안 된다.

## 9. 디펜스에서 주장할 수 있는 범위

- PIT 패널과 명시된 가용시각 가정 아래, currency strength 차분과 상태 의존 그래프를 유지한 제한적 모델·손실 탐색을 수행했다.
- 전체 날짜·통화 평가, 모든 후보·seed 로그, 재현 가능한 출력과 사후 유리한 결과를 분리해 제시할 수 있다.
- 전체 0.52 도달 여부와 baseline 대비 신뢰구간은 위 표의 실제 수치에 한해서 주장한다. 일부 연도/seed 최고값을 일반적인 예측 우위로 제시할 수 없다.
- 반복해서 본 보고 기간, 작은 후보/seed 수, 짧은 epoch 예산, vintage/시장시각 가정, 거래비용·실행 가능성 미평가를 한계로 명시해야 한다.

## 10. 실행과 산출물

완료 neural fits 54, 시도 54, 실패 0. Logistic baseline은 하나의 pipeline 안에서 통화별 추정기 10개를 fit했다. Majority 추정과 250-step smoke는 별도다.
최종 fit 학습: 2005-01-18 00:00:00–2019-08-23 00:00:00; checkpoint 검증: 2019-08-27 00:00:00–2021-10-05 00:00:00.

```powershell
python src/check_arcfx_search.py
python -u src/run_arcfx_search.py
python src/analyze_arcfx_search.py
```

전체 설정/손실/구조는 각 run의 result.json, 학습 추이는 history.csv, weights는 best.pt, 재개 상태는 resume.pt, 날짜·통화별 예측은 parquet다. 집계는 all_results.csv, aggregate.csv, all_seed_year.csv, all_currency.csv, paired_bootstrap.csv에 저장된다. SHA256 manifest와 원래 예측 비교표도 같은 결과 폴더에 있다.

상세 정의와 제한은 `docs/arcfx_search_protocol.md`를 참조한다.