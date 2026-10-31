# Top-k / Dense signed / Validation-selected Signed soft-threshold 비교

두 본체 각각에서 graph 선택 연산만 변경했다. 기간·통화는 성능을 보고 고르지 않았다. 전체 10개 USD 대비 대상, seeds 0–4, 동일한 test 1,049일을 사용했다.

## 비교 조건

- Compact: 최근 temporal MLP hidden32, QK score, direct + graph residual strength, 전체 raw-label BCE.
- Original: 기존 채널별 GRU hidden48, static + QK + macro low-rank score, 원래 strength/gate/message 경로, 기존 Q40 masked directional core + component/graph penalties.
- 두 본체 모두 이번 실행에서는 공통 max20 epochs/patience4/batch256/AdamW lr0.0003/weight decay0.0001/gradient clip1, validation Mean Hit 최대/동률 validation 학습목적 최소를 적용했다. 원래 pit_main의 80 epochs/patience10/batch128/mse_hit checkpoint/LR scheduler를 그대로 재현한 실험은 아니다.
- 입력은 공급된 PIT 패널의 decision t까지 10개 row로 통일했다. 원래 sequence 코드의 추가 t−1 지연은 이번 통제 비교에서 사용하지 않았다. 당일 FX의 실시간 공개/시장 피드 가용성 가정은 기존 논의와 같다. 추가 14일 lag는 이번 실험에 도입하지 않았다.
- Top-k=3, dense와 Top-k는 softmax(abs score) × tanh(score). Soft-threshold는 sign(score) relu(abs(score)−tau), diagonal=0, 행별 abs-sum을 max(1,sum)으로 제한.
- Original의 static reference adjacency/graph regularizer 계수/edge-dropout0.05는 세 selector에서 동일하게 유지했다. Static reference의 기존 Top-k 처리도 공통으로 고정했다. Compact는 기존처럼 edge dropout과 추가 graph penalty가 없다.
- tau grid=[0,0.025,0.05,0.1,0.2]. 각 본체·분할에서 seeds 0–4의 validation 평균 Hit로 공통 tau 하나를 선택한다. 동률이면 평균 validation 목적이 작고, 그다음 tau가 작은 후보를 선택한다. Test는 tau/checkpoint 선택에 사용하지 않았다.

## 기간

| role                                          |   n_dates | decision_start   | decision_end   | return_end_start   | return_end_end   |
|:----------------------------------------------|----------:|:-----------------|:---------------|:-------------------|:-----------------|
| train (both)                                  |      3145 | 2005-01-18       | 2017-07-24     | 2005-01-19         | 2017-07-25       |
| validation: pit_original                      |      1048 | 2017-07-25       | 2021-10-05     | 2017-07-26         | 2021-10-06       |
| validation: overlap50                         |      1048 | 2019-08-30       | 2023-11-09     | 2019-09-03         | 2023-11-13       |
| test (both)                                   |      1049 | 2021-10-06       | 2025-12-17     | 2021-10-07         | 2025-12-18       |
| test early half / overlapping under overlap50 |       524 | 2021-10-06       | 2023-11-09     | 2021-10-07         | 2023-11-13       |
| test late half / not overlapping              |       525 | 2023-11-13       | 2025-12-17     | 2023-11-14         | 2025-12-18       |

Overlap50의 공통 날짜는 524일: validation의 50.00%, test의 49.95%다. Validation을 뒤로 이동했기 때문에 train과 shifted validation 사이 524개 decision row는 학습 label이나 validation label로 사용하지 않는다. Lookback의 과거 입력으로는 사용 가능하다.

Overlap은 optimizer가 test label로 직접 학습한다는 뜻이 아니다. 그러나 겹친 label로 checkpoint와 tau를 고르므로 test 전체를 독립 일반화 성능으로 해석할 수 없다. 뒤쪽 525일의 평가도 함께 보고한다. 이미 반복 확인한 역사적 기간이라는 제한도 남는다.

## 요청한 모델별 최고 성능: validation-selected 설정에서 test 최고 seed

| backbone       | protocol     | selector       |       tau |   best_test_hit |   best_test_seed |   best_epoch |   mean_hit |   seed_std |
|:---------------|:-------------|:---------------|----------:|----------------:|-----------------:|-------------:|-----------:|-----------:|
| compact_mlp    | overlap50    | dense_signed   | nan       |         0.52031 |                3 |            5 |    0.51287 |    0.00447 |
| compact_mlp    | overlap50    | soft_threshold |   0.02500 |         0.51983 |                3 |            4 |    0.51003 |    0.00701 |
| compact_mlp    | overlap50    | topk           | nan       |         0.51973 |                3 |            5 |    0.51087 |    0.00679 |
| compact_mlp    | pit_original | dense_signed   | nan       |         0.51201 |                1 |            1 |    0.50650 |    0.00390 |
| compact_mlp    | pit_original | soft_threshold |   0.00000 |         0.51192 |                1 |            1 |    0.50604 |    0.00525 |
| compact_mlp    | pit_original | topk           | nan       |         0.51840 |                3 |            4 |    0.50845 |    0.00718 |
| original_arcfx | overlap50    | dense_signed   | nan       |         0.51192 |                3 |            3 |    0.50663 |    0.00458 |
| original_arcfx | overlap50    | soft_threshold |   0.20000 |         0.51192 |                3 |            3 |    0.50797 |    0.00390 |
| original_arcfx | overlap50    | topk           | nan       |         0.51277 |                3 |            3 |    0.50843 |    0.00370 |
| original_arcfx | pit_original | dense_signed   | nan       |         0.50829 |                3 |            2 |    0.50288 |    0.00903 |
| original_arcfx | pit_original | soft_threshold |   0.00000 |         0.50991 |                3 |            3 |    0.50376 |    0.00937 |
| original_arcfx | pit_original | topk           | nan       |         0.51277 |                3 |            3 |    0.50444 |    0.01016 |

best_test_hit는 선택된 설정의 공통 다섯 seed 중 test Hit 최대값이다. 이 seed는 test를 보고 선택한 사후 최고값이며, 대표 성능은 mean_hit와 seed_std로 함께 확인한다. Tau를 test 최고값으로 다시 바꾸거나 epoch를 test로 선택하지 않았다.

## Test의 공통 앞·뒤 구간 결과

| backbone       | protocol     | selector       |       tau | subset                 |   mean_hit |   seed_std |   best_test_hit |   best_test_seed |
|:---------------|:-------------|:---------------|----------:|:-----------------------|-----------:|-----------:|----------------:|-----------------:|
| compact_mlp    | overlap50    | dense_signed   | nan       | test_common_early_half |    0.51775 |    0.01082 |         0.53378 |                3 |
| compact_mlp    | overlap50    | dense_signed   | nan       | test_common_late_half  |    0.50800 |    0.00819 |         0.52076 |                2 |
| compact_mlp    | overlap50    | soft_threshold |   0.02500 | test_common_early_half |    0.51687 |    0.01008 |         0.53206 |                3 |
| compact_mlp    | overlap50    | soft_threshold |   0.02500 | test_common_late_half  |    0.50320 |    0.00640 |         0.50971 |                2 |
| compact_mlp    | overlap50    | topk           | nan       | test_common_early_half |    0.51534 |    0.01178 |         0.53244 |                3 |
| compact_mlp    | overlap50    | topk           | nan       | test_common_late_half  |    0.50640 |    0.00804 |         0.51848 |                2 |
| compact_mlp    | pit_original | dense_signed   | nan       | test_common_early_half |    0.50935 |    0.00892 |         0.51947 |                1 |
| compact_mlp    | pit_original | dense_signed   | nan       | test_common_late_half  |    0.50366 |    0.01002 |         0.51867 |                2 |
| compact_mlp    | pit_original | soft_threshold |   0.00000 | test_common_early_half |    0.51111 |    0.00829 |         0.51966 |                1 |
| compact_mlp    | pit_original | soft_threshold |   0.00000 | test_common_late_half  |    0.50099 |    0.00513 |         0.50800 |                2 |
| compact_mlp    | pit_original | topk           | nan       | test_common_early_half |    0.51153 |    0.01363 |         0.52958 |                3 |
| compact_mlp    | pit_original | topk           | nan       | test_common_late_half  |    0.50537 |    0.00981 |         0.52000 |                2 |
| original_arcfx | overlap50    | dense_signed   | nan       | test_common_early_half |    0.50489 |    0.00549 |         0.51107 |                1 |
| original_arcfx | overlap50    | dense_signed   | nan       | test_common_late_half  |    0.50838 |    0.00783 |         0.51981 |                3 |
| original_arcfx | overlap50    | soft_threshold |   0.20000 | test_common_early_half |    0.50588 |    0.00535 |         0.51202 |                1 |
| original_arcfx | overlap50    | soft_threshold |   0.20000 | test_common_late_half  |    0.51006 |    0.00587 |         0.52000 |                3 |
| original_arcfx | overlap50    | topk           | nan       | test_common_early_half |    0.50508 |    0.00560 |         0.51088 |                1 |
| original_arcfx | overlap50    | topk           | nan       | test_common_late_half  |    0.51177 |    0.00558 |         0.52133 |                3 |
| original_arcfx | pit_original | dense_signed   | nan       | test_common_early_half |    0.49863 |    0.01326 |         0.50954 |                1 |
| original_arcfx | pit_original | dense_signed   | nan       | test_common_late_half  |    0.50712 |    0.00852 |         0.52057 |                3 |
| original_arcfx | pit_original | soft_threshold |   0.00000 | test_common_early_half |    0.50023 |    0.01300 |         0.50954 |                1 |
| original_arcfx | pit_original | soft_threshold |   0.00000 | test_common_late_half  |    0.50728 |    0.00704 |         0.51676 |                3 |
| original_arcfx | pit_original | topk           | nan       | test_common_early_half |    0.50061 |    0.01335 |         0.51088 |                1 |
| original_arcfx | pit_original | topk           | nan       | test_common_late_half  |    0.50827 |    0.00878 |         0.52133 |                3 |

앞 524일과 뒤 525일은 원래 분할에서도 동일한 날짜로 평가했다. 따라서 두 protocol의 부분구간 점수를 날짜가 다른 점수와 혼동하지 않고 비교할 수 있다.

## Validation tau 탐색 전체 결과

| backbone       | protocol     |     tau |   validation_hit |   seed_std |   objective |
|:---------------|:-------------|--------:|-----------------:|-----------:|------------:|
| compact_mlp    | overlap50    | 0.00000 |          0.50553 |    0.00525 |     0.69446 |
| compact_mlp    | overlap50    | 0.02500 |          0.50662 |    0.00326 |     0.69490 |
| compact_mlp    | overlap50    | 0.05000 |          0.50574 |    0.00580 |     0.69425 |
| compact_mlp    | overlap50    | 0.10000 |          0.50506 |    0.00612 |     0.69455 |
| compact_mlp    | overlap50    | 0.20000 |          0.50468 |    0.00407 |     0.69438 |
| compact_mlp    | pit_original | 0.00000 |          0.49927 |    0.00182 |     0.69627 |
| compact_mlp    | pit_original | 0.02500 |          0.49884 |    0.00234 |     0.69634 |
| compact_mlp    | pit_original | 0.05000 |          0.49889 |    0.00259 |     0.69628 |
| compact_mlp    | pit_original | 0.10000 |          0.49788 |    0.00290 |     0.69553 |
| compact_mlp    | pit_original | 0.20000 |          0.49760 |    0.00239 |     0.69556 |
| original_arcfx | overlap50    | 0.00000 |          0.50618 |    0.00736 |     0.69379 |
| original_arcfx | overlap50    | 0.02500 |          0.50599 |    0.00697 |     0.69382 |
| original_arcfx | overlap50    | 0.05000 |          0.50603 |    0.00650 |     0.69376 |
| original_arcfx | overlap50    | 0.10000 |          0.50597 |    0.00714 |     0.69455 |
| original_arcfx | overlap50    | 0.20000 |          0.50647 |    0.00660 |     0.69372 |
| original_arcfx | pit_original | 0.00000 |          0.50359 |    0.00875 |     0.69491 |
| original_arcfx | pit_original | 0.02500 |          0.50359 |    0.00852 |     0.69492 |
| original_arcfx | pit_original | 0.05000 |          0.50342 |    0.00798 |     0.69496 |
| original_arcfx | pit_original | 0.10000 |          0.50357 |    0.00793 |     0.69489 |
| original_arcfx | pit_original | 0.20000 |          0.50307 |    0.00817 |     0.69491 |

## 동일 seed·동일 날짜의 평균 효과와 조건부 시간 불확실성

| backbone       | protocol     | comparison                  |   mean_effect_pp |   ci_low_pp |   ci_high_pp |   seed_effect_std_pp |
|:---------------|:-------------|:----------------------------|-----------------:|------------:|-------------:|---------------------:|
| compact_mlp    | pit_original | dense_signed-topk           |         -0.19447 |    -0.57960 |      0.17350 |              0.62955 |
| compact_mlp    | pit_original | soft_threshold-topk         |         -0.24023 |    -0.58918 |      0.11254 |              0.43646 |
| compact_mlp    | pit_original | soft_threshold-dense_signed |         -0.04576 |    -0.44423 |      0.33365 |              0.38289 |
| compact_mlp    | overlap50    | dense_signed-topk           |          0.20019 |    -0.07626 |      0.47092 |              0.35509 |
| compact_mlp    | overlap50    | soft_threshold-topk         |         -0.08389 |    -0.46330 |      0.31654 |              0.37113 |
| compact_mlp    | overlap50    | soft_threshold-dense_signed |         -0.28408 |    -0.60639 |      0.06673 |              0.33238 |
| original_arcfx | pit_original | dense_signed-topk           |         -0.15634 |    -0.40229 |      0.07440 |              0.20761 |
| original_arcfx | pit_original | soft_threshold-topk         |         -0.06864 |    -0.15825 |      0.02293 |              0.12564 |
| original_arcfx | pit_original | soft_threshold-dense_signed |          0.08770 |    -0.15067 |      0.32602 |              0.12122 |
| original_arcfx | overlap50    | dense_signed-topk           |         -0.17922 |    -0.35848 |     -0.00377 |              0.30321 |
| original_arcfx | overlap50    | soft_threshold-topk         |         -0.04576 |    -0.21735 |      0.12583 |              0.06754 |
| original_arcfx | overlap50    | soft_threshold-dense_signed |          0.13346 |    -0.09533 |      0.35653 |              0.34896 |

Paired circular block bootstrap은 같은 날짜의 통화를 함께 평균하고 공통 seed 평균 차이에 적용했다. Block=20, 2,000회이며 5/60 민감도를 CSV에 저장했다. 이 CI는 모델/seed 집합 조건부 시간 불확실성으로, 최고 seed 선택·tau 선택·의도적 overlap을 보정하지 않는다.

## 실행·검증·산출물

물리적 optimizer trajectories는 70개다(본체 2 × 후보 7 × seed 5). 각 trajectory에서 두 validation protocol의 checkpoint/early-stop 상태를 독립적으로 추적했다. 고정 learning rate이며 validation이 gradient나 optimizer update를 바꾸지 않으므로, 같은 epoch의 학습 가중치는 두 protocol에서 같다. 한 protocol이 멈추면 해당 선택 상태를 고정하고 다른 protocol만 계속 평가한다. 논리적 validation 조건은 140개이며, tau 선택 후 test 조건은 60개다.

각 trajectory에 history.csv/resume.pt와 protocol별 best.pt/validation predictions를 저장했다. 선택된 60개 test 결과에는 전체·앞·뒤 구간의 날짜/통화별 예측을 저장했다. 초기화 해시와 source/data manifest를 함께 저장했다.

```powershell
python src/arcfx_selector_compare.py
python src/arcfx_selector_compare.py --analysis-only
python src/check_arcfx_selector_compare.py
```
