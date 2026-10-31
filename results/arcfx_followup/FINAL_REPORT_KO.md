# ARC_FX 후속 진단 및 평가 범위 탐색

사용자 허용에 따라 보고 기간·통화 조합을 탐색했다. 선택 범위에서는 모든 모델에 같은 날짜·대상·학습 규칙을 적용했다. 보고 label을 이용해 선택한 최대값은 해당 표본의 기술적 결과이며 일반화 성능의 독립 검증이 아니다. Seed는 0–4를 모두 평균했고 유리한 seed를 선택하지 않았다.

총 60개 model conditions: 새 neural 학습 52, 검증 후 재사용 8. 두 가용성 정책의 2×2가 40개, 동일 조건 MLP/Transformer가 20개다. Validation-only 보정은 모든 neural condition에 같은 절차로 60회 수행했다.

## 공통 조건과 가용성

A1=t−1/no graph, A2=t−1/dynamic, A3=t/no graph, A4=t/dynamic. Dense signed MLP, hidden 32, lookback 10, raw-label BCE, AdamW lr=0.0003, weight decay=0.0001, batch=256, epochs≤20, patience=4, validation sign-hit 최대/동률 BCE 최소. 모든 neural 비교는 같은 train/validation/report row, 손실, optimizer, shuffle seed, checkpoint 규칙을 사용한다.

보수적 정책: FX 파생특징에 14 calendar-day lag, 기존 PIT 비FX 특징에 1 panel-day 추가 lag, 이후 regime 재계산과 train-only scaling. 시장 피드 가정 정책: 기존 탐색의 t 입력을 그대로 사용한다.

연준은 H.10의 전주 daily rates를 월요일 16:15 ET(연방 공휴일이면 다음 영업일)에 공개한다. 따라서 당일 noon 관측과 당일 공개는 다르다. 2006-05~2009-01 weekly publication 중단, 후속 개정, 비FX 원천별 실제 공표시각은 완전한 vintage 감사를 마치지 않았다. 14일 buffer도 공개 일정 가정 하의 보수적 설계이며 실시간 가용성 인증은 아니다. [연준 공식 안내](https://www.federalreserve.gov/releases/h10/default.htm), [H.10 설명](https://www.federalreserve.gov/releases/h10/about.htm).

["2005-01-18 00:00:00", "2019-08-23 00:00:00"] 학습; ["2019-08-27 00:00:00", "2021-10-05 00:00:00"] checkpoint validation; ["2021-10-07", "2025-12-18"] return 종료일 보고.

## 전체 범위 결과

| policy              | condition         |   mean_hit |   seed_std |   binary_accuracy |     bce |     auc |
|:--------------------|:------------------|-----------:|-----------:|------------------:|--------:|--------:|
| conservative_lag14  | A1                |    0.50362 |    0.00781 |           0.50803 | 0.69507 | 0.50671 |
| conservative_lag14  | A2                |    0.50309 |    0.00828 |           0.50761 | 0.69541 | 0.50687 |
| conservative_lag14  | A3                |    0.49964 |    0.00791 |           0.50383 | 0.69576 | 0.50506 |
| conservative_lag14  | A4                |    0.50147 |    0.00888 |           0.50606 | 0.69540 | 0.50675 |
| conservative_lag14  | half              |    0.51211 |  nan       |           0.51992 | 0.69315 | 0.50000 |
| conservative_lag14  | majority          |    0.50982 |  nan       |           0.51554 | 0.69292 | 0.50325 |
| conservative_lag14  | mlp               |    0.50675 |    0.00552 |           0.51249 | 0.69282 | 0.50310 |
| conservative_lag14  | train_probability |    0.50982 |  nan       |           0.51554 | 0.69292 | 0.50325 |
| conservative_lag14  | transformer       |    0.50002 |    0.00624 |           0.50532 | 0.69336 | 0.49712 |
| market_feed_assumed | A1                |    0.50328 |    0.00567 |           0.50856 | 0.69394 | 0.50543 |
| market_feed_assumed | A2                |    0.50112 |    0.00378 |           0.50620 | 0.69522 | 0.50413 |
| market_feed_assumed | A3                |    0.50503 |    0.00168 |           0.51052 | 0.69326 | 0.50900 |
| market_feed_assumed | A4                |    0.50545 |    0.00231 |           0.51049 | 0.69481 | 0.50781 |
| market_feed_assumed | half              |    0.51211 |  nan       |           0.51992 | 0.69315 | 0.50000 |
| market_feed_assumed | majority          |    0.50982 |  nan       |           0.51554 | 0.69292 | 0.50325 |
| market_feed_assumed | mlp               |    0.50391 |    0.00936 |           0.51003 | 0.69301 | 0.49933 |
| market_feed_assumed | train_probability |    0.50982 |  nan       |           0.51554 | 0.69292 | 0.50325 |
| market_feed_assumed | transformer       |    0.49865 |    0.00463 |           0.50324 | 0.69349 | 0.49811 |

주 지표는 기존 np.sign(logit)==np.sign(return)이며 0 return도 분모에 남긴다. Binary accuracy는 (logit>0)==(return>0)이다. Probability baseline은 p>0.5일 때 상승, 나머지는 하락으로 예측한다. p=0.5의 logit=0을 모델의 3-way zero 예측과 혼동하지 않도록 baseline 방향을 명시했다. 모든 출력은 방향 logit이며 RMSE를 계산하지 않았다.

## 1. 입력 정렬 효과

| policy              | comparison   |   effect |   ci_low |   ci_high |   seed_effect_std |
|:--------------------|:-------------|---------:|---------:|----------:|------------------:|
| conservative_lag14  | A3-A1        | -0.39847 | -1.26416 |   0.44233 |           1.24227 |
| conservative_lag14  | A4-A2        | -0.16206 | -0.95524 |   0.62355 |           1.19167 |
| market_feed_assumed | A3-A1        |  0.17541 | -0.43279 |   0.77989 |           0.53186 |
| market_feed_assumed | A4-A2        |  0.43279 | -0.24600 |   1.13270 |           0.49629 |

단위는 %p다. 보수적 정책에서 t는 14일 이상 지난 FX snapshot과 추가 lag를 준 비FX snapshot의 최신 row를 뜻한다. 기존 당일 spot 입력의 효과는 시장 피드 가정 정책에서만 재현한다.

conservative_lag14: A3-A1 = -0.39847 %p, 95% CI [-1.26416, 0.44233]; 차이의 신뢰구간이 0을 포함하여 우위 미확인.
conservative_lag14: A4-A2 = -0.16206 %p, 95% CI [-0.95524, 0.62355]; 차이의 신뢰구간이 0을 포함하여 우위 미확인.
market_feed_assumed: A3-A1 = +0.17541 %p, 95% CI [-0.43279, 0.77989]; 차이의 신뢰구간이 0을 포함하여 우위 미확인.
market_feed_assumed: A4-A2 = +0.43279 %p, 95% CI [-0.24600, 1.13270]; 차이의 신뢰구간이 0을 포함하여 우위 미확인.

## 2. 최신 입력의 실제 가용성

원천 코드의 availability audit상 위반은 0건이나, H.10 공식 공개 지연이 그 audit의 FX 파생특징에 반영되지 않았다. 원래 50.5%는 시장 피드 가정에 의존한다. 신규 보수적 결과도 역사적 release별 인증 완료 결과로 해석해서는 안 된다.

## 3. 통화별 방향 빈도 이상의 정보

| policy              | comparison            |   effect |   ci_low |   ci_high |
|:--------------------|:----------------------|---------:|---------:|----------:|
| conservative_lag14  | A4-train_probability  | -0.83508 | -2.10891 |   0.38894 |
| conservative_lag14  | A4-A4:train_mean      |  0.71497 | -0.38132 |   1.86482 |
| conservative_lag14  | A4-A4:validation_mean | -0.11821 | -1.04480 |   0.73408 |
| market_feed_assumed | A4-train_probability  | -0.43661 | -1.76397 |   0.77412 |
| market_feed_assumed | A4-A4:train_mean      |  0.01716 | -1.04485 |   0.97426 |
| market_feed_assumed | A4-A4:validation_mean | -0.42517 | -2.17169 |   1.08508 |

통화별 train 상수 확률, 학습/validation 평균 logit 고정, 실제 동적 예측을 비교했다. BCE 개선은 paired_effects.csv의 metric=bce에서 양수가 개선이다. 통화별 AUC와 residual correlation은 currency_metrics_confusion.csv, residual_correlations.csv에 있다. Circular shift는 모든 통화에 같은 shift를 주며 ±60일 이내를 제외했다. 비정상성 때문에 엄밀한 유의성 검정으로 해석하지 않는다.

conservative_lag14: A4-train_probability = -0.83508 %p, 95% CI [-2.10891, 0.38894]; 차이의 신뢰구간이 0을 포함하여 우위 미확인.
conservative_lag14: A4-train_probability = -0.00248 BCE 감소량, 95% CI [-0.00377, -0.00115]; 조건부 신뢰구간은 0을 포함하지 않음(탐색 선택 효과 미보정).
market_feed_assumed: A4-train_probability = -0.43661 %p, 95% CI [-1.76397, 0.77412]; 차이의 신뢰구간이 0을 포함하여 우위 미확인.
market_feed_assumed: A4-train_probability = -0.00189 BCE 감소량, 95% CI [-0.00336, -0.00037]; 조건부 신뢰구간은 0을 포함하지 않음(탐색 선택 효과 미보정).

## 4. 확률 보정과 구분 능력

AUC는 구분 능력, BCE/Brier/reliability는 확률 품질을 본다. z′=a z+b_i, a>0, b_USD=0 보정은 validation BCE로만 학습했다. 보정은 새로운 시점별 정보를 추가하지 않는다. 보정 전후는 별도 variant로 저장했다. 전체 및 통화별 수치는 metrics.csv/currency_metrics_confusion.csv, 그림은 reliability.png/confusion_A4.png를 참조한다.

conservative_lag14: A4_calibrated-A4 = -0.33556 %p, 95% CI [-2.14690, 1.26406]; 차이의 신뢰구간이 0을 포함하여 우위 미확인.
conservative_lag14: A4_calibrated-A4 = +0.00148 BCE 감소량, 95% CI [-0.00034, 0.00325]; 차이의 신뢰구간이 0을 포함하여 우위 미확인.
market_feed_assumed: A4_calibrated-A4 = -0.87512 %p, 95% CI [-2.91349, 1.11187]; 차이의 신뢰구간이 0을 포함하여 우위 미확인.
market_feed_assumed: A4_calibrated-A4 = +0.00117 BCE 감소량, 95% CI [-0.00079, 0.00309]; 차이의 신뢰구간이 0을 포함하여 우위 미확인.

## 5. 동일 조건에서 dynamic graph 추가 가치

| policy              |   effect |   ci_low |   ci_high |   seed_effect_std |
|:--------------------|---------:|---------:|----------:|------------------:|
| conservative_lag14  |  0.18303 | -0.18308 |   0.56058 |           0.47413 |
| market_feed_assumed |  0.04194 | -0.43470 |   0.48437 |           0.22240 |

No-graph는 동일 초기화된 모든 모듈을 유지하고 메시지를 0으로 만든 학습 variant다. Gate와 direct head는 각 노드 h만 사용하고 foreign-node pooling 우회 경로는 없다. 사전 정의 common/global/regime 특징과 USD gauge는 공통으로 남는다. 따라서 no-graph도 공통 시장 특징은 사용할 수 있다.

conservative_lag14: A4-A3 = +0.18303 %p, 95% CI [-0.18308, 0.56058]; 차이의 신뢰구간이 0을 포함하여 우위 미확인.
market_feed_assumed: A4-A3 = +0.04194 %p, 95% CI [-0.43470, 0.48437]; 차이의 신뢰구간이 0을 포함하여 우위 미확인.

## 6. Graph가 바꾼 예측의 정답/오답

| policy              | intervention   |   sign_changes |   correct_to_wrong |   wrong_to_correct |   abs_logit_change |   message_rms |
|:--------------------|:---------------|---------------:|-------------------:|-------------------:|-------------------:|--------------:|
| conservative_lag14  | mean_adjacency |           1283 |                642 |                634 |            0.00932 |       0.06087 |
| conservative_lag14  | zero_message   |           1237 |                615 |                614 |            0.00931 |       0.00000 |
| market_feed_assumed | mean_adjacency |           5352 |               2685 |               2623 |            0.04539 |       0.06230 |
| market_feed_assumed | zero_message   |           4869 |               2390 |               2444 |            0.04179 |       0.00000 |

이 표는 원래 graph 예측에서 intervention으로 변경했을 때의 전환이다. correct_to_wrong가 크면 원래 graph가 맞힌 사례를 제거한 것이다. 고정 평균 adjacency는 학습 구간에서만 계산했으며 static graph 재학습과 같지 않다. zero_message는 실제 집계 메시지 m을 0으로 만들고 direct h/gate/weights는 유지한다. per-date/currency 기여는 각 run의 graph_contributions.parquet에 저장했다.

## 7. 기존 약 50.5% 해석과 유리한 범위

기존 전체 결과를 그대로 일반적인 예측 우위로 주장할 수 없다. 동일 조건의 no-graph·MLP·Transformer·train 빈도 baseline과 paired 결과를 우선 봐야 한다. 아래 범위는 사용자 요청에 따라 보고 구간에서 선택한 결과이며 그 선택 효과를 bootstrap이 보정하지 않는다.

### conservative_lag14: 모든 지정 baseline 대비 최소 차이를 최대화한 조합

| candidate     | start      | end        |   n_dates | targets   |   mean_hit |   minimum_advantage_pp |   baseline_A3 |   baseline_mlp |   baseline_transformer |   baseline_train_probability |
|:--------------|:-----------|:-----------|----------:|:----------|-----------:|-----------------------:|--------------:|---------------:|-----------------------:|-----------------------------:|
| A4            | 2024-07-01 | 2024-12-31 |       126 | KRW/NOK   |    0.55397 |                2.06349 |       0.53333 |        0.53016 |                0.50317 |                      0.49206 |
| A4            | 2023-01-03 | 2023-12-29 |       249 | CAD/NZD   |    0.53574 |                1.92771 |       0.51647 |        0.50040 |                0.50683 |                      0.49398 |
| A4_calibrated | 2024-01-02 | 2024-12-31 |       251 | GBP/CHF   |    0.50518 |                1.67331 |       0.48765 |        0.48127 |                0.48845 |                      0.43426 |

절대 Hit 최고는 A4_calibrated, JPY/NZD, 2021-10-07–2022-06-30의 0.59011이며 지정 baseline 중 최고 대비 차이는 -0.43956%p다. 절대 Hit 최고와 baseline 대비 우위 최고는 서로 다른 목적이다.

선택한 최고 조합의 조건부 bootstrap (범위 선택 효과를 보정하지 않음):

| comparator        |   effect_pp |   ci_low_pp |   ci_high_pp |   winning_seeds |
|:------------------|------------:|------------:|-------------:|----------------:|
| A3                |     2.06349 |     0.95238 |      3.33333 |               1 |
| mlp               |     2.38095 |    -2.85714 |      7.61905 |               3 |
| transformer       |     5.07937 |     0.15873 |     10.47619 |               3 |
| train_probability |     6.19048 |    -5.40079 |     17.93651 |               5 |

같은 조합의 모든 seed 결과:

|    seed |   mean_hit |   baseline_A3 |   baseline_mlp |   baseline_transformer |   baseline_train_probability |
|--------:|-----------:|--------------:|---------------:|-----------------------:|-----------------------------:|
| 0.00000 |    0.57143 |       0.58730 |        0.50794 |                0.50794 |                      0.49206 |
| 1.00000 |    0.50000 |       0.50000 |        0.46825 |                0.56349 |                      0.49206 |
| 2.00000 |    0.55556 |       0.43651 |        0.62698 |                0.50794 |                      0.49206 |
| 3.00000 |    0.52381 |       0.52381 |        0.55556 |                0.52381 |                      0.49206 |
| 4.00000 |    0.61905 |       0.61905 |        0.49206 |                0.41270 |                      0.49206 |

### market_feed_assumed: 모든 지정 baseline 대비 최소 차이를 최대화한 조합

| candidate   | start      | end        |   n_dates | targets   |   mean_hit |   minimum_advantage_pp |   baseline_A3 |   baseline_mlp |   baseline_transformer |   baseline_train_probability |
|:------------|:-----------|:-----------|----------:|:----------|-----------:|-----------------------:|--------------:|---------------:|-----------------------:|-----------------------------:|
| A4          | 2024-07-01 | 2024-12-31 |       126 | AUD/NZD   |    0.53810 |                2.53968 |       0.51270 |        0.50317 |                0.49841 |                      0.48413 |
| A4          | 2025-01-02 | 2025-12-18 |       242 | NZD/SEK   |    0.51901 |                2.31405 |       0.49587 |        0.47025 |                0.49174 |                      0.43802 |
| A4          | 2023-01-03 | 2024-06-28 |       374 | CAD/CHF   |    0.52781 |                2.08556 |       0.50321 |        0.50374 |                0.50695 |                      0.48396 |

절대 Hit 최고는 A4_calibrated, JPY/NZD, 2021-10-07–2022-06-30의 0.59341이며 지정 baseline 중 최고 대비 차이는 +0.00000%p다. 절대 Hit 최고와 baseline 대비 우위 최고는 서로 다른 목적이다.

선택한 최고 조합의 조건부 bootstrap (범위 선택 효과를 보정하지 않음):

| comparator        |   effect_pp |   ci_low_pp |   ci_high_pp |   winning_seeds |
|:------------------|------------:|------------:|-------------:|----------------:|
| A3                |     2.53968 |    -0.47619 |      5.23810 |               3 |
| mlp               |     3.49206 |    -1.43254 |      8.57143 |               4 |
| transformer       |     3.96825 |    -0.79365 |      8.09524 |               4 |
| train_probability |     5.39683 |    -7.93651 |     18.41667 |               5 |

같은 조합의 모든 seed 결과:

|    seed |   mean_hit |   baseline_A3 |   baseline_mlp |   baseline_transformer |   baseline_train_probability |
|--------:|-----------:|--------------:|---------------:|-----------------------:|-----------------------------:|
| 0.00000 |    0.53175 |       0.53968 |        0.47619 |                0.47619 |                      0.48413 |
| 1.00000 |    0.50000 |       0.48413 |        0.46825 |                0.42063 |                      0.48413 |
| 2.00000 |    0.50000 |       0.52381 |        0.51587 |                0.51587 |                      0.48413 |
| 3.00000 |    0.57937 |       0.52381 |        0.54762 |                0.52381 |                      0.48413 |
| 4.00000 |    0.57937 |       0.49206 |        0.50794 |                0.55556 |                      0.48413 |

평가 universe는 최대 55개 strength-difference cross rates까지 진단했다. USD 대상 1,023개 부분집합, 37개 기간, 2개 보정 상태, 2개 가용성 정책을 포함해 159,544개 행(158,064개 고유 조합)을 계산했다. 많은 조합 중 최대값을 고른 것이므로 독립 검증과 구분해야 한다. 보정한 후보 행은 상대 neural 모델도 동일하게 보정한 결과와 비교한다.

Cross-rate i/j의 logit은 z_i−z_j, 실제 target은 r_i,USD−r_j,USD다. 예를 들어 KRW/NOK는 NOK 대비 KRW 절상 방향을 예측한다. 모델 graph는 계속 11-node이며 학습 타깃은 동일한 USD 대비 10개 통화다. 선택 후 학습 대상을 바꿔 재학습한 결과가 아니다. 전체 조합은 ALL_scope_search.parquet, 세 목적별 순위는 selected_scopes_POSTHOC.csv, validation만으로 통화를 선택한 결과는 validation_selected_currencies.csv다.

## 8. 다음 실험 (최대 2개)

1. 가장 유망한 통화·기간 규칙을 지금 고정하고, 아직 보지 않은 이후 기간 또는 사전에 고정한 walk-forward 일정에서 모든 모델을 같은 방식으로 재학습·평가한다. 보고 label로 고른 성능이 미래에도 유지되는지 확인한다.
2. 당일 spot 피드의 실제 timestamp 또는 FX vintage archive를 확보해 공개 가능 시각을 재구성하고, 그 정보집합에서 dynamic graph와 no-graph를 동일 조건으로 다시 비교한다. 가용성 불확실성이 architecture 효과와 섞이지 않게 한다.

## 재현과 파일

```powershell
python src/arcfx_followup.py
python src/arcfx_followup.py --analysis-only
```

manifest.json과 configs/arcfx_followup.json에 정의/해시/가용성 가정을 저장했다. 각 run에 초기화 해시·재사용 근거·best.pt·resume.pt·학습 추이·예측이 있다. 재사용은 원본 data/model hash, loss/graph/seed/설정, 예측 재계산 일치를 확인했다. 중단 후 완료 run을 건너뛰고 epoch checkpoint에서 재개한다. paired_effects.csv는 공통 5 seeds의 날짜별 평균 차이를 날짜 단위 5/20/60일 circular block으로 2,000회 재표집했다. Seed를 시장 표본처럼 취급하지 않았다.

## 시점별 구분 능력과 permutation 수치

conservative_lag14: 통화 내 AUC 평균 0.50411 (보정 후 0.50411), logit/label residual correlation 평균 0.00964. 원래 Hit 0.50147, calibration Hit 0.49811; 원래 BCE 0.69540, calibration BCE 0.69392. 실제 정렬 Hit의 seed 평균 0.50147, shifted prediction Hit 평균 0.49954.

통화 내 구분 능력이 0.5 근처에 머문다. 보정으로 확률/기본 방향을 바꿀 수 있어도, 이를 강한 날짜별 예측 신호의 증거로 설명하기는 어렵다. 보정 문제와 약한 구분 능력을 함께 검토해야 한다.

실제 메시지의 정답 순기여(원래 graph 정답→메시지 제거 오답 수 − 반대 전환 수)는 5 seeds 전체에서 +1건이다. 양수는 graph가 더 많이 맞힌 경우, 음수는 메시지를 제거한 쪽이 더 많이 맞힌 경우다. 반복 seed는 독립 시장 관측 수로 해석하지 않는다.

market_feed_assumed: 통화 내 AUC 평균 0.50607 (보정 후 0.50607), logit/label residual correlation 평균 0.00883. 원래 Hit 0.50545, calibration Hit 0.49670; 원래 BCE 0.69481, calibration BCE 0.69364. 실제 정렬 Hit의 seed 평균 0.50545, shifted prediction Hit 평균 0.50196.

통화 내 구분 능력이 0.5 근처에 머문다. 보정으로 확률/기본 방향을 바꿀 수 있어도, 이를 강한 날짜별 예측 신호의 증거로 설명하기는 어렵다. 보정 문제와 약한 구분 능력을 함께 검토해야 한다.

실제 메시지의 정답 순기여(원래 graph 정답→메시지 제거 오답 수 − 반대 전환 수)는 5 seeds 전체에서 -54건이다. 양수는 graph가 더 많이 맞힌 경우, 음수는 메시지를 제거한 쪽이 더 많이 맞힌 경우다. 반복 seed는 독립 시장 관측 수로 해석하지 않는다.
