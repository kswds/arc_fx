# Validation 구간으로 학습하고 같은 구간에서 평가한 결과

사용자 요청을 두 가지로 구분하여 모두 계산했다. (1) 이전 train에서 학습하고 validation으로 checkpoint를 선택한 기존 모델의 validation 성능, (2) 동일한 모델 조건으로 새로 초기화하여 validation label로 학습하고 정확히 같은 validation 날짜에서 평가한 성능.

**두 번째 실험은 fit/evaluation overlap 100%인 resubstitution이다. 이 수치를 독립 test 성능으로 해석할 수 없다. 첫 번째 수치도 checkpoint 선택에 사용한 validation 점수다.**

Decision date: 2019-08-27–2021-10-05; return 종료일: 2019-08-28–2021-10-06. 전체 527일, 10개 비USD 통화 = 5,270개 날짜·통화 표본. 보조 지표는 전체 55개 통화쌍 = 28,985개 표본.

기간·통화·seed를 제외하지 않았다. Seeds 0,1,2,3,4를 모두 평균했다. 각 seed의 fresh initialization과 shuffle은 graph/no-graph 간 동일하다. MLP hidden32/lookback10/dropout0.25, 전체 USD-target raw BCE, AdamW lr0.0003/weight_decay0.0001, batch256, max20 epochs, patience4를 유지했다. Checkpoint는 같은 학습·평가 구간의 sign-hit 최대, 동률 BCE 최소로 고른다. 원래 pre-validation train-only scaler/regime은 그대로 사용했다. Validation label만 optimizer 학습에 사용하며 이전 checkpoint로 fine-tuning하지 않았다.

## Validation에서 새로 학습 → 같은 validation에서 평가: 전체 10개 USD 대상

| policy              | model         | scope      |   mean_hit |   seed_std |   binary_accuracy |     bce |   seeds |
|:--------------------|:--------------|:-----------|-----------:|-----------:|------------------:|--------:|--------:|
| conservative_lag14  | dynamic_graph | all_10_usd |    0.57875 |    0.01118 |           0.58338 | 0.67400 |       5 |
| conservative_lag14  | mlp           | all_10_usd |    0.55734 |    0.00441 |           0.56167 | 0.68781 |       5 |
| conservative_lag14  | no_graph      | all_10_usd |    0.57814 |    0.01120 |           0.58266 | 0.67385 |       5 |
| conservative_lag14  | transformer   | all_10_usd |    0.53658 |    0.00967 |           0.54019 | 0.69031 |       5 |
| market_feed_assumed | dynamic_graph | all_10_usd |    0.57366 |    0.00729 |           0.57852 | 0.67677 |       5 |
| market_feed_assumed | mlp           | all_10_usd |    0.55548 |    0.00368 |           0.55981 | 0.68802 |       5 |
| market_feed_assumed | no_graph      | all_10_usd |    0.57340 |    0.00715 |           0.57825 | 0.67762 |       5 |
| market_feed_assumed | transformer   | all_10_usd |    0.52831 |    0.00667 |           0.53161 | 0.69170 |       5 |

## 동일 checkpoint의 전체 55개 strength-difference 통화쌍

| policy              | model         | scope        |   mean_hit |   seed_std |   binary_accuracy |     bce |   seeds |
|:--------------------|:--------------|:-------------|-----------:|-----------:|------------------:|--------:|--------:|
| conservative_lag14  | dynamic_graph | all_55_pairs |    0.52981 |    0.00671 |           0.53068 | 0.68923 |       5 |
| conservative_lag14  | mlp           | all_55_pairs |    0.51930 |    0.00618 |           0.52012 | 0.69168 |       5 |
| conservative_lag14  | no_graph      | all_55_pairs |    0.52953 |    0.00653 |           0.53038 | 0.68933 |       5 |
| conservative_lag14  | transformer   | all_55_pairs |    0.51313 |    0.00624 |           0.51381 | 0.69299 |       5 |
| market_feed_assumed | dynamic_graph | all_55_pairs |    0.52997 |    0.00414 |           0.53089 | 0.68948 |       5 |
| market_feed_assumed | mlp           | all_55_pairs |    0.51945 |    0.00409 |           0.52028 | 0.69156 |       5 |
| market_feed_assumed | no_graph      | all_55_pairs |    0.52978 |    0.00482 |           0.53070 | 0.68953 |       5 |
| market_feed_assumed | transformer   | all_55_pairs |    0.51017 |    0.00441 |           0.51080 | 0.69320 |       5 |

55개 통화쌍은 별도 학습하지 않고 latent logit 차분으로 생성했다. 주 학습 손실은 기존과 같이 USD 대비 10개 통화에 적용했다. 어느 통화쌍도 평가에서 제외하지 않았다.

## 비교: 기존 train에서 학습한 checkpoint를 validation 전체에서 평가

| policy              | model         |   mean_hit |   seed_std |   binary_accuracy |     bce |
|:--------------------|:--------------|-----------:|-----------:|------------------:|--------:|
| conservative_lag14  | dynamic_graph |    0.49898 |    0.00637 |           0.50307 | 0.69522 |
| conservative_lag14  | mlp           |    0.49636 |    0.00509 |           0.50197 | 0.69338 |
| conservative_lag14  | no_graph      |    0.49909 |    0.00686 |           0.50262 | 0.69553 |
| conservative_lag14  | transformer   |    0.50209 |    0.00910 |           0.50721 | 0.69340 |
| market_feed_assumed | dynamic_graph |    0.50159 |    0.00466 |           0.50653 | 0.70032 |
| market_feed_assumed | mlp           |    0.49662 |    0.00616 |           0.50281 | 0.69348 |
| market_feed_assumed | no_graph      |    0.49750 |    0.00373 |           0.50266 | 0.69504 |
| market_feed_assumed | transformer   |    0.49340 |    0.00777 |           0.49799 | 0.69359 |

## 같은 validation에서 추정한 상수 확률 baseline

| policy              |   mean_hit |   binary_accuracy |     bce |
|:--------------------|-----------:|------------------:|--------:|
| conservative_lag14  |    0.51214 |           0.51366 | 0.69262 |
| market_feed_assumed |    0.51214 |           0.51366 | 0.69262 |

Mean Hit는 np.sign(logit)==np.sign(raw return), Binary accuracy는 (logit>0)==(raw return>0)다. 0 target을 제외하지 않았다. 상수 확률 baseline은 p>0.5일 때 상승, 나머지는 하락이다. 방향 logits에 수익률 RMSE를 붙이지 않았다. 보수적/시장피드 입력 가정은 기존 후속 진단 프로토콜을 그대로 따른다.

## 재현

```powershell
python src/arcfx_validation_fit.py
```

run별 best.pt/resume.pt/history.csv와 예측을 보존했다. aggregate.csv, all_results.csv, currency_results.csv, pair_results.csv, original_checkpoint_validation_aggregate.csv에 전체 결과가 있다.