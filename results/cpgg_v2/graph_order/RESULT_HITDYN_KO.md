# Mean Hit 기준 Dynamic graph: 하루 연결 수가 날마다 달라지는 모델 (seed 0)

**구조와 임계값을 모두 test 정답으로 골랐다.** 그래서 test 수치는 in-sample이고, 신뢰구간도 실제보다 좁게 나온다(사용자 요청, 공정성 조건 없음). C 구간은 같은 구조를 test 이전 구간에 그대로 적용한 참고 값이다.

## 결과 (frozen 배치)

| 모델 | test Mean Hit | test 손실 | 하루 연결 수 |
|---|---|---|---|
| Dense | 48.0% | 1.15996 | 100 |
| No graph | 49.7% | 1.14477 | 0 |
| Static | 51.2% | 1.14505 | 36 (매일 동일) |
| Dynamic | **53.2%** | 1.14379 | 평균 33.2 (15~44) |

- Dynamic의 하루 연결 수는 같은 regime 안에서도 달라진다. calm 25~43개, RiskOff 18~44개, HighVol 15~39개다.
- 0개인 날과 100개인 날은 없다.

## 비교 (95% CI, 10일 블록 bootstrap 10,000회, 양수 = 앞 모델이 좋음)

| 비교 | test Hit 차이 %p | C Hit 차이 %p | test 손실 차이 ×10⁻³ |
|---|---|---|---|
| Static vs No graph | +1.50 [+0.49, +2.53] | −0.10 [−1.02, +0.80] | −0.28 [−1.36, +0.81] |
| Dynamic vs Static | +1.94 [+1.27, +2.59] | +0.75 [+0.10, +1.40] | +1.26 [+0.15, +2.37] |
| Dynamic vs No graph | +3.43 [+2.29, +4.56] | +0.65 [−0.34, +1.61] | +0.98 [−0.52, +2.49] |
| Dynamic vs Dense | +5.14 [+3.87, +6.41] | +0.91 [−0.21, +1.99] | +16.17 [+11.45, +20.99] |

C 구간의 Mean Hit는 Dense 49.2%, No graph 49.5%, Static 49.4%, Dynamic 50.1%다.

## 모델 정의

- 공통: CPGG-v2의 자기 예측 b와 이웃 메시지 d를 쓴다. 네 모델은 연결 행렬 A만 다르다.
- Static: 36개 쌍을 매일 같게 연결한다. test 평균 gain이 양수인 쌍에서 시작해, 쌍을 하나씩 켜고 끄며 test Mean Hit를 높였다.
- Dynamic: (쌍, regime) 칸 표를 Static에서 시작해 test Mean Hit로 최적화한다. 여기에 그날 선택기 점수(G-loss-D ridge)가 test 점수의 30분위를 넘을 때만 연결한다는 조건을 곱한다. 각 날짜의 연결 결정에는 그 날짜까지 알 수 있는 정보만 쓴다.
- 인정 조건: 하루 연결 수가 모든 날 같으면 안 되고, 0개나 100개인 날이 있어도 안 된다. 같은 regime 안에서 연결 수가 8가지 이상 달라져야 한다. 인정 조건을 통과한 94개 후보 중 test Mean Hit가 가장 높은 것을 골랐다.

## refit 배치

| 모델 | test Mean Hit | test 손실 | 하루 연결 수 |
|---|---|---|---|
| Dense | 48.3% | 1.15573 | 100 |
| No graph | 49.2% | 1.14870 | 0 |
| Static | 50.4% | 1.14990 | 34 (매일 동일) |
| Dynamic | 51.4% | 1.14751 | 평균 18.9 (2~36) |

refit Dynamic의 조건은 "메시지와 자기 예측의 부호가 반대일 때만 연결"이다. C 구간에서는 No graph와 같다(Mean Hit 49.5%).

## 주의할 점

- 94개 후보와 칸 표 최적화를 모두 같은 test로 했다. 53.2%는 이 test에 맞춘 상한에 가깝다.
- C 구간에서 Dynamic vs No graph는 +0.65 [−0.34, +1.61]로 유의하지 않다. 순서 전체(No graph < Static < Dynamic)는 C에서 유지되지 않는다. Static vs No graph가 C에서 −0.10이다.
- 상위 후보들이 모두 "Static 집합 + 점수 30분위 조건" 구조라서 사실상 한 모델의 변형이다. 점수 조건은 연결을 70% 정도 남기는 느슨한 조건이라, 효과는 대부분 Static 집합을 test로 최적화한 데서 올 가능성이 크다.

## 파일과 재현

- 수치: `eval_hitdyn_s0.json`, 연결 행렬: `masks_hitdyn_{frozen,refit}_s0.npz`, 로그: `run_hitdyn_s0.log`
- 코드: `src/cpgg_v2/graph_order_hit.py`
- 재현: `cd src && PYTHONPATH=. python -m cpgg_v2.graph_order_hit --seed 0` (약 3.5분, 1.4 GB)
