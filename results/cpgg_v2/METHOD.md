# CPGG-v2 방법 요약

이 연구를 위한 설계이며 널리 알려진 기존 알고리즘이 아니다. 유지한 것: 데이터·특징·정보 가용성(FX 입력 t−1까지), 공식 split, 예측 손실(Huber δ=1 + 1·BCE, 변동성 단위), 자기 예측기(CPGG OwnModel), latent strength 차분 출력, 채널 분해, 결정적 추론, 연결 수·null penalty 없음. test 미사용.

## 분할 (`split_manifest.json`)
train(행 10..3154)을 시간순 5블록(각 629행)으로 나눈다.
- 자기 예측 O_k: 블록 1..k(끝 5행 purge)로 학습 → 블록 k+1(OOF), O_5 → C.
- 메시지 fold f=1,2,3: 블록 1..f+1의 잔차 z − b로 학습(끝 5행 purge) → 블록 f+2에 gain 기록. 최종 메시지: 블록 1..5 → C.
- 선택기 학습: 블록 3–4 기록. isotonic 보정: 블록 5 기록. c0: 블록 3–5 placebo 후보. 평가: C.

## M-v2 메시지 (`message_linear.py`)
d_ij,t = Σ_r 1[r_t = r]·(w_ij + u_ij,r)ᵀ φ_j,t. φ_j(32개): FX 수익률 lag 슬롯 5개(CPGG FX 채널의 표준화 값, t−1까지), |r_j|/v_j의 1·5·20일 평균, 금리 변화 lag·flag(11), 주식 수익률 lag·flag(13). 글로벌 없음. regime: 평시 / RiskOff / HighVol, 동시 해당 시 RiskOff 우선. 계층형 ridge 벌점 λ‖w‖² + λ·m‖u‖², λ ∈ {10, …, 1e6}, m ∈ {1, 10, 100}을 쌍마다 시간순 3-split expanding CV로 선택. 채널 블록(fx, rate, equity)별 부분합 = d (정확 분해). 비교용 M-v1은 CPGG 신경망 메시지를 같은 분할로 학습.

## gain 기록과 placebo (`gain_records.py`)
각 기록 날짜·(i, j)에 e = z − b, d, e·d, D = L(b) − L(b + d)(Huber+BCE)를 저장한다. placebo 후보: 쌍마다 송신 입력 φ_j를 행 10..4202 안에서 20행 이상 순환 이동한 시점의 값으로 바꾸고, 같은 w_ij,r과 현재 regime으로 d를 만든다. 진짜:가짜 = 1:1, 가짜 여부는 입력이 아니다.

## 칸별 과거 신뢰도 (`cell_reliability.py`)
(i, j, regime) 칸의 ρ = Σe·d / Σd²를 결정 시점 t에 target이 확정된 기록(t−2 이하)만으로 계산한다. EB: 쌍 ρ를 0 쪽으로(가중 100일), 칸 ρ를 쌍 ρ 쪽으로(가중 50일) 축소. se는 자기상관 보정 없는 sandwich 형태. 누수는 단위 테스트로 확인.

## 선택기 (`selector.py`)
입력: 칸 ρ_EB, 쌍 ρ_EB, log(1+n), se, |d|/sd, d/sd, 채널별 d_c/sd, regime 원핫, regime×|d|, 수신 통화 log v·|r|/v(1, 5일)·|b|, 다른 후보 합과의 부호 일치·크기. 
- G-loss-S: ridge(α ∈ {10, 100, 1e3, 1e4}) on [d·x̃, d²·x̃] → q̂ ≈ E[e·d | x], ĝ = 2q̂ − d².
- G-loss-D: ridge on [x̃, d·x̃, d²·x̃] → D 직접.
- 보조 MLP(hidden 8, 16 × wd 1e-3, 1e-2; regime×쌍 4차원 embedding).
- 보정: 블록 5에서 isotonic(raw ĝ → 실현 D). c0 = placebo 후보(블록 3–5) 보정 ĝ의 95분위.

## 추론 (`selection.py`)
주 규칙 M = 1[보정 ĝ > c0], s_i = b_i + Σ_j M_ij Σ_c d_ij,c. 독립 판단, 0~10개 허용. 보조: 임계값 0, max(c0, 0), 순차 greedy(S: ĝ_j|S = 2(q̂_j − d_j Σ_{k∈S} d_k) − d_j²), ρ_EB 축소, 칸 규칙(ρ_EB − 1.645·se > 0 또는 > 0.5), 같은 K 무작위, 정적 칸(기록 구간 평균 실현 D > 0).

## 평가 (`evaluate.py`)
주 지표: 날짜별 10개 통화 평균 prediction loss의 paired 차이, 10일 원형 블록 bootstrap 10,000회(민감도 5, 20일). H1 Dense vs Own, H2 주 모델 vs Dense, H3 주 모델 vs Own. 선택기 설정은 C loss로 선택(사용자 허용, 낙관적). 보조: 방향 log-loss·skill, Mean Hit, Brier, IC, return loss, RMSE, 통화별, stress(VIX·금리·FX 이동).

## 합성 점검 (`synthetic_checks.py`)
CPGG 12절과 같은 주입(CHF lag 슬롯에 독립 N(0,1), JPY target에 c·s_{t−1}). A에 SEK 무신호 후보 추가. B는 regime 집합을 주입 regime(VIX 변화 > 0)으로 둔 실행으로 판정, 실제 3-regime 실행은 참고. C는 SEK 무신호·NOK 중복. D는 시차·분해·자기쌍·USD·누수·placebo 선택률.
