# 데이터 재수집 계획 (Point-in-Time FX Panel)

작성일: 2026-10-29

## 0. 기존 데이터에서 확인된 문제 (`260714/data/processed`)

| 문제 | 근거 |
|---|---|
| **통화마다 FX 시점이 다름** | 기존 일별 수익률과 FRB H.10(뉴욕 정오) 수익률의 상관: AUD = 1.00(H.10 그 자체), KRW는 **전일** H.10과 0.28(서울 마감 기준), 나머지 통화는 당일 0.39–0.57, 익일 0.21–0.47(그 사이 어딘가의 시각). 통화 간 cross-section이 서로 다른 시각에 찍혀 있음 |
| 주말 행 존재 | 토/일 행이 각 36개 |
| 10Y 금리 = OECD **월평균**(`IRLTLT01*M156N`) | 월 1회만 바뀜(약 257회 변화). 월평균에는 그 달의 미래 일별 값이 들어 있어 **look-ahead** 발생 |
| Macro = 최신 개정값 + 고정 lag | CPI/GDP가 월말/분기말 날짜에 찍혀 있음. 실제 발표일·당시 값(vintage)과 무관 |
| 결측 | PPI, RealRate10Y 전 통화 100% 결측, CHF GDP 100% 결측 → 코드에서 0으로 채워짐 |
| 레짐/스트레스 분위수 | `add_regime_onehot_features`가 **전체 표본** 분위수 사용 (논문은 training-sample 기준) — 코드 측 수정 필요 |

## 1. 핵심 설계: 결정 시각(decision time)과 as-of join

- 결정 시각 τ_t = **영업일 t의 12:00 America/New_York** (H.10 noon buying rate 시각).
- 타깃: `r_{i,t+1} = log S_i(τ_{t+1}) − log S_i(τ_t)` (USD 대비 현지통화 절상 = +). 10개 통화 모두 **같은 시각**의 quote.
- 피처: 모든 시계열에 "가용 시각(available_at, UTC)"을 붙이고, `available_at ≤ τ_t − 30분`인 가장 최근 관측값만 사용 (`merge_asof`). 서머타임은 tz-aware 시각으로 자동 처리.
  - 아시아·오세아니아 주가(N225, KS11, AXJO, NZ50): 당일 t 종가 사용 가능
  - 유럽 주가(FTSE, DAX/STOXX, SMI, OMX, OBX): 마감이 ET 10:30–12:30 사이 → 규칙에 따라 자동으로 t 또는 t−1
  - 미국 시장(S&P500, VIX, DXY, 원자재 선물 정산, UST 금리): 항상 t−1
  - 각국 10Y 금리(장 마감 후/익일 공표): t−1
  - Macro: vintage 가용일 기준 (아래 3절)
- 대안(robustness): ECB 14:15 CET 기준환율로 동일 파이프라인 재생성 → 결과가 fixing 시각에 의존하지 않는지 확인.

## 2. 데이터 항목과 소스 (접근성 검증 결과)

### FX (타깃 + local 채널)
| 항목 | 소스 | 검증 |
|---|---|---|
| 10개 통화 USD 환율 (주) | FRED H.10: DEXUSEU, DEXJPUS, DEXUSUK, DEXCAUS, DEXUSAL, DEXKOUS, DEXSZUS, DEXUSNZ, DEXSDUS, DEXNOUS | ✅ 키 불필요, 2004–2026-10, 10개 통화 결측일 동일(미 공휴일) |
| 10개 통화 EUR 기준율 (robustness) | ECB EXR `D.{CCY}.EUR.SP00.A` | ✅ 2004–2026-10 전 통화 |

### Rate 채널
| 통화 | 일별 10Y 소스 | 검증 |
|---|---|---|
| USD (1Y/2Y/5Y/10Y) | FRED DGS1/2/5/10 | ✅ |
| EUR | ECB YC (AAA 10Y spot) | ✅ 2004-09~ |
| GBP | BoE IUDMNZC | ✅ 2004~ |
| JPY | 재무성 jgbcme_all.csv + jgbcme.csv | ✅ |
| CAD | BoC Valet BD.CDN.10YR.DQ.YLD | ✅ |
| SEK | Riksbank SWEA SEGVB10YC | ✅ |
| KRW | 한국은행 ECOS 817Y002 (국고채 10년) | ⚠️ **ECOS API 키 필요**(무료) |
| AUD | RBA F2 | ❌ 자동 접근 차단(403) → **수동 다운로드 필요** |
| NZD | RBNZ B2 | ❌ 자동 접근 차단(403) → **수동 다운로드 필요** |
| NOK | Norges Bank GOVT_GENERIC_RATES | ⚠️ API는 **2019년 이후만** → 2005–2018 구간 수동 파일 필요 |
| CHF | SNB rendoblid | ⚠️ **2025-07-31에서 갱신 중단**(포털 공표일 2025-09-01) → 이후 구간 대체 필요 |
| RealRate10Y | 10Y(t−1) − 당시 vintage CPI YoY | 파생 |

### Equity 채널 (Yahoo, 로컬 종가 + 거래소 tz)
^GSPC, ^STOXX50E(2007~; 2005–2007은 ^GDAXI로 대체 또는 ^GDAXI 단일 사용), ^N225, ^FTSE, ^GSPTSE, ^AXJO, ^KS11, ^SSMI, ^NZ50, ^OMX(**2008-11~**), NOK(**OBX.OL/^OSEAX 모두 Yahoo에서 데이터 없음** → 대체 필요). 과거 데이터의 Sweden=EWD(뉴욕 상장 ETF)는 시점이 달라 사용하지 않음.

### Global 블록
| 항목 | 소스 | 비고 |
|---|---|---|
| VIX | FRED VIXCLS | t−1 |
| S&P500 | Yahoo ^GSPC | t−1 |
| DXY | Yahoo DX-Y.NYB | t−1 |
| Broad dollar | FRED DTWEXBGS | **2006-01 시작**, 가중치 연례 개정으로 과거값 개정됨 → ALFRED vintage 필요(FRED API 키) |
| Oil | FRED DCOILBRENTEU (Brent spot) 권장 | CL=F는 2020-04-20 음수가격(-37.63)·롤 점프 |
| Gold, Copper | Yahoo GC=F, HG=F | 연속선물 롤 점프 존재(문서화) |
| Regime 지표 | 위 시계열에서 **training 구간 분위수**로 생성 | |

## 3. Macro: 발표 시점 + 개정 문제 해결

- 소스: **OECD `DSD_STES_REVISIONS@DF_STES_REVISIONS`** (단기경제통계 개정 DB). 11개 경제권(USA, EA, JPN, GBR, CAN, AUS, KOR, CHE, NZL, SWE, NOR) 모두 포함, 월별 에디션(vintage) 1999-02 ~ 2026-10. ✅ 키 불필요.
  - 지표: GDP volume(`B1GQ_Q`), CPI(`CP`). 예) KOR CPI 2025-10분 → 2025-11 에디션에 최초 등장, CHE GDP 2025-Q3 → 2025-12 에디션.
- 규칙:
  1. 에디션 YYYYMM의 값은 **다음 달 첫 영업일부터** 사용 가능 (에디션 작성 시점 이전에 이미 공표된 값이므로 look-ahead 없음, 대신 실제 발표보다 수주 늦을 수 있음 — 보수적).
  2. 날짜 t의 피처 = t 시점에 사용 가능한 **가장 최근 에디션** 안에서 계산한 값. YoY 인플레이션·GDP 성장률은 분자/분모 모두 **같은 vintage**에서 계산 (개정값 혼입 방지).
  3. 첫 vintage 이전은 forward/backfill 하지 않고 결측 + availability flag.
- 확인 필요: CHE CPI가 이 DB에서 2025-12에 끊김 → 보완 필요 여부 점검.
- **PPI는 이 DB에 없음** → 결정 필요 (아래 5절).
- (선택) FRED API 키가 있으면 미국은 ALFRED로 **정확한 발표일**·vintage 사용 가능.

## 4. 산출물

```
data/
  raw/<source>/...            # 원본 그대로 + 다운로드 시각 기록
  interim/series_long.parquet # (series_id, obs_date, available_at_utc, value, vintage)
  processed/
    fx_panel_pit.csv          # Date, {ccy}_FX(H.10 noon), TARGET_{ccy}_FX_RET_FWD1
    nonfx_features_pit.csv    # 기존 data_pipeline.py 컬럼 규약 유지
    availability_audit.csv    # 피처별 lag 분포(t 대비 사용된 관측일) — look-ahead 0건 검증
scripts/collect_*.py, build_panel.py
```
기존 `src/data_pipeline.py`가 그대로 읽을 수 있도록 컬럼명을 유지한다.

## 5. 진행 전 필요한 것

1. **FRED API 키** (ALFRED vintage: DTWEXBGS, 미국 macro 정확한 발표일, PPI 대안)
2. **ECOS API 키** (KRW 10Y)
3. 수동 다운로드 파일: RBA F2(AUD 10Y), RBNZ B2(NZD 10Y), Norges Bank 2005–2018 10Y, CHF 2025-08 이후 10Y, 노르웨이 주가지수(OBX/OSEBX) 일별
4. **PPI 처리 방침**

---
## 6. 진행 현황 (2026-10-29)

키 검증 완료(FRED, ECOS) → `.env`(gitignore). 원본은 `data/raw/`, 다운로드 로그는 `data/raw/DOWNLOAD_LOG.tsv`.
`scripts/collect_raw.py`, `scripts/collect_oecd_vintages.py` 로 재현 가능.

| 상태 | 항목 |
|---|---|
| ✅ 수집 완료 | H.10 FX 10종, ECB FX 10종, FRED(UST 1/2/5/10Y, VIX, Brent), Yahoo(주가 9지수·DXY·Gold·Copper), 10Y: EUR/GBP/JPY/CAD/SEK/KRW, OECD vintage(11개 경제권 CPI·GDP) |
| ⚠️ 부분 | NOK 10Y(2019~), CHF 10Y(~2025-07-31), EUR 주가(DAX 사용; STOXX50E는 2007~), SEK 주가(2008-11~) |
| ❌ 미수집 | AUD 10Y, NZD 10Y, NOK 주가, PPI |

### 새로 확인된 macro 이슈
- **JPN CPI**: OECD 개정 DB가 2022-07 판부터 일본 CPI를 갱신하지 않음(마지막 관측 2022-04). FRED/OECD 미러도 2021~2022에서 종료. 대체 소스 필요(총무성 e-Stat; appId 필요).
- **EUR**: 유로존 집계 코드가 판마다 달라 EA18(~2015-01) → EA19(~2023-03) → EA20(2023-04~)로 이어 붙임(공백 없음). 판 안에서는 항상 같은 집계로 YoY 계산.
- **DTWEXBGS(Broad dollar)**: ALFRED vintage가 2019-02부터만 존재 → 2019 이전 구간은 point-in-time 보장 불가.
- CHE/SWE/NOR/EA20 CPI는 최근 4개 판(2026-07~10)만 지연 → 표본(~2025-12) 영향 없음.

## 7. 사용자 결정 반영 (2026-10-29)
- JPN CPI: e-Stat 2020년 기준 표(0003427113, 전국·총합·원지수/전년동월비) 수집 완료(~2026-09). 원지수는 사후 개정이 없어 최신 다운로드 = 최초 발표값. 가용 시점은 보수적 규칙(익월 말 이후) 적용, 필요시 공표 캘린더로 교체.
- Broad dollar(DTWEXBGS): 제외, DXY(Yahoo DX-Y.NYB)로 대체. 논문 `Global_BroadDollar_ret`는 DXY 수익률로 대체되어 `Global_DXY_ret`와 중복 → 논문 서술/피처 정의 수정 필요.
- PPI: 전 통화 제외 (`*_PPIInfl`, `*_PPIInfl_minus_USD` 삭제; 논문 Table 1 수정 필요).
- 수동 파일 필요 목록: 8절 참조.

---
## 8. 구축 결과 (2026-10-29)

### 재현 순서
```
cd scripts
python collect_raw.py fred ecb rates yahoo     # 자동 수집 원본 -> data/raw/
python collect_oecd_vintages.py                # OECD 개정 DB(CPI, GDP) vintage
python collect_estat.py                        # 일본 CPI (e-Stat)
# 수동 파일(data/raw/manual): f02dhist.xls, f02d.xlsx, hb2-daily-close*.xlsx, government-bonds-synthetic-business-day.xlsx
python build_panel.py                          # -> data/processed/*
```
산출물: `fx_panel_pit.csv`(32열), `nonfx_features_pit.csv`(195열), `availability_audit.csv`, `missingness_by_split.csv`, `method_breaks.csv`. 5,252행, 2005-01-03~2025-12-17.

### 시점 규칙
- 결정 시각 = 영업일 12:00 뉴욕 (H.10), 피처는 11:30 뉴욕까지 공개된 값만 사용. 모든 시리즈에서 공개 시점 > 결정 시점 위반 0건(빌더가 assert).
- 아시아·오세아니아 주가는 당일, 유럽·미국 주가·VIX·DXY·원자재·모든 금리는 전일 이전 값.
- 금리는 관측일 다음 날 09:00(현지)부터 사용(공표 지연 보수 가정).
- Macro: OECD edition(YYYYMM)의 값은 다음 달 1일부터 사용, YoY는 같은 edition 안에서 계산(개정값 혼입 없음). 일본 CPI는 e-Stat 원지수(무개정), 기준월 M은 M+2월 1일부터 사용. 공표 캘린더가 아니라 보수적 규칙이므로 실제 발표보다 최대 약 1~2개월 늦게 반영됨.
- GDP_growth·CPIInfl는 전년동기비(%)로 새로 정의(기존 파일의 정의는 통화마다 일관되지 않았음: 미국 GDP는 연율 QoQ 유사, 한국은 QoQ 등).
- RealRate10Y = 10Y 금리 − CPIInfl(YoY).

### 코드 변경 (src/, 원본 백업: src/data_pipeline.py.orig)
1. **Regime 누수 수정**: 변동성 레짐이 다음 날 타깃(`TARGET_*_FWD1`)의 20일 평균으로 계산되어 라벨이 입력에 들어갔음 → 과거 수익률(`*_FX_ret_1d`) 기반으로 변경. 모든 분위수 임계값은 훈련 구간만으로 계산.
2. `{ccy}_Stock_Available` 플래그 추가(equity 채널 5→6차원).
3. PPI 피처 제거, `Global_BroadDollar_ret` 제거(DXY로 대체).
4. 설정 키 추가: `data.start_date`(민감도 A), `data.drop_equity_ccys`(B), `data.eval_end_date`(C).

### 실험 설정 (configs/)
`pit_main.yaml`, `pit_sensA_start2009.yaml`, `pit_sensB_no_nok_sek_equity.yaml`, `pit_sensC_eval_to_2025-07.yaml` (C는 main 예측을 재사용하고 2025-07-31까지만 평가). 출력은 `results/pit_*`로 분리되어 기존 결과를 덮어쓰지 않음.

### 알려진 한계
- 결측(학습 구간): NOK 주가 65%(2013-03-05~), NZD 금리 33%(2009-01-30~), SEK 주가 31%(2008-11-21~); 테스트: CHF 금리 8.3%(2025-08-13~).
- 연속 선물(Oil/Gold/Copper)의 롤 구간 점프, WTI 2020-04-20 음수가격(해당 수익률 NaN).
- NOK 주가는 총수익(GI) 지수, DAX는 성과지수, 나머지는 가격지수 혼재.
- 위 4개 지연 가정은 모두 PIT 안전 방향의 보수적 선택이며 정보량을 줄일 수 있음.
