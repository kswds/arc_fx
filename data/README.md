# Data Description

This repository ships the code that builds a **point-in-time (PIT) daily FX panel** for eleven currencies
(`USD, EUR, JPY, GBP, CAD, AUD, KRW, CHF, NZD, SEK, NOK`) and the two CSV files the experiments read:

| File | Content |
|---|---|
| `data/processed/fx_panel_pit.csv` | FX levels at one common instant and next-observation targets (32 columns) |
| `data/processed/nonfx_features_pit.csv` | rate, equity, country-macro and global features (195 columns) |
| `data/processed/availability_audit.csv` | per-series audit: first/last valid date, information lag, look-ahead violations (must be 0) |
| `data/processed/missingness_by_split.csv` | share of missing values per column in the train / validation / test blocks |
| `data/processed/method_breaks.csv` | known definition/source breaks that remain in the series |

Raw downloads are not redistributed (`data/raw/` is git-ignored); `scripts/` re-creates them. A download log with
UTC timestamps is written to `data/raw/DOWNLOAD_LOG.tsv`.

An earlier version of the panel (`factor_daily_alligned_krw.csv`, `score_vA_nonfx_features.csv`) had time-alignment
and revision problems. Section 1 compares the two, Section 2 documents the new construction, Section 3 documents how the old files were built.

---

## 1. Old vs. new dataset

| | Old panel | New PIT panel |
|---|---|---|
| Rows / range | 5,542 FX rows incl. 72 weekend rows (2005-01-01 – 2025-12-19); 5,609 non-FX rows (to 2026-07-02); 5,470 after inner join | 5,252 rows (2005-01-03 – 2025-12-17), one row per US business day on which all ten H.10 rates exist |
| Columns | 63 + 208 | 32 + 195 |
| FX quote time | **different for each currency** (see below) | **one instant for all currencies**: FRB H.10 noon buying rate, New York |
| Target | next row's level / this row's level (weekend rows included) | noon-to-noon log return to the next H.10 date (gap ≤ 5 calendar days) |
| Feature timing | each series stamped with its own local calendar day, no availability rule | each value is the latest observation published **before 11:30 New York** on the decision date (asserted in the build; 0 violations) |
| 10Y yields | OECD **monthly averages**; 95 % of days have zero change, ~12 distinct values per year; value for month M appears on the last day of M | **daily** national-source yields; 1–13 % zero-change days, 120–250 distinct values per year |
| CPI / GDP | latest-revised values, fixed lag; CPI/GDP definitions inconsistent between countries | OECD real-time **editions** (vintages) + e-Stat for Japan CPI; year-on-year computed inside one edition |
| `*_RealRate10Y` | 100 % missing (11 columns) | filled: 10Y yield − CPI y/y |
| PPI | 91 % missing, filled with 0 in code | removed from the feature set |
| Broad dollar | `Global_BroadDollar` (FRED DTWEXBGS; revised annually, vintages exist only since 2019) | removed; DXY (never revised) is the only dollar index |
| SEK equity | `EWD` (USD-denominated NYSE ETF) | Nasdaq OMXS30 (`^OMX`), local currency; starts 2008-11-21 |
| Oil | `CL=F` front-month futures | `CL=F` front-month futures (settlement time handled; negative price 2020-04-20 → return NaN) |
| Missing data | silently turned into 0 after standardisation | NaN + availability flags (`*_Yield10Y_Available`, `*_Stock_Available`) |
| Regime indicators | volatility regime computed from the *next-day* targets; quantiles from the whole sample | computed from past returns only; quantiles from the training block only |

### Evidence

**FX clocks.** Correlation of the old daily FX returns with the H.10 noon returns (same day / next day / previous day):

| | EUR | JPY | GBP | CAD | AUD | KRW | CHF | NZD | SEK | NOK |
|---|---|---|---|---|---|---|---|---|---|---|
| same day | 0.39 | 0.45 | 0.57 | 0.40 | **1.00** | 0.46 | 0.56 | 0.43 | 0.47 | 0.39 |
| next day | 0.34 | 0.32 | 0.41 | 0.47 | -0.06 | 0.04 | 0.37 | 0.43 | 0.38 | 0.21 |
| previous day | 0.02 | -0.03 | 0.01 | 0.01 | -0.06 | **0.28** | 0.01 | 0.03 | -0.00 | -0.03 |

AUD in the old panel is the H.10 series itself (correlation 1.00); KRW correlates with the previous H.10 day more than any other currency (0.28, consistent with the Seoul close); the rest sit between the same-day and next-day H.10 returns:
the cross-section was not observed at one time, so cross-currency features and cross rates mixed different instants.

**Yields.** In the old file the 10Y value changes on the last calendar day of the month (e.g. EUR: 1.034 on 2022-08-30, 1.795 on 2022-09-30),
i.e. the monthly statistic is visible before the month it summarises is over, and is then frozen for a month.

**Macro columns.** In the old file 33 columns are 100 % missing and 9 `*_CPIInfl` columns contain `inf`
(EUR 549 rows, JPY 535, GBP 478, NOK 367, CAD 282, AUD 193, KRW 171, NZD 65, SEK 21; apparently a division by a zero CPI level, cause not verified). The code replaced them with 0 after standardisation.
Units were not comparable: mean |CPIInfl| is 0.026 for the USD but 2.55 for CHF. The GDP-growth definition also appeared to differ between countries
(spot checks: USD values are about four times the quarter-on-quarter change, KRW values equal it), so the old definitions were not reconstructed.

---

## 2. New dataset: sources and construction

### 2.1 Decision time and the as-of rule
For panel date *t* the decision time is **12:00 America/New_York**, the time of the FRB H.10 noon buying rates.
A feature observed on *t* is the most recent observation whose **availability timestamp (UTC) ≤ 11:30 New York on t**.
`scripts/build_panel.py` computes the timestamp per series, joins with `merge_asof`, and asserts that no value used
was published after the cutoff or dated after *t* (`availability_audit.csv`). Values older than 12 days (market series) become missing, so that a source that stops
(e.g. SNB CHF after 2025-07-31) is not carried forward.

Availability rules (all time zones handled with daylight saving):

| Series | Availability |
|---|---|
| Equity indices | local official close + 30 min on the observation date (Asia/Oceania: same day; Europe/Americas: previous day) |
| DXY | 17:00 NY + 30 min; futures (WTI, gold, copper) 15:00 NY + 30 min |
| VIX, US Treasury yields | next day 09:00 NY (FRED publication delay) |
| National 10Y yields | next calendar day 09:00 local (conservative publication delay) |
| OECD CPI / GDP | the 1st day of the month after the OECD edition month |
| Japan CPI (e-Stat) | the 1st day of month M+2 for reference month M |

### 2.2 Sources

| Block | Series | Source (identifier) | Access |
|---|---|---|---|
| FX (target and local features) | 10 USD cross rates, noon buying rates | FRB H.10 via FRED: `DEXUSEU DEXJPUS DEXUSUK DEXCAUS DEXUSAL DEXKOUS DEXSZUS DEXUSNZ DEXSDUS DEXNOUS` | FRED API (key) |
| FX (robustness, not used in the panel) | EUR-based rates 14:15 CET | ECB `EXR/D.{CCY}.EUR.SP00.A` | ECB SDMX API |
| Yields | USD 1Y/2Y/5Y/10Y | FRED `DGS1 DGS2 DGS5 DGS10` | FRED API |
| | EUR | ECB AAA euro-area spot curve, 10Y (`YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y`) | ECB SDMX API |
| | GBP | Bank of England, `IUDMNZC` | BoE database CSV |
| | JPY | Ministry of Finance, JGB benchmark yields (`jgbcme_all.csv`, `jgbcme.csv`) | download |
| | CAD | Bank of Canada Valet `BD.CDN.10YR.DQ.YLD` | API |
| | SEK | Riksbank SWEA `SEGVB10YC` | API |
| | KRW | Bank of Korea ECOS `817Y002` / `010210000` (Treasury 10Y, daily) | ECOS API (key) |
| | AUD | RBA table F2, series `FCMYGBAG10D` (`f02dhist.xls` to 2013-05-17, `f02d.xlsx` after) | manual download |
| | NZD | RBNZ table B2, daily close, 10-year secondary-market government yield (`hb2-daily-close-1985-2017.xlsx`, `hb2-daily-close.xlsx`) | manual download |
| | NOK | Norges Bank synthetic 10Y yields to 2018-12-31 (`government-bonds-synthetic-business-day.xlsx`), generic benchmark `GOVT_GENERIC_RATES/B.10Y.GBON` from 2019-01-02 | manual + API |
| | CHF | SNB `rendoblid` 10J0, ends 2025-07-31; later data not collected | API |
| Equity (local index, close) | USD `^GSPC`, EUR `^GDAXI`, JPY `^N225`, GBP `^FTSE`, CAD `^GSPTSE`, AUD `^AXJO`, KRW `^KS11`, CHF `^SSMI`, NZD `^NZ50`, SEK `^OMX`, NOK `OSEBX.OL` | Yahoo Finance (`yfinance`, unadjusted close) | API |
| Global | VIX | FRED `VIXCLS` | FRED API |
| | DXY | Yahoo `DX-Y.NYB` | API |
| | oil, gold, copper | Yahoo front-month futures `CL=F`, `GC=F`, `HG=F` | API |
| Macro (CPI, GDP) | USA, EA (EA18 → EA19 → EA20 chain), JPN (GDP only), GBR, CAN, AUS, KOR, CHE, NZL, SWE, NOR | OECD *Short-term economic statistics revisions* `OECD.SDD.STES,DSD_STES_REVISIONS@DF_STES_REVISIONS`, measures `CP` (CPI index) and `B1GQ_Q` (GDP volume), all monthly editions 1999-02 … | OECD SDMX API |
| | Japan CPI | e-Stat table 0003427113 (CPI, 2020 base, nationwide, all items, not seasonally adjusted) | e-Stat API (appId) |

Credentials are read from environment variables / `.env` (`FRED_API_KEY`, `ECOS_API_KEY`, `ESTAT_APP_ID`); none are stored in the repository.

### 2.3 How the vintage (revision) problem is handled
* The OECD revisions database stores, for every monthly **edition**, the values of the series *as published in that edition*.
  For each edition we compute the latest observation's year-on-year growth (CPI: 12 months, quarterly CPI/GDP: 4 quarters) **using only the values inside that edition**, so
  a later revision of the base period can never enter an earlier decision date.
* A decision date uses the newest edition whose availability date is not after the decision cutoff.
  Editions are made available only from the 1st of the following month; this is deliberately conservative: the information is typically 1–2 months older than the actual release.
  Example: on 2022-09-23 the US CPI feature is 9.06 % (June 2022), although August 2022 had already been released.
* The euro-area aggregate changes with time; within an edition the aggregate valid at that time is used (EA18 for editions ≤ 2015-01, EA19 for 2015-02 … 2023-03, EA20 after).
* Japan: the OECD database stopped updating Japanese CPI in the 2022-07 edition, so CPI y/y is computed from the e-Stat index (unrevised after publication; re-basing rescales levels but not the ratio).
* Market prices and yields are not revised and are used as downloaded.
* Not used because they cannot be made point-in-time: FRED `DTWEXBGS` (broad dollar; vintages only since 2019), FRED/EIA Brent spot (published with a multi-day delay),
  monthly OECD 10Y yields (monthly averages), PPI (no vintage database across the eleven economies).

### 2.4 Feature definitions (new)
* **FX**: H.10 levels; the pipeline converts them to a common local-currency-appreciation log return (`EUR GBP AUD NZD`: `log(x)`; `JPY CAD CHF SEK NOK KRW`: `-log(x)`).
  `TARGET_{ccy}_FX_RET_FWD1` = appreciation of the currency against USD from this H.10 date to the next.
* **Rate**: `Yield10Y`, `Yield10Y_change` (difference between consecutive panel dates), `Yield10Y_minus_US10Y`, `RealRate10Y = Yield10Y − CPIInfl`, `Yield10Y_Available`; US curve `Global_US1Y/2Y/5Y/10Y`, changes and 10Y−2Y slope.
* **Equity**: `Stock_ret` (log return between consecutive as-of values), 5/20-day momentum, 20-day volatility, return minus S&P 500, `Stock_Available`.
* **Country macro**: `CPIInfl` and `GDP_growth` in **percent year-on-year** (same definition for all countries), and their differences from the US.
* **Global**: `Global_VIX`, `Global_VIX_change`, `Global_SP500_ret`, `Global_DXY_ret`, `Global_Oil_ret`, `Global_Gold_ret`, `Global_Copper_ret`, `Global_US2Y_change`, `Global_US10Y_change`, `Global_US10Y_minus_US2Y`.
* **Regime indicators** are built in `src/data_pipeline.py` from the above, with thresholds estimated on the training block only.

### 2.5 Missing data and remaining limitations
Missing values stay missing (no back-filling); the model sees an availability flag.

| Series | Missing | Reason |
|---|---|---|
| `NOK_Stock` | 2005-01-03 – 2013-03-04 (65 % of the training block) | Yahoo's `OSEBX.OL` ("Oslo Børs Benchmark Index_GI"; the name suggests a gross-return index) starts 2013-03-05 |
| `SEK_Stock` | until 2008-11-20 (31 %) | Yahoo's `^OMX` starts 2008-11-20 |
| `NZD_Yield10Y` | until 2009-01-29 (33 %) | RBNZ daily 10Y series starts 2009-01-29 |
| `CHF_Yield10Y` | from 2025-08-13 (8 % of the test block) | SNB legacy series ends 2025-07-31 |

Other known issues (see `method_breaks.csv`): the NOK yield switches from synthetic to generic benchmark yields on 2019-01-02 (on the overlap 2019-01 … 2021-06 the two differ by 0.001 pp on average, std 0.020 pp);
NZD yields change from mid to closing quotes on 2025-08-25 according to the source notes; futures series are continuous front-month contracts and contain roll jumps;
the WTI price was negative on 2020-04-20 (affected returns are NaN); the equity indices mix price indices with the DAX (performance index) and the OSEBX (gross-return index).
OECD edition lags make macro features older than the true release by up to ~2 months (conservative by design).

---

## 3. Old dataset: how it was built

The old panel (`factor_daily_alligned_krw.csv`, `score_vA_nonfx_features.csv`) was inherited from an earlier project; the collection scripts were not part of this repository.
Its sources, as documented at the time:

* FX levels and targets: a "legacy aligned" table (not re-fetched; provenance of the quote time not documented). Evidence in Section 1 shows mixed clocks.
* FRED: VIX `VIXCLS`, broad dollar `DTWEXBGS`, Treasury yields `DGS1/2/5/10`, country 10Y yields `IRLTLT01*M156N` (OECD monthly averages), macro series for CPI/GDP/PPI (latest available vintage).
* Yahoo Finance: `^GSPC, ^GDAXI, ^N225, ^FTSE, ^GSPTSE, ^AXJO, ^KS11, ^SSMI, ^NZ50, EWD` (SEK proxy), `OBX.OL` (NOK; Yahoo now returns almost no history for it), `CL=F, GC=F, HG=F`, `DX-Y.NYB`.
* Low-frequency series were forward-filled to business days with a fixed publication lag and never back-filled, so early values are missing.

Problems that motivated the rebuild: mixed FX quote times; monthly-average yields stamped at month end; latest-vintage macro data with inconsistent definitions, `inf` values and silently zero-filled gaps;
the SEK equity proxy was a USD-denominated ETF; and the volatility regime indicator used next-day targets.

---

## 4. Rebuilding the dataset

```bash
cd scripts
python collect_raw.py fred ecb rates yahoo       # market data  -> data/raw/
python collect_oecd_vintages.py                  # OECD CPI/GDP editions
python collect_estat.py                          # Japan CPI (e-Stat)
# manual files in data/raw/manual/: f02dhist.xls, f02d.xlsx (RBA F2),
#   hb2-daily-close-1985-2017.xlsx, hb2-daily-close.xlsx (RBNZ B2),
#   government-bonds-synthetic-business-day.xlsx (Norges Bank)
python build_panel.py                            # -> data/processed/*
```

Training and evaluation:

```bash
python src/train.py    --config configs/pit_main.yaml --device cpu
python src/evaluate.py --config configs/pit_main.yaml
```

Sensitivity configurations for the missing-data blocks: `configs/pit_sensA_start2009.yaml` (training starts 2009-02, no NZD/SEK gaps),
`configs/pit_sensB_no_nok_sek_equity.yaml` (equity channel of NOK and SEK removed), `configs/pit_sensC_eval_to_2025-07.yaml` (test scored only through 2025-07-31, no CHF gap).
