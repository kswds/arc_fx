# FX manual source collection — target 2005-01-03 to 2025-12-31

**Important:** This is an executable acquisition bundle, **not the actual collected data**. In the generation environment, external file network downloads are blocked. When run on a computer with internet, the script attempts to download all available direct-file originals, then records outcomes. It never claims a successful download without checking actual file bytes.

## Running

- macOS: Extract ZIP. Open a Terminal in the extracted folder and run `python3 download_sources.py`, or right-click `run_mac.command` and open it.
- Windows: Double-click `run_windows.cmd` with Python 3 installed (or run `python download_sources.py`).
- `data/raw/manual/` receives originals, plus SEK normalized CSV. `data/reports/download_manifest.csv` reports success/errors and SHA256.
- Does not require packages/pip. Excel `.xls`/`.xlsx` are saved **as original source bytes**, not reshaped/merged.

## Required sources & collection checklist

| Series | Requested coverage | Source | Acquisition |
|---|---|---|---|
| AUD 10Y | 2005-01-03 to 2025-12-31 | RBA F2 daily historical + current | automatic **both** xls and xlsx |
| NZD 10Y | 2005-01-03 to 2025-12-31 | RBNZ B2 daily close, historical + current | automatic **both** xlsx |
| NOK 10Y | 2005-01-03 to 2018-12-31 | Norges Bank discontinued daily synthetic yields | automatic xlsx; select 10Y, and check difference vs API generic series |
| CHF 10Y | 2025-08-01 to 2025-12-31 | New SNB yield series, **methodology changed in 2025-08** | BROWSER DOWNLOAD REQUIRED; do not use stale cube; see below |
| NOK OSEBX | 2005-01-03 to 2025-12-31 | Euronext Live historical prices, ISIN NO0007035327-XOSL | BROWSER DOWNLOAD REQUIRED; source may restrict export range |
| SEK OMXS30 | 2005-01-03 to 2008-11-19 | public unofficial OMXS30 Gist (2000–2023), Nasdaq official historical alternative | script downloads and extracts a single date/close CSV **if external source accessible** |

## Browser-only sources: exactly what to do

**CHF 10Y**

1. Open SNB yield tables (the legacy `rendoblid` daily series stops at **July 2025** because of discontinuation of the "2002 methodology"). See [SNB discontinued methodology example](https://data.snb.ch/de/topics/ziredev/cube/rendoblim?dimSel=D0%2810J%29). The updated series may be accessible through the [new Swiss Confederation spot-yield chart](https://data.snb.ch/en/topics/ziredev/chart/rendeidgeurch) or the [SNB data portal](https://data.snb.ch/en). The [older `rendeiduebd` portal page](https://data.snb.ch/en/topics/ziredev/cube/rendeiduebd) is **not independently verified** as containing the successor 10Y daily series.
2. Search for *spot interest rates on Swiss Confederation bonds* / *Kassazinssätze von Eidgenössischen Anleihen*, select **daily**, **10 years**, filter `2025-08-01` through `2025-12-31` and use CSV/Excel export; place the resulting file unchanged inside `data/raw/manual/`.
3. Validate actual observation dates: Do not accept monthly averages, monthly ending yields, bond coupon returns, or a synthetic repeated series for the requested daily data.
4. Since **a methodology break occurred at July/August 2025**, use a different flag for post-change observations and compare overlap or provider methodology before joining. A series from a third-party website is an **alternative proxy** until definitions are verified.

**NOK OSEBX**

1. Open [Euronext OSEBX historical data](https://live.euronext.com/en/popout-page/getHistoricalPrice/NO0007035327-XOSL).
2. Set **From: 2005-01-03 / To: 2025-12-31**, select **Comma delimited (.csv)**, decimal separator `.` and consistent date formatting. Download. If history limits prevent selecting all years, preserve each available chunk, and record any inaccessible dates as missing.
3. Put source file(s) into `data/raw/manual/` without renaming. The source labels this product **OSEBX GR** (gross-return index). Compare with existing series' return/dividend convention; **OSEBP** is the price-return counterpart. Do not silently mix GR and price return.

**SEK OMXS30**

- Community source: https://gist.github.com/mtornwall/902c8330f5ad4ac64d6677d3586af231 — historical source is a user-uploaded Gist, **not audited as Nasdaq data**.
- Official alternative (browser, choose custom date range 2005-01-03..2008-11-19): https://indexes.nasdaqomx.com/Index/History/OMXS30
- Useful date checks: 2008-11-17, 2008-11-18, 2008-11-19 overlap with FRED's Nasdaq series `NASDAQOMXS30`; verify them against Yahoo prior to stitching. Note Nasdaq data redistribution/copyright restrictions.

## Data/source changes

- AUD RBA F2 daily `f02dhist.xls` contains 1995–2013; `f02d.xlsx` has subsequent daily updates. **Download both**. RBA historical listing: https://www.rba.gov.au/statistics/historical-data.html
- NZD B2 10-year is a **benchmark bond** not necessarily constant residual 10y. On **2025-08-25**, RBNZ changed the source/timing from mid rates to end-of-day closing rates; check discontinuity. RBNZ B2: https://www.rbnz.govt.nz/statistics/series/exchange-and-interest-rates/wholesale-interest-rates
- NOK source workbook contains discontinued **synthetic** sovereign yields through 2021-06-30, while current Norges Bank **generic** yield is the security closest to maturity. This requires explicit method reconciliation. https://www.norges-bank.no/en/topics/statistics/norwegian-government-securities/synthetic-yields-no-longer-updated/
- CHF SNB former yields stop at 2025-07; newer spot-yield methodology is not directly comparable without checking source definition. https://data.snb.ch/de/topics/ziredev/cube/rendoblim?dimSel=D0%2810J%29

## Verification state

The downloader validates only the XLS/XLSX file format and SHA256 and verifies the SEK normalized CSV date range. It does **not** certify that the downloaded workbook contains a complete 2005–2025 ten-year series: inspect the 10Y column and date continuity after extraction. Do not fabricate missing observations; keep a method-break flag at 2025-08 for CHF and NZD. For non-holiday gaps of >8 days, inspect source availability before filling.
