# Collection status in this ChatGPT session (2026-10-29)

Target: 6 daily financial series, 2005-01-03 to 2025-12-31 (NOK 10Y only through 2018-12-31, CHF only from 2025-08-01, SEK only through 2008-11-19).

External spreadsheet/CSV binaries could **not** be fetched in this execution environment because direct external networking is disabled. Consequently, **no actual observation files are bundled**. Zero of the six series have been delivered as verified market-data observations. No data were invented.

Confirmed locations:
- AUD historical/current: RBA `f02dhist.xls` and `f02d.xlsx`.
- NZD historical/current: RBNZ B2 historical/current XLSX.
- NOK historic: Norges Bank discontinued synthetic bond yields daily XLSX.
- SEK historic: community OMXS30.csv Gist with 2000-2023 data; verify against official Nasdaq before production use.
- CHF 2025-08 onwards: the previous SNB 2002-methodology yield series ended 2025-07; need new method daily 10Y spot-yield observations and explicit break flag.
- OSEBX: Euronext Live official historical CSV download interface accessible via browser but downloaded file not obtained.

`download_sources.py` fetches all directly addressable files when run from a computer with internet and writes an explicit download report. It leaves CHF and OSEBX as manual until a valid verified direct-download pathway is known.
