"""Download OECD STES revisions (real-time vintages) for CPI and GDP of the 11 economies.

Each row = (area, measure, EDITION, TIME_PERIOD, value): the value of the observation as published in that monthly edition.
"""
from __future__ import annotations
import io, sys
import pandas as pd
from common import *

AREAS = {"USD": "USA", "EUR": "EA20", "JPY": "JPN", "GBP": "GBR", "CAD": "CAN", "AUD": "AUS",
         "KRW": "KOR", "CHF": "CHE", "NZD": "NZL", "SEK": "SWE", "NOK": "NOR"}
EXTRA = ["EA11", "EA17", "EA18", "EA19", "DEU"]   # euro-area aggregate codes vary by edition; keep all, choose later
FLOW = "OECD.SDD.STES,DSD_STES_REVISIONS@DF_STES_REVISIONS,"


def fetch(area: str) -> pd.DataFrame:
    url = f"https://sdmx.oecd.org/public/rest/data/{FLOW}/{area}.M+Q.CP+B1GQ_Q..."
    r = get(url, params={"startPeriod": "2002-01", "dimensionAtObservation": "AllDimensions", "format": "csvfile"}, timeout=600)
    d = pd.read_csv(io.StringIO(r.text))
    keep = ["REF_AREA", "FREQ", "MEASURE", "UNIT_MEASURE", "ACTIVITY", "EDITION", "TIME_PERIOD", "OBS_VALUE"]
    return d[keep]


if __name__ == "__main__":
    (RAW / "oecd").mkdir(parents=True, exist_ok=True)
    for ccy, area in list(AREAS.items()) + [(a, a) for a in EXTRA]:
        p = RAW / "oecd" / f"vintages_{area}.csv"
        if p.exists():
            print("skip", area); continue
        try:
            d = fetch(area)
        except Exception as e:  # noqa: BLE001
            print(f"{area}: FAIL {str(e)[:120]}"); continue
        d.to_csv(p, index=False)
        log_download("oecd_stes_revisions", area, p, len(d))
        g = d.groupby(["MEASURE", "FREQ", "UNIT_MEASURE", "ACTIVITY"]).agg(
            ed_min=("EDITION", "min"), ed_max=("EDITION", "max"), n_ed=("EDITION", "nunique"), tp_max=("TIME_PERIOD", "max"))
        print(f"{ccy}/{area}: {len(d)} rows"); print(g.to_string())
