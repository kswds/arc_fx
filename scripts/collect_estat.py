"""Japan CPI (nationwide, all items, not seasonally adjusted) from e-Stat, 2020-base table 0003427113.

Raw CPI levels are not revised after publication (re-basing only rescales levels, y/y is unchanged),
so the latest download is equivalent to first-release values.  Availability dates are handled in build_panel.py.
"""
from __future__ import annotations
import pandas as pd
from common import *

load_env()
APP = os.environ["ESTAT_APP_ID"]
TABLE = "0003427113"   # CPI, 2020 base (monthly 1970-)


def main() -> None:
    r = get("https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData",
            params={"appId": APP, "statsDataId": TABLE, "cdCat01": "0001", "cdArea": "00000", "cdTab": "1,3",
                    "limit": 100000, "metaGetFlg": "N"})
    j = r.json()["GET_STATS_DATA"]
    assert j["RESULT"]["STATUS"] == 0, j["RESULT"]
    v = pd.DataFrame(j["STATISTICAL_DATA"]["DATA_INF"]["VALUE"])
    v = v.rename(columns={"@tab": "tab", "@time": "time", "$": "value"})[["tab", "time", "value"]]
    v["value"] = pd.to_numeric(v["value"], errors="coerce")
    print(v.head(3).to_string(), "\n", v.time.str.len().value_counts().to_dict())
    (RAW / "estat").mkdir(parents=True, exist_ok=True)
    p = RAW / "estat" / "JPN_CPI_2020base_national_all_items.csv"
    v.to_csv(p, index=False)
    log_download("estat", "JPN_CPI_2020base", p, len(v))
    m = v[(v.time.str.len() == 10) & (v.time.str[4:6] == "00")]  # monthly codes look like YYYY00MMMM
    print(len(v), "rows;", "tab counts", v.tab.value_counts().to_dict())


if __name__ == "__main__":
    main()
