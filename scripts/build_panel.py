"""Build the point-in-time FX panel.

Decision time for panel date t is 12:00 America/New_York (the FRB H.10 noon buying-rate time).
Every feature observed at t is the most recent observation whose *availability timestamp* (UTC)
is <= 11:30 New York time on t.  Targets are H.10 noon-to-noon log returns, so all ten currencies
are quoted at the same instant.  Macro variables come from OECD real-time editions (vintages) and are used
only after the edition was compiled; Japan CPI comes from e-Stat (raw index, never revised).

Outputs (data/processed):
  fx_panel_pit.csv, nonfx_features_pit.csv, availability_audit.csv, missingness_by_split.csv, method_breaks.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from common import *

PROC = ROOT / "data" / "processed"
MAN = RAW / "manual"
ALL = ["USD"] + CCYS
CUTOFF_MARGIN = pd.Timedelta(minutes=30)       # information must be available 30 min before the noon fixing
MARKET_TOLERANCE = pd.Timedelta(days=12)       # longest holiday gap; beyond this a stale value becomes missing
MACRO_STALE_DAYS = 540

# H.10: (series, sign) ; sign=+1 -> USD per unit of foreign currency, -1 -> foreign per USD
H10 = {"EUR": ("DEXUSEU", 1), "JPY": ("DEXJPUS", -1), "GBP": ("DEXUSUK", 1), "CAD": ("DEXCAUS", -1),
       "AUD": ("DEXUSAL", 1), "KRW": ("DEXKOUS", -1), "CHF": ("DEXSZUS", -1), "NZD": ("DEXUSNZ", 1),
       "SEK": ("DEXSDUS", -1), "NOK": ("DEXNOUS", -1)}

# (file, tz, local close "HH:MM") : availability = local close + 30 min on the observation date
EQUITY = {
    "USD": ("yahoo/USD_Stock.csv", "America/New_York", "16:00"),
    "EUR": ("yahoo/EUR_Stock.csv", "Europe/Berlin", "17:30"),
    "JPY": ("yahoo/JPY_Stock.csv", "Asia/Tokyo", "15:30"),
    "GBP": ("yahoo/GBP_Stock.csv", "Europe/London", "16:30"),
    "CAD": ("yahoo/CAD_Stock.csv", "America/Toronto", "16:00"),
    "AUD": ("yahoo/AUD_Stock.csv", "Australia/Sydney", "16:10"),
    "KRW": ("yahoo/KRW_Stock.csv", "Asia/Seoul", "15:30"),
    "CHF": ("yahoo/CHF_Stock.csv", "Europe/Zurich", "17:30"),
    "NZD": ("yahoo/NZD_Stock.csv", "Pacific/Auckland", "16:45"),
    "SEK": ("yahoo/SEK_Stock.csv", "Europe/Stockholm", "17:30"),
    "NOK": ("yahoo/NOK_Stock.csv", "Europe/Oslo", "16:20"),
}
# Daily yields: availability = 09:00 local time on the NEXT calendar day (conservative publication delay)
YIELD_TZ = {"EUR": "Europe/Berlin", "JPY": "Asia/Tokyo", "GBP": "Europe/London", "CAD": "America/Toronto",
            "AUD": "Australia/Sydney", "KRW": "Asia/Seoul", "CHF": "Europe/Zurich", "NZD": "Pacific/Auckland",
            "SEK": "Europe/Stockholm", "NOK": "Europe/Oslo"}
NY = "America/New_York"


# ----------------------------------------------------------------------------- helpers
def read_series(path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = ["date", "value"]
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d.dropna().drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def avail_local(dates: pd.Series, tz: str, hhmm: str, day_offset: int = 0, buffer_min: int = 0) -> pd.Series:
    h, m = map(int, hhmm.split(":"))
    local = dates + pd.Timedelta(days=day_offset) + pd.Timedelta(hours=h, minutes=m + buffer_min)
    return local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")


def with_avail(d: pd.DataFrame, tz: str, hhmm: str, day_offset: int = 0, buffer_min: int = 0) -> pd.DataFrame:
    d = d.copy()
    d["avail"] = avail_local(d["date"], tz, hhmm, day_offset, buffer_min)
    return d.dropna(subset=["avail"]).sort_values("avail").reset_index(drop=True)


def asof(panel: pd.DataFrame, s: pd.DataFrame, name: str, tol=MARKET_TOLERANCE) -> pd.DataFrame:
    """Attach the latest observation with avail <= cutoff. Returns frame with value/obs_date/avail columns."""
    left = panel[["cutoff"]].reset_index().sort_values("cutoff")
    right = s[["avail", "value", "date"]].rename(columns={"date": "obs_date"})
    m = pd.merge_asof(left, right, left_on="cutoff", right_on="avail", direction="backward", tolerance=tol)
    m = m.sort_values("index").set_index("index")
    m.columns = [f"{c}" if c == "cutoff" else f"{name}|{c}" for c in m.columns]
    return m.drop(columns=["cutoff"])


# ----------------------------------------------------------------------------- manual / special sources
def load_aud() -> pd.DataFrame:
    parts = []
    for f, skip in [("f02dhist.xls", 11), ("f02d.xlsx", 11)]:
        d = pd.read_excel(MAN / f, sheet_name="Data", header=None, skiprows=skip)
        parts.append(pd.DataFrame({"date": pd.to_datetime(d[0], errors="coerce"),
                                   "value": pd.to_numeric(d[4], errors="coerce")}).dropna())
    d = pd.concat(parts).drop_duplicates("date", keep="last").sort_values("date")
    return d.reset_index(drop=True)


def load_nzd() -> pd.DataFrame:
    parts = []
    for f in ["hb2-daily-close-1985-2017.xlsx", "hb2-daily-close.xlsx"]:
        d = pd.read_excel(MAN / f, sheet_name="Data", header=None)
        hdr = d.iloc[:6, :].astype(str)
        col = [c for c in d.columns if hdr[c].str.contains("10 year").any() and hdr[c].str.contains("government", case=False).any()]
        assert len(col) == 1, (f, col)
        body = d.iloc[5:]
        parts.append(pd.DataFrame({"date": pd.to_datetime(body[0], errors="coerce"),
                                   "value": pd.to_numeric(body[col[0]], errors="coerce")}).dropna())
    d = pd.concat(parts).drop_duplicates("date", keep="last").sort_values("date")
    return d.reset_index(drop=True)


def load_nok() -> pd.DataFrame:
    d = pd.read_excel(MAN / "government-bonds-synthetic-business-day.xlsx", sheet_name="10 years", header=None, skiprows=6)
    syn = pd.DataFrame({"date": pd.to_datetime(d[1], errors="coerce"), "value": pd.to_numeric(d[2], errors="coerce")}).dropna()
    syn = syn[syn["date"] <= "2018-12-31"]                      # synthetic series through 2018, generic benchmark afterwards
    api = read_series(RAW / "rates" / "NOK_10Y_NorgesBank_2019plus.csv")
    api = api[api["date"] >= "2019-01-01"]
    return pd.concat([syn, api]).sort_values("date").reset_index(drop=True)


def load_yield(ccy: str) -> pd.DataFrame:
    if ccy == "AUD": return load_aud()
    if ccy == "NZD": return load_nzd()
    if ccy == "NOK": return load_nok()
    f = {"EUR": "EUR_10Y_ECB_AAA", "JPY": "JPY_10Y_MOF", "GBP": "GBP_10Y_BoE", "CAD": "CAD_10Y_BoC",
         "SEK": "SEK_10Y_Riksbank", "KRW": "KRW_10Y_ECOS", "CHF": "CHF_10Y_SNB"}[ccy]
    return read_series(RAW / "rates" / f"{f}.csv")


# ----------------------------------------------------------------------------- macro (vintage based)
OECD_AREA = {"USD": "USA", "GBP": "GBR", "CAD": "CAN", "AUD": "AUS", "KRW": "KOR", "CHF": "CHE", "NZD": "NZL",
             "SEK": "SWE", "NOK": "NOR", "JPY": "JPN"}
EA_CHAIN = [("EA18", 0, 201501), ("EA19", 201502, 202303), ("EA20", 202304, 999999)]   # euro-area aggregate valid per edition


def _yoy_last(g: pd.DataFrame, freq: str) -> tuple[pd.Period, float] | None:
    per = pd.PeriodIndex(g["TIME_PERIOD"].str.replace("-", "", regex=False), freq=freq)
    s = pd.Series(g["OBS_VALUE"].values, index=per).dropna().sort_index()
    if len(s) < 6: return None
    p = s.index[-1]
    base = p - (12 if freq == "M" else 4)
    if base not in s.index or s[base] <= 0: return None
    return p, 100.0 * (s[p] / s[base] - 1.0)


def oecd_edition_table(area: str, measure: str, ed_lo=0, ed_hi=999999) -> pd.DataFrame:
    d = pd.read_csv(RAW / "oecd" / f"vintages_{area}.csv")
    d = d[(d.MEASURE == measure) & (d.EDITION >= ed_lo) & (d.EDITION <= ed_hi)]
    if measure == "CP":
        freq = "M" if (d.FREQ == "M").any() else "Q"
        d = d[d.FREQ == freq]
    else:
        freq = "Q"; d = d[d.FREQ == "Q"]
    rows = []
    for ed, g in d.groupby("EDITION"):
        r = _yoy_last(g, freq)
        if r is None: continue
        p, v = r
        y, m = divmod(int(ed), 100)
        avail = pd.Timestamp(year=y + (m == 12), month=1 if m == 12 else m + 1, day=1, tz="UTC")   # 1st day of month after the edition
        rows.append({"avail": avail, "value": v, "date": p.end_time.normalize(), "edition": int(ed)})
    return pd.DataFrame(rows)


def macro_tables(ccy: str, measure: str) -> pd.DataFrame:
    if ccy == "EUR":
        return pd.concat([oecd_edition_table(a, measure, lo, hi) for a, lo, hi in EA_CHAIN]).sort_values("avail").reset_index(drop=True)
    return oecd_edition_table(OECD_AREA[ccy], measure).sort_values("avail").reset_index(drop=True)


def japan_cpi_table() -> pd.DataFrame:
    v = pd.read_csv(RAW / "estat" / "JPN_CPI_2020base_national_all_items.csv", dtype={"time": str, "tab": str})
    v = v[(v.tab == "1") & (v.time.str[:4] >= "2000") & (v.time.str[4:6] == "00")
          & (v.time.str[6:8] == v.time.str[8:10]) & (v.time.str[6:8] != "00")].copy()
    v["period"] = pd.PeriodIndex(v.time.str[:4] + "-" + v.time.str[6:8], freq="M")
    s = v.set_index("period")["value"].sort_index()
    s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
    yoy = (100 * (s / s.shift(12) - 1.0)).dropna()
    # nationwide CPI is released in the 3rd-4th week of month M+1; conservative: usable from the 1st day of month M+2
    return pd.DataFrame({"value": yoy.values,
                         "date": [p.to_timestamp(how="end").normalize() for p in yoy.index],
                         "avail": [(p + 2).to_timestamp(how="start").tz_localize("UTC") for p in yoy.index],
                         "edition": 0}).sort_values("avail").reset_index(drop=True)


# ----------------------------------------------------------------------------- main build
def main(start: str, end: str) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    # ---- decision calendar and FX
    fx = {}
    for c, (sid, sign) in H10.items():
        fx[c] = read_series(RAW / "fred" / f"{sid}.csv").set_index("date")["value"]
    fxdf = pd.DataFrame(fx).dropna()                                   # dates when all ten H.10 rates exist
    fxdf = fxdf[fxdf.index >= "2004-01-02"]
    dates_all = fxdf.index
    panel = pd.DataFrame(index=range(len(dates_all)))
    panel["Date"] = dates_all
    panel["cutoff"] = (dates_all + pd.Timedelta(hours=12)).tz_localize(NY).tz_convert("UTC") - CUTOFF_MARGIN
    panel["cutoff"] = panel["cutoff"].astype("datetime64[ns, UTC]")
    out = pd.DataFrame({"Date": panel["Date"]})
    audit: list[dict] = []

    def attach(name: str, s: pd.DataFrame, tol=MARKET_TOLERANCE, kind="market", source=""):
        s = s.copy(); s["avail"] = s["avail"].astype("datetime64[ns, UTC]")
        m = asof(panel, s, name, tol)
        out[name] = m[f"{name}|value"].values
        a, o = m[f"{name}|avail"], m[f"{name}|obs_date"]
        valid = out[name].notna() & (out["Date"] >= pd.Timestamp(start)) & (out["Date"] <= pd.Timestamp(end))
        viol_avail = int((a[valid.values] > panel["cutoff"][valid.values]).sum())
        viol_obs = int((o[valid.values] > out["Date"][valid.values]).sum())
        lag = (out["Date"][valid.values] - o[valid.values]).dt.days
        audit.append({"series": name, "kind": kind, "source": source, "n_valid": int(valid.sum()),
                      "first_valid": str(out.loc[valid, "Date"].min().date()) if valid.any() else "",
                      "last_valid": str(out.loc[valid, "Date"].max().date()) if valid.any() else "",
                      "lag_days_mean": round(float(lag.mean()), 2) if valid.any() else np.nan,
                      "lag_days_max": int(lag.max()) if valid.any() else np.nan,
                      "same_day_share": round(float((lag == 0).mean()), 3) if valid.any() else np.nan,
                      "violations_avail_after_cutoff": viol_avail, "violations_obs_date_after_t": viol_obs})
        assert viol_avail == 0 and viol_obs == 0, f"look-ahead in {name}"

    # ---- markets
    for c, (f, tz, close) in EQUITY.items():
        attach(f"{c}_Stock", with_avail(read_series(RAW / f), tz, close, 0, 30), source=f)
    attach("Global_VIX", with_avail(read_series(RAW / "fred" / "VIXCLS.csv"), NY, "09:00", 1), source="FRED VIXCLS")
    for k, sid in {"US1Y": "DGS1", "US2Y": "DGS2", "US5Y": "DGS5", "US10Y": "DGS10"}.items():
        attach(f"Global_{k}", with_avail(read_series(RAW / "fred" / f"{sid}.csv"), NY, "09:00", 1), source=f"FRED {sid}")
    attach("Global_DXY", with_avail(read_series(RAW / "yahoo" / "DXY.csv"), NY, "17:00", 0, 30), source="Yahoo DX-Y.NYB")
    for k, f in {"Oil": "Oil", "Gold": "Gold", "Copper": "Copper"}.items():
        attach(f"Global_{k}", with_avail(read_series(RAW / "yahoo" / f"{f}.csv"), NY, "15:00", 0, 30), source=f"Yahoo front-month futures {f}")
    for c in CCYS:
        attach(f"{c}_Yield10Y", with_avail(load_yield(c), YIELD_TZ[c], "09:00", 1), source=f"national 10Y {c}")
    out["USD_Yield10Y"] = out["Global_US10Y"]

    # ---- macro from vintages
    def attach_macro(name: str, tab: pd.DataFrame, source: str):
        tab = tab.copy(); tab["avail"] = tab["avail"].astype("datetime64[ns, UTC]")
        m = asof(panel, tab.assign(value=tab["value"]), name, tol=None)
        ed = tab.set_index("avail")["edition"]
        v, od = m[f"{name}|value"].copy(), m[f"{name}|obs_date"]
        stale = (out["Date"] - od).dt.days > MACRO_STALE_DAYS
        v[stale.values] = np.nan
        out[name] = v.values
        valid = out[name].notna() & (out["Date"] >= pd.Timestamp(start)) & (out["Date"] <= pd.Timestamp(end))
        a = m[f"{name}|avail"]
        viol = int((a[valid.values] > panel["cutoff"][valid.values]).sum())
        lag = (out["Date"][valid.values] - od[valid.values]).dt.days
        audit.append({"series": name, "kind": "macro_vintage", "source": source, "n_valid": int(valid.sum()),
                      "first_valid": str(out.loc[valid, "Date"].min().date()) if valid.any() else "",
                      "last_valid": str(out.loc[valid, "Date"].max().date()) if valid.any() else "",
                      "lag_days_mean": round(float(lag.mean()), 1) if valid.any() else np.nan,
                      "lag_days_max": int(lag.max()) if valid.any() else np.nan, "same_day_share": 0.0,
                      "violations_avail_after_cutoff": viol, "violations_obs_date_after_t": int((od[valid.values] > out["Date"][valid.values]).sum())})
        assert viol == 0, f"macro look-ahead in {name}"

    for c in ALL:
        if c == "JPY":
            attach_macro("JPY_CPIInfl", japan_cpi_table(), "e-Stat CPI 2020-base (unrevised index), usable from 1st of M+2")
        else:
            attach_macro(f"{c}_CPIInfl", macro_tables(c, "CP"), "OECD STES revisions: edition vintage")
        attach_macro(f"{c}_GDP_growth", macro_tables(c, "B1GQ_Q"), "OECD STES revisions: edition vintage")

    # ---- derived features on panel dates
    def lr(x: pd.Series) -> pd.Series:
        x = x.where(x > 0)
        return np.log(x).diff()
    feats = {}
    spx_ret = lr(out["USD_Stock"])
    for c in ALL:
        st = out[f"{c}_Stock"]; r = lr(st)
        feats[f"{c}_Stock_ret"] = r
        feats[f"{c}_Stock_mom_5"] = np.log(st.where(st > 0)).diff(5)
        feats[f"{c}_Stock_mom_20"] = np.log(st.where(st > 0)).diff(20)
        feats[f"{c}_Stock_vol_20"] = r.rolling(20).std()
        feats[f"{c}_Stock_ret_minus_SP500_ret"] = r - spx_ret
        feats[f"{c}_Stock_Available"] = st.notna().astype(float)
        y = out[f"{c}_Yield10Y"]
        feats[f"{c}_Yield10Y_change"] = y.diff()
        feats[f"{c}_Yield10Y_minus_US10Y"] = y - out["Global_US10Y"]
        feats[f"{c}_RealRate10Y"] = y - out[f"{c}_CPIInfl"]
        feats[f"{c}_Yield10Y_Available"] = y.notna().astype(float)
        for k in ("GDP_growth", "CPIInfl"):
            feats[f"{c}_{k}_minus_USD"] = out[f"{c}_{k}"] - out[f"USD_{k}"]
    feats["Global_SP500"] = out["USD_Stock"]
    feats["Global_SP500_ret"] = spx_ret
    feats["Global_VIX_change"] = out["Global_VIX"].diff()
    for k in ("DXY", "Oil", "Gold", "Copper"):
        feats[f"Global_{k}_ret"] = lr(out[f"Global_{k}"])                 # non-positive prices (WTI 2020-04-20) -> NaN
    feats["Global_US10Y_minus_US2Y"] = out["Global_US10Y"] - out["Global_US2Y"]
    feats["Global_US10Y_change"] = out["Global_US10Y"].diff()
    feats["Global_US2Y_change"] = out["Global_US2Y"].diff()
    out = pd.concat([out, pd.DataFrame(feats)], axis=1)

    # ---- FX file with H.10 levels and next-panel-date targets
    fxo = pd.DataFrame({"Date": dates_all})
    fxo["USD_FX"] = 1.0
    for c, (_, sign) in H10.items():
        lvl = fxdf[c].values
        fxo[f"{c}_FX"] = lvl
        nxt = pd.Series(lvl).shift(-1).values
        fxo[f"Target_{c}_FX"] = nxt
        fxo[f"TARGET_{c}_FX_RET_FWD1"] = sign * (np.log(nxt) - np.log(lvl))
    fxo["Target_Date"] = pd.Series(dates_all).shift(-1).values

    sel = (out["Date"] >= pd.Timestamp(start)) & (out["Date"] <= pd.Timestamp(end))
    out, fxo = out[sel].reset_index(drop=True), fxo[sel].reset_index(drop=True)
    assert fxo[[c for c in fxo if c.startswith("TARGET_")]].notna().all().all(), "missing targets"
    gap = (fxo["Target_Date"] - fxo["Date"]).dt.days
    print(f"panel rows {len(out)}  {out.Date.min().date()} .. {out.Date.max().date()}  target gap days max {gap.max()}")
    fxo.to_csv(PROC / "fx_panel_pit.csv", index=False)
    out.to_csv(PROC / "nonfx_features_pit.csv", index=False)
    pd.DataFrame(audit).to_csv(PROC / "availability_audit.csv", index=False)

    # ---- missingness by chronological 60/20/20 split
    n = len(out); b = [0, int(n * .6), int(n * .8), n]
    rows = []
    for col in out.columns[1:]:
        if col.endswith("_Available"): continue
        rows.append({"column": col, **{nm: round(float(out[col].iloc[b[i]:b[i + 1]].isna().mean()), 3) for i, nm in enumerate(["train", "val", "test"])},
                     "first_valid": str(out.loc[out[col].notna(), "Date"].min().date()) if out[col].notna().any() else ""})
    miss = pd.DataFrame(rows); miss.to_csv(PROC / "missingness_by_split.csv", index=False)
    print("splits:", [str(out.Date.iloc[i].date()) for i in (0, b[1], b[2], n - 1)])
    print(miss[(miss[["train", "val", "test"]].max(axis=1) > 0.02)].to_string(index=False))
    pd.DataFrame([
        {"series": "NOK_Yield10Y", "break_date": "2019-01-02", "note": "Norges Bank synthetic yields (to 2018-12-31) -> generic benchmark yields; overlap 2019-01..2021-06 diff mean 0.001pp std 0.020pp"},
        {"series": "NZD_Yield10Y", "break_date": "2025-08-25", "note": "RBNZ B2 changed from mid rates to end-of-day closing rates (per source notes; not independently verified)"},
        {"series": "NOK_Stock", "break_date": "2013-03-05", "note": "Yahoo OSEBX.OL (Oslo Børs Benchmark Index_GI, gross-return) starts; earlier history not collected"},
        {"series": "SEK_Stock", "break_date": "2008-11-20", "note": "Yahoo ^OMX starts; earlier history not collected"},
        {"series": "CHF_Yield10Y", "break_date": "2025-07-31", "note": "SNB legacy series ends; later history not collected -> missing"},
        {"series": "Global_Oil", "break_date": "2020-04-20", "note": "WTI front-month settle was negative (-37.63); returns involving it are NaN"},
        {"series": "Global_Oil/Gold/Copper", "break_date": "", "note": "continuous front-month futures: returns across roll dates include roll jumps"},
    ]).to_csv(PROC / "method_breaks.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2005-01-03")
    ap.add_argument("--end", default="2025-12-17")
    a = ap.parse_args()
    main(a.start, a.end)
