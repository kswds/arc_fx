"""Download raw series (no transformation). One file per series under data/raw/<source>/."""
from __future__ import annotations
import io, sys
import pandas as pd
from common import *

load_env()
FRED_KEY = os.environ["FRED_API_KEY"]
ECOS_KEY = os.environ["ECOS_API_KEY"]


def save(df: pd.DataFrame, source: str, name: str) -> None:
    (RAW / source).mkdir(parents=True, exist_ok=True)
    p = RAW / source / f"{name}.csv"
    df.to_csv(p, index=False)
    log_download(source, name, p, len(df))
    print(f"  {source}/{name}: {len(df)} rows  {df.iloc[0,0]} .. {df.iloc[-1,0]}")


# ---------- FRED (daily market series; not revised in practice) ----------
FRED_SERIES = ["DEXUSEU", "DEXJPUS", "DEXUSUK", "DEXCAUS", "DEXUSAL", "DEXKOUS", "DEXSZUS", "DEXUSNZ",
               "DEXSDUS", "DEXNOUS", "DGS1", "DGS2", "DGS5", "DGS10", "VIXCLS", "DCOILBRENTEU"]


def fred_obs(series_id: str, **extra) -> pd.DataFrame:
    r = get("https://api.stlouisfed.org/fred/series/observations",
            params=dict(series_id=series_id, api_key=FRED_KEY, file_type="json",
                        observation_start=START, **extra))
    df = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()


def run_fred():
    print("FRED")
    for sid in FRED_SERIES:
        save(fred_obs(sid), "fred", sid)


# ---------- ECB reference rates (14:15 CET), robustness set ----------
def run_ecb():
    print("ECB EXR")
    for c in ["USD"] + [c for c in CCYS if c != "EUR"]:
        r = get(f"https://data-api.ecb.europa.eu/service/data/EXR/D.{c}.EUR.SP00.A",
                params=dict(format="csvdata", startPeriod=START))
        d = pd.read_csv(io.StringIO(r.text))[["TIME_PERIOD", "OBS_VALUE"]]
        d.columns = ["date", "value"]
        save(d, "ecb", f"EXR_{c}")


# ---------- Daily 10Y government yields, national sources ----------
def run_rates():
    print("10Y yields")
    # EUR: ECB AAA euro-area spot curve, 10Y
    r = get("https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
            params=dict(format="csvdata", startPeriod=START))
    d = pd.read_csv(io.StringIO(r.text))[["TIME_PERIOD", "OBS_VALUE"]]; d.columns = ["date", "value"]
    save(d, "rates", "EUR_10Y_ECB_AAA")
    # GBP: BoE nominal par yield 10Y (IUDMNZC)
    r = get("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp",
            params={"csv.x": "yes", "Datefrom": "01/Jan/2004", "Dateto": "now", "SeriesCodes": "IUDMNZC",
                    "CSVF": "TN", "UsingCodes": "Y"},
            headers={"User-Agent": "academic-research-data-collector/1.0 (daily 10Y gilt yield, non-commercial)"})
    d = pd.read_csv(io.StringIO(r.text)); d.columns = ["date", "value"]
    d["date"] = pd.to_datetime(d["date"], format="%d %b %Y").dt.strftime("%Y-%m-%d")
    save(d, "rates", "GBP_10Y_BoE")
    # JPY: MOF JGB 10Y (historical + current files)
    parts = []
    for u in ["historical/jgbcme_all.csv", "jgbcme.csv"]:
        t = get("https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/" + u).content.decode("latin1")
        parts.append(pd.read_csv(io.StringIO(t), skiprows=1)[["Date", "10Y"]])
    d = pd.concat(parts); d.columns = ["date", "value"]
    d["date"] = pd.to_datetime(d["date"], errors="coerce"); d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna().drop_duplicates("date", keep="last").sort_values("date")
    d = d[d["date"] >= START]; d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    save(d, "rates", "JPY_10Y_MOF")
    # CAD: BoC benchmark 10Y
    t = get("https://www.bankofcanada.ca/valet/observations/BD.CDN.10YR.DQ.YLD/csv", params={"start_date": START}).text
    d = pd.read_csv(io.StringIO(t[t.find('"date"'):])); d.columns = ["date", "value"]
    save(d.dropna(), "rates", "CAD_10Y_BoC")
    # SEK: Riksbank SWEA 10Y govt bond
    j = get(f"https://api.riksbank.se/swea/v1/Observations/SEGVB10YC/{START}").json()
    save(pd.DataFrame(j)[["date", "value"]].dropna(), "rates", "SEK_10Y_Riksbank")
    # KRW: Bank of Korea ECOS 817Y002 / 010210000 (Treasury 10Y, daily)
    rows, step = [], 5000
    for s in range(1, 20000, step):
        j = get(f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_KEY}/json/kr/{s}/{s+step-1}/817Y002/D/20040101/20261231/010210000").json()
        rr = j.get("StatisticSearch", {}).get("row", [])
        rows += rr
        if len(rr) < step:
            break
    d = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]]; d.columns = ["date", "value"]
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"); d["value"] = pd.to_numeric(d["value"], errors="coerce")
    save(d.dropna(), "rates", "KRW_10Y_ECOS")
    # NOK: Norges Bank generic 10Y (API holds 2019+ only; earlier needs a manual file)
    t = get("https://data.norges-bank.no/api/data/GOVT_GENERIC_RATES/B.10Y.GBON", params={"format": "csv", "startPeriod": START}).text
    d = pd.read_csv(io.StringIO(t), sep=None, engine="python")[["TIME_PERIOD", "OBS_VALUE"]]; d.columns = ["date", "value"]
    save(d.dropna(), "rates", "NOK_10Y_NorgesBank_2019plus")
    # CHF: SNB rendoblid 10J0 (API cube stops 2025-07-31)
    t = get("https://data.snb.ch/api/cube/rendoblid/data/csv/en", params={"fromDate": START, "dimSel": "D0(10J0)"}).content.decode("utf-8-sig")
    x = pd.read_csv(io.StringIO(t), sep=";", skiprows=3)
    d = x[["Date", "Value"]].dropna(); d.columns = ["date", "value"]
    save(d, "rates", "CHF_10Y_SNB")


# ---------- Yahoo Finance: equity indices, DXY, gold, copper (unadjusted close, exchange-local session date) ----------
YAHOO = {
    "USD_Stock": "^GSPC", "EUR_Stock": "^GDAXI", "EUR_Stock_alt": "^STOXX50E", "JPY_Stock": "^N225",
    "GBP_Stock": "^FTSE", "CAD_Stock": "^GSPTSE", "AUD_Stock": "^AXJO", "KRW_Stock": "^KS11",
    "CHF_Stock": "^SSMI", "NZD_Stock": "^NZ50", "SEK_Stock": "^OMX", "NOK_Stock": "OSEBX.OL",
    "DXY": "DX-Y.NYB", "Gold": "GC=F", "Copper": "HG=F", "Oil": "CL=F",
}


def run_yahoo():
    import yfinance as yf
    print("Yahoo")
    for name, tk in YAHOO.items():
        h = yf.Ticker(tk).history(start=START, auto_adjust=False)
        d = h.reset_index()[["Date", "Close"]]
        d["Date"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
        d.columns = ["date", "value"]
        save(d.dropna(), "yahoo", name)


if __name__ == "__main__":
    which = sys.argv[1:] or ["fred", "ecb", "rates", "yahoo"]
    fns = {"fred": run_fred, "ecb": run_ecb, "rates": run_rates, "yahoo": run_yahoo}
    for w in which:
        fns[w]()
