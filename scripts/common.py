"""Shared helpers for point-in-time data collection."""
from __future__ import annotations
import os, time, datetime as dt
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
START = "2004-01-01"   # extra history for rolling features / lookbacks
CCYS = ["EUR", "JPY", "GBP", "CAD", "AUD", "KRW", "CHF", "NZD", "SEK", "NOK"]


def load_env() -> None:
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def get(url: str, params=None, retries: int = 3, timeout: int = 120, **kw) -> requests.Response:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, **kw)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET failed: {url} :: {last}")


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_download(source: str, name: str, path: Path, n: int) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / "DOWNLOAD_LOG.tsv", "a", encoding="utf8") as f:
        f.write(f"{stamp()}\t{source}\t{name}\t{path.relative_to(ROOT)}\t{n}\n")
