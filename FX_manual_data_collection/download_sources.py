#!/usr/bin/env python3
"""Obtain source files for the six specifically requested missing FX macro series.

Stores actual untouched downloads under data/raw/manual (no synthetic observations).
The Swiss post-2025 methodology and the OSEBX browser-download are intentionally
NOT silently substituted for incompatible indicators. Uses Python stdlib only.
"""
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/raw/manual'
REPORT = ROOT / 'data/reports'
RAW.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
           'Accept': 'text/html,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/csv,*/*'}

SOURCE_FILES = [
    ('AUD_RBA_F2_historical_1995_2013.xls', 'https://www.rba.gov.au/statistics/tables/xls-hist/f02dhist.xls', 'AUD', '1995-2013 historical file, includes 2005-01-03'),
    ('AUD_RBA_F2_current_2013_onwards.xlsx', 'https://www.rba.gov.au/statistics/tables/xls/f02d.xlsx', 'AUD', '2013 onward; full target coverage requires the historical workbook'),
    ('NZD_RBNZ_B2_1985_2017.xlsx', 'https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics/series/b/b2/hb2-daily-close-1985-2017.xlsx', 'NZD', '1985-2017: select 10-year secondary-market government bond yield'),
    ('NZD_RBNZ_B2_2018_current.xlsx', 'https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics/series/b/b2/hb2-daily-close.xlsx', 'NZD', '2018-current: select 10-year secondary-market government bond yield'),
    ('NOK_Norges_Bank_government_bonds_synthetic_daily.xlsx', 'https://www.norges-bank.no/contentassets/f851e16643634e72a63b13967e0e463e/government-bonds-synthetic-business-day.xlsx?v=18112022125538', 'NOK', 'select 10-year 2005-01-03 to 2018-12-31; methodology may differ from newer generic yields'),
]

SEK_GIST = 'https://gist.github.com/mtornwall/902c8330f5ad4ac64d6677d3586af231/raw/5e2da8116a0676bc945c79383e7fb01d9c08be3a/OMXS30.csv'
SEK_REFERENCE = 'https://gist.github.com/mtornwall/902c8330f5ad4ac64d6677d3586af231'

def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read()
    return data

def is_valid_spreadsheet(name, content):
    if len(content) < 500: return False, 'Too short'
    if name.endswith('.xls'):
        return (content[:8] == bytes.fromhex('D0CF11E0A1B11AE1'), 'XLS signature not found')
    if not content.startswith(b'PK\x03\x04'):
        return False, 'XLSX ZIP signature not found (maybe blocked/HTML)'
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            return '[Content_Types].xml' in z.namelist(), 'XLSX structure invalid'
    except (zipfile.BadZipFile, OSError) as e:
        return False, repr(e)

def download_one(name,url,series,desc):
    dest = RAW / name
    status,detail = 'FAILED',''
    if dest.exists():
        content = dest.read_bytes()
        valid,_ = is_valid_spreadsheet(name,content)
        if valid:
            status,detail = 'ALREADY_PRESENT','Existing original validated'
    if status == 'FAILED':
        try:
            content = fetch(url)
            valid,reason = is_valid_spreadsheet(name,content)
            if not valid: raise ValueError(reason)
            dest.write_bytes(content)
            status,detail = 'DOWNLOADED', 'Original bytes saved'
        except Exception as e:
            detail = f'{type(e).__name__}: {e}'[:300]
    return dict(series=series,output=str(dest.relative_to(ROOT)),status=status,bytes=dest.stat().st_size if status!='FAILED' else 0,
                sha256=hashlib.sha256(dest.read_bytes()).hexdigest() if status!='FAILED' else '',date_coverage='NOT VERIFIED: inspect workbook',source_url=url,note=desc,error=detail if status=='FAILED' else '')

def parse_sek_gist(raw):
    content = raw.decode('utf-8-sig',errors='replace').replace('\r','')
    rows = list(csv.DictReader(io.StringIO(content)))
    if not rows or 'Date' not in rows[0] or 'Close' not in rows[0]:
        raise ValueError(f'Unexpected Gist header; got {list(rows[0]) if rows else []}')
    output={}
    start,stop=dt.date(2005,1,3),dt.date(2008,11,19)
    for row in rows:
        try:
            day=dt.date.fromisoformat(row['Date'].strip())
            val=float(row['Close'].strip().replace(',',''))
        except (TypeError,ValueError,KeyError): continue
        if start<=day<=stop and val>0:
            output[day]=val
    days=sorted(output)
    if len(days)<800 or days[0] > dt.date(2005,1,10) or days[-1] < dt.date(2008,11,12):
        raise ValueError(f'Insufficient requested SEK data: {len(days)} rows; {days[0] if days else None} - {days[-1] if days else None}')
    return days,output

def download_sek():
    name='SEK_OMXS30_2005-01-03_to_2008-11-19.csv'
    dest=RAW/name
    src=RAW/'SEK_OMXS30_gist_original.csv'
    try:
        if src.is_file(): raw=src.read_bytes()
        else:
            raw=fetch(SEK_GIST)
            src.write_bytes(raw)
        days,values = parse_sek_gist(raw)
        with dest.open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['date','close'])
            for d in days: w.writerow([d.isoformat(),f'{values[d]:.8f}'])
        return dict(series='SEK',output=str(dest.relative_to(ROOT)),status='DOWNLOADED_AND_EXTRACTED',bytes=dest.stat().st_size,
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),date_coverage=f'{days[0]} to {days[-1]} ({len(days)} rows)',source_url=SEK_REFERENCE,
                    note='Unofficial public Gist, not Nasdaq; validate against existing Yahoo ^OMX on overlap 2008-11-17 to 2008-11-19',error='')
    except Exception as e:
        return dict(series='SEK',output=str(dest.relative_to(ROOT)),status='FAILED',bytes=0,sha256='',date_coverage='NOT VERIFIED',source_url=SEK_REFERENCE,
                    note='Use official Nasdaq historical page if community file unavailable',error=f'{type(e).__name__}: {e}'[:300])

def csv_check(path,start,end,min_obs):
    days=[]; bad=0
    with path.open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        cols=r.fieldnames or []
        datecol=next((x for x in cols if x.lower().strip() in ('date','datum','trade date')),None)
        valcol=next((x for x in cols if x.lower().strip() in ('close','last','value','price','yield','rate')),None)
        if not (datecol and valcol): return {'status':'UNKNOWN_COLUMNS','headers':cols}
        for row in r:
            try:
                d=dt.date.fromisoformat(row[datecol].strip()[:10]);v=float(row[valcol].strip().replace(',',''))
                if start<=d<=end and v==v: days.append(d)
            except (TypeError,ValueError): bad+=1
    days=sorted(set(days))
    gap=max(((b-a).days for a,b in zip(days,days[1:])),default=0)
    return {'status':'COVERED' if days and len(days)>=min_obs and days[0]<=start+dt.timedelta(days=10) and days[-1]>=end-dt.timedelta(days=10) and gap<=8 else 'INCOMPLETE',
            'valid_rows':len(days),'first':str(days[0]) if days else None,'last':str(days[-1]) if days else None,'max_calendar_gap_days':gap,'rejected_rows':bad}

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-sek',action='store_true',help='Only official central-bank original files')
    args=p.parse_args()
    manifest=[]
    print('Source downloads; raw originals kept without modifying data.\n')
    for name,url,series,desc in SOURCE_FILES:
        rec=download_one(name,url,series,desc)
        print(f"{series}: {rec['status']} - {name} {rec['error']}")
        manifest.append(rec)
    if not args.skip_sek:
        rec=download_sek()
        print(f"SEK: {rec['status']} - {rec['output']} {rec['error']}")
        manifest.append(rec)
    for series, source, note in [
         ('CHF','https://data.snb.ch/en/topics/ziredev/cube/rendeiduebd','2025-08 to 2025-12: new SNB methodology; select 10Y DAILY, compare definitions; see README'),
         ('NOK OSEBX','https://live.euronext.com/en/popout-page/getHistoricalPrice/NO0007035327-XOSL','2005-01-03 to 2025-12-31: use browser date filter and CSV export; beware GR vs price return')]:
        manifest.append(dict(series=series,output='data/raw/manual/ [BROWSER DOWNLOAD REQUIRED]',status='MANUAL_PENDING',bytes=0,sha256='',date_coverage='NOT VERIFIED',source_url=source,note=note,error=''))
    reportfile=REPORT/'download_manifest.csv'
    with reportfile.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    with (REPORT/'download_manifest.json').open('w',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    sek=RAW/'SEK_OMXS30_2005-01-03_to_2008-11-19.csv'
    if sek.exists():
        qa=csv_check(sek,dt.date(2005,1,3),dt.date(2008,11,19),800)
        (REPORT/'SEK_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
        print(f'SEK verification: {qa}')
    ok=sum(r['status'] in ('ALREADY_PRESENT','DOWNLOADED','DOWNLOADED_AND_EXTRACTED') for r in manifest)
    print(f'\nActual source downloads OK: {ok}/{len(manifest)} entries (5 official files + SEK + CHF/OSEBX manual).')
    print(f'Manifest saved to {reportfile}')
    print('Source coverage is NOT fully verified until originals are parsed and methods aligned.')
if __name__=='__main__': main()
