import io
import json
import time
import zipfile
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "abusech"
DATA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_CUTOFF = datetime(2025, 6, 1)

HEADERS = {"User-Agent": "TempCTI-KG-Research/1.0"}

def months_between(d1: datetime, d2: datetime) -> float:
    return max(0.0, (d2 - d1).days / 30.44)

def parse_dt(s: str) -> datetime | None:
    if not s or s.strip() in ("", "None", "null"):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def fetch_urlhaus() -> pd.DataFrame:
    print("\n=== URLhaus (CSV bulk download) ===")
    cache = DATA_DIR / "urlhaus_raw.csv"

    if cache.exists() and cache.stat().st_size > 10_000:
        print("  [CACHE] Ham CSV mevcut, yukleniyor...")
        raw_text = cache.read_text(encoding="utf-8", errors="replace")
    else:
        url = "https://urlhaus.abuse.ch/downloads/csv_recent/"
        print(f"  [DOWN] {url}")
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        raw_text = r.text
        cache.write_text(raw_text, encoding="utf-8")
        print(f"  [OK]   {len(raw_text)//1024} KB alindi")


    lines = [l for l in raw_text.splitlines() if not l.startswith("#") and l.strip()]
    if not lines:
        print("  [WARN] Veri bos")
        return pd.DataFrame()

    from io import StringIO
    df_raw = pd.read_csv(
        StringIO("\n".join(lines)),
        names=["id","dateadded","url","url_status","last_online",
               "threat","tags","urlhaus_link","reporter"],
        on_bad_lines="skip"
    )
    print(f"  {len(df_raw)} URL satirı okundu")

    records = []
    for _, row in df_raw.iterrows():
        first_seen = parse_dt(str(row.get("dateadded", "")))
        if first_seen is None:
            continue

        status = str(row.get("url_status", "")).strip()
        last_seen_str = str(row.get("last_online", ""))
        last_seen = parse_dt(last_seen_str)

        if last_seen is not None:
            observed = 1
        elif status.lower() == "offline":
            last_seen = first_seen
            observed = 1
        else:
            last_seen = COLLECTION_CUTOFF
            observed = 0

        records.append({
            "ioc_type":        "network_ioc_url",
            "source":          "urlhaus",
            "first_seen":      first_seen.isoformat(),
            "last_seen":       last_seen.isoformat(),
            "lifetime_months": round(months_between(first_seen, last_seen), 4),
            "observed":        observed,
            "status":          status,
        })

    df = pd.DataFrame(records)
    out = DATA_DIR / "urlhaus_lifetimes.csv"
    df.to_csv(out, index=False)
    print(f"  Kaydedildi: {out} ({len(df)} kayit)")
    return df


def fetch_threatfox() -> pd.DataFrame:
    print("\n=== ThreatFox (JSON bulk export) ===")
    cache = DATA_DIR / "threatfox_raw.json"

    if cache.exists() and cache.stat().st_size > 10_000:
        print("  [CACHE] Ham JSON mevcut, yukleniyor...")
        data = json.loads(cache.read_text())
    else:
        url = "https://threatfox.abuse.ch/export/json/recent/"
        print(f"  [DOWN] {url}")
        r = requests.get(url, headers=HEADERS, timeout=120)
        r.raise_for_status()
        data = r.json()
        cache.write_text(json.dumps(data))
        print(f"  [OK]   {len(data)} kayit alindi")


    records = []
    items_iter = []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                items_iter.extend(v)
    elif isinstance(data, list):
        items_iter = data

    print(f"  {len(items_iter)} IOC isleniyor...")
    for item in items_iter:
        first_seen = parse_dt(str(item.get("first_seen_utc",
                              item.get("first_seen", ""))))
        if first_seen is None:
            continue

        last_seen_str = item.get("last_seen_utc", item.get("last_seen", None))
        last_seen = parse_dt(str(last_seen_str)) if last_seen_str else None

        if last_seen:
            observed = 1
        else:
            last_seen = COLLECTION_CUTOFF
            observed = 0

        ioc_type = str(item.get("ioc_type", "unknown")).lower()
        if any(x in ioc_type for x in ["url", "domain", "ip"]):
            cat = "network_ioc_url"
        else:
            cat = "file_hash"

        records.append({
            "ioc_type":        cat,
            "source":          "threatfox",
            "first_seen":      first_seen.isoformat(),
            "last_seen":       last_seen.isoformat(),
            "lifetime_months": round(months_between(first_seen, last_seen), 4),
            "observed":        observed,
            "confidence":      item.get("confidence_level", 50),
        })

    df = pd.DataFrame(records)
    out = DATA_DIR / "threatfox_lifetimes.csv"
    df.to_csv(out, index=False)
    print(f"  Kaydedildi: {out} ({len(df)} kayit)")
    return df


def fetch_malwarebazaar() -> pd.DataFrame:
    print("\n=== MalwareBazaar (CSV bulk download) ===")
    cache = DATA_DIR / "malwarebazaar_raw.zip"

    if cache.exists() and cache.stat().st_size > 10_000:
        print("  [CACHE] Ham dosya mevcut, yukleniyor...")
        raw_bytes = cache.read_bytes()
    else:
        url2 = "https://bazaar.abuse.ch/export/csv/recent/"
        print(f"  [DOWN] {url2}")
        r = requests.get(url2, headers=HEADERS, timeout=120)
        r.raise_for_status()
        cache.write_bytes(r.content)
        raw_bytes = r.content
        print(f"  [OK]   {len(raw_bytes)//1024} KB alindi")


    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            raw_text = z.read(csv_name).decode("utf-8", errors="replace")
    except Exception:
        raw_text = raw_bytes.decode("utf-8", errors="replace")


    lines = [l for l in raw_text.splitlines() if not l.startswith("#") and l.strip()]
    print(f"  {len(lines)} veri satiri bulundu")


    import csv as csv_mod
    records = []

    MB_COLS = ["first_seen","sha256_hash","md5_hash","sha1_hash",
               "reporter","file_name","file_type","mime_type",
               "file_size","signature","tags","clamav",
               "vtpercent","imphash","ssdeep","tlsh"]
    for line in lines:
        try:
            row = list(csv_mod.reader([line]))[0]
            row = [v.strip().strip('"') for v in row]
        except Exception:
            continue
        if len(row) < 3:
            continue
        first_seen_str = row[0] if len(row) > 0 else ""
        first_seen = parse_dt(first_seen_str)
        if first_seen is None:
            continue

        last_seen = COLLECTION_CUTOFF
        observed = 0
        records.append({
            "ioc_type":        "file_hash",
            "source":          "malwarebazaar",
            "first_seen":      first_seen.isoformat(),
            "last_seen":       last_seen.isoformat(),
            "lifetime_months": round(months_between(first_seen, last_seen), 4),
            "observed":        observed,
        })

    df = pd.DataFrame(records)
    out = DATA_DIR / "malwarebazaar_lifetimes.csv"
    df.to_csv(out, index=False)
    print(f"  Kaydedildi: {out} ({len(df)} kayit)")
    return df


def main():
    print("=== abuse.ch IOC Yasam Suresi Toplayici ===")

    dfs = []
    for fn, name in [(fetch_urlhaus, "URLhaus"),
                     (fetch_threatfox, "ThreatFox"),
                     (fetch_malwarebazaar, "MalwareBazaar")]:
        try:
            df = fn()
            if not df.empty:
                dfs.append(df[["ioc_type","source","lifetime_months","observed"]])
        except Exception as e:
            print(f"  [ERR] {name}: {e}")
        time.sleep(1)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out = DATA_DIR / "ioc_lifetimes.csv"
        combined.to_csv(out, index=False)
        print(f"\n=== Birlesmis: {len(combined)} kayit ===")
        print(combined.groupby(["ioc_type","source"])["lifetime_months"]
              .describe()[["count","mean","50%"]].round(2))
        print(f"Kaydedildi: {out}")

if __name__ == "__main__":
    main()
