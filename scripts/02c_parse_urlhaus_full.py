import io
import zipfile
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "abusech"

MONTH_DAYS = 30.44
EPS_LIFETIME_MO = 0.0164


def main():
    zpath = DATA_DIR / "urlhaus_full.zip"
    print(f"Parse ediliyor: {zpath}")
    z = zipfile.ZipFile(zpath)
    raw = z.read("csv.txt").decode("utf-8", errors="replace")
    lines = [l for l in raw.splitlines() if l and not l.startswith("#")]
    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        names=["id", "dateadded", "url", "url_status", "last_online",
               "threat", "tags", "urlhaus_link", "reporter"],
        on_bad_lines="skip")
    df["dateadded"] = pd.to_datetime(df["dateadded"], errors="coerce")
    df["last_online"] = pd.to_datetime(df["last_online"], errors="coerce")
    df = df.dropna(subset=["dateadded"])
    print(f"Toplam kayit: {len(df)}")

    snapshot_end = df["dateadded"].max()

    prune_horizon = df.loc[df["url_status"] == "offline", "last_online"].min()
    print(f"Dump zamani: {snapshot_end} | Budama ufku: {prune_horizon}")

    win = df[df["dateadded"] >= prune_horizon].copy()
    print(f"Tam-kapsama penceresi (first_seen >= {prune_horizon.date()}): "
          f"{len(win)} kayit")


    offline = win["url_status"].str.lower().eq("offline")
    has_lo = win["last_online"].notna()

    win["observed"] = 0
    win["last_seen"] = snapshot_end

    m1 = offline & has_lo
    win.loc[m1, "observed"] = 1
    win.loc[m1, "last_seen"] = win.loc[m1, "last_online"]

    m2 = offline & ~has_lo
    win.loc[m2, "observed"] = 1
    win.loc[m2, "last_seen"] = win.loc[m2, "dateadded"]


    lt = ((win["last_seen"] - win["dateadded"]).dt.total_seconds()
          / 86400.0 / MONTH_DAYS).clip(lower=0.0)

    zero_ev = (win["observed"] == 1) & (lt < 0.005)
    lt[zero_ev] = EPS_LIFETIME_MO
    win["lifetime_months"] = lt.round(4)

    out = pd.DataFrame({
        "ioc_type": "network_ioc_url",
        "source": "urlhaus",
        "first_seen": win["dateadded"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "last_seen": win["last_seen"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "lifetime_months": win["lifetime_months"],
        "observed": win["observed"].astype(int),
        "status": win["url_status"],
    })
    p1 = DATA_DIR / "urlhaus_lifetimes.csv"
    out.to_csv(p1, index=False)
    print(f"Kaydedildi: {p1} | n={len(out)} "
          f"events={int(out.observed.sum())} "
          f"censored={int((out.observed == 0).sum())}")
    print(f"  first_seen: {out.first_seen.min()} -> {out.first_seen.max()}")


    p2 = DATA_DIR / "urlhaus_full_parsed.csv"
    win[["dateadded", "url_status", "last_online"]].to_csv(p2, index=False)
    print(f"Kaydedildi: {p2}")


if __name__ == "__main__":
    main()
