import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "abusech"


def months_between(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((b - a).dt.total_seconds() / 86400.0 / 30.44).clip(lower=0.0)


def main():
    frames = []


    u = pd.read_csv(DATA_DIR / "urlhaus_lifetimes.csv",
                    parse_dates=["first_seen", "last_seen"])
    snapshot_end = u["last_seen"].max()
    print(f"URLhaus snapshot end: {snapshot_end}")

    online = u["status"].str.lower().eq("online")

    u.loc[online, "observed"] = 0
    u.loc[online, "last_seen"] = snapshot_end
    u["lifetime_months"] = months_between(u["first_seen"], u["last_seen"]).round(4)


    zero_event = (u["observed"] == 1) & (u["lifetime_months"] < 0.01)
    u.loc[zero_event, "lifetime_months"] = 0.0164
    print(f"URLhaus: {int(zero_event.sum())} zero-lifetime event -> 0.5 gun atandi")
    print(f"URLhaus: {len(u)} kayit | events={int(u.observed.sum())} "
          f"censored={int((u.observed == 0).sum())}")
    frames.append(u[["ioc_type", "source", "first_seen",
                     "lifetime_months", "observed"]])


    t = pd.read_csv(DATA_DIR / "threatfox_lifetimes.csv",
                    parse_dates=["first_seen", "last_seen"])


    tf_horizon = t.loc[t["observed"] == 1, "last_seen"].min()
    n_before = len(t)
    t = t[t["first_seen"] >= tf_horizon].copy()
    print(f"ThreatFox full-coverage: {tf_horizon} sonrasi "
          f"{len(t)}/{n_before} kayit tutuldu")

    tf_snapshot = t["last_seen"].max()
    cens = t["observed"] == 0
    t.loc[cens, "last_seen"] = tf_snapshot
    t["lifetime_months"] = months_between(t["first_seen"], t["last_seen"]).round(4)
    print(f"ThreatFox: {len(t)} kayit | events={int(t.observed.sum())} "
          f"censored={int((t.observed == 0).sum())}")
    frames.append(t[["ioc_type", "source", "first_seen",
                     "lifetime_months", "observed"]])


    m = pd.read_csv(DATA_DIR / "malwarebazaar_lifetimes.csv",
                    parse_dates=["first_seen", "last_seen"])
    mb_snapshot = m["last_seen"].max()
    cens = m["observed"] == 0
    m.loc[cens, "last_seen"] = mb_snapshot
    m["lifetime_months"] = months_between(m["first_seen"], m["last_seen"]).round(4)
    print(f"MalwareBazaar: {len(m)} kayit | events={int(m.observed.sum())} "
          f"censored={int((m.observed == 0).sum())}")
    frames.append(m[["ioc_type", "source", "first_seen",
                     "lifetime_months", "observed"]])

    merged = pd.concat(frames, ignore_index=True)
    out = DATA_DIR / "ioc_lifetimes.csv"
    merged.to_csv(out, index=False)
    print(f"\nKaydedildi: {out} ({len(merged)} kayit)")
    print(merged.groupby(["source"])["observed"].agg(["count", "sum"]))


if __name__ == "__main__":
    main()
