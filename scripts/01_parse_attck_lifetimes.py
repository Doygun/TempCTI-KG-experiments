import json
import re
from pathlib import Path
import pandas as pd
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "attck"


VERSION_DATES = {
    "1.0":  "2018-01-17", "2.0":  "2019-04-12",
    "3.0":  "2019-07-19", "4.0":  "2019-07-31",
    "5.0":  "2019-10-01", "5.1":  "2019-11-01", "5.2": "2019-12-01",
    "6.0":  "2019-10-24", "6.1":  "2020-01-15", "6.2": "2020-02-15",
    "6.3":  "2020-03-01",
    "7.0":  "2020-03-31", "7.1":  "2020-06-01", "7.2": "2020-09-01",
    "8.0":  "2020-10-27", "8.1":  "2021-01-01", "8.2": "2021-02-01",
    "9.0":  "2021-04-29",
    "10.0": "2021-10-21", "10.1": "2022-04-01",
    "11.0": "2022-04-25", "11.1": "2022-06-15",
    "11.2": "2022-07-18", "11.3": "2022-09-01",
    "12.0": "2022-10-25", "12.1": "2023-01-25",
    "13.0": "2023-04-25", "13.1": "2023-05-22",
    "14.0": "2023-10-31", "14.1": "2024-02-06",
    "15.0": "2024-05-21", "15.1": "2024-08-06",
    "16.0": "2024-10-14", "16.1": "2024-11-21",
    "17.0": "2025-04-01", "17.1": "2025-05-01",
    "18.0": "2025-10-01", "18.1": "2025-11-01",
    "19.0": "2026-04-01", "19.1": "2026-06-01",
}

RELATION_CATEGORY = {
    "uses":          "ttp_use",
    "subtechnique-of": "technique_hierarchy",
    "mitigates":     "mitigation",
    "detects":       "detection",
    "revoked-by":    "meta",
    "attributed-to": "attribution",
    "targets":       "targeting",
}

def version_to_date(version_str: str) -> datetime | None:
    v = version_str.strip()
    if v in VERSION_DATES:
        return datetime.fromisoformat(VERSION_DATES[v])

    v2 = v.lstrip("v")
    if v2 in VERSION_DATES:
        return datetime.fromisoformat(VERSION_DATES[v2])
    return None

def load_bundle(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        bundle = json.load(f)
    objects = {}
    for obj in bundle.get("objects", []):
        objects[obj["id"]] = obj
    return objects

def get_name(obj: dict) -> str:
    return obj.get("name", obj.get("id", "UNKNOWN"))

def parse_relationships(objects: dict) -> list[dict]:
    records = []
    for obj in objects.values():
        if obj.get("type") != "relationship":
            continue
        src_id  = obj.get("source_ref", "")
        tgt_id  = obj.get("target_ref", "")
        rel_type = obj.get("relationship_type", "unknown")
        revoked  = obj.get("revoked", False)
        deprecated = obj.get("x_mitre_deprecated", False)
        modified = obj.get("modified", "")
        created  = obj.get("created", "")

        src_obj = objects.get(src_id, {})
        tgt_obj = objects.get(tgt_id, {})

        records.append({
            "stix_id":       obj["id"],
            "relation_type": rel_type,
            "subject":       get_name(src_obj),
            "subject_type":  src_obj.get("type", ""),
            "object":        get_name(tgt_obj),
            "object_type":   tgt_obj.get("type", ""),
            "revoked":       revoked,
            "deprecated":    deprecated,
            "created":       created,
            "modified":      modified,
        })
    return records

def compute_lifetimes() -> pd.DataFrame:
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "manifest.json bulunamadi. Once 00_download_attck.py calistirin."
        )
    manifest = json.loads(manifest_path.read_text())
    versions = [v for v in manifest["versions"] if v not in manifest.get("failed", [])]

    print(f"{len(versions)} surum bulundu: {versions}")


    bundles: dict[str, dict] = {}
    for v in versions:
        json_path = DATA_DIR / v / f"enterprise-attack-{v}.json"
        if not json_path.exists():
            print(f"  [WARN] {json_path} bulunamadi, atlaniyor.")
            continue
        print(f"  [LOAD] v{v} ...")
        bundles[v] = load_bundle(json_path)

    sorted_versions = sorted(bundles.keys(),
                             key=lambda v: [int(x) for x in v.split(".")])


    version_rels: dict[str, dict] = {}
    for v in sorted_versions:
        rels = parse_relationships(bundles[v])
        version_rels[v] = {r["stix_id"]: r for r in rels}


    all_ids = set()
    for rels in version_rels.values():
        all_ids.update(rels.keys())

    print(f"\nToplam benzersiz iliski ID: {len(all_ids)}")

    lifetime_records = []
    collection_cutoff = datetime.fromisoformat("2026-07-01")

    for stix_id in all_ids:

        seen_in = [v for v in sorted_versions if stix_id in version_rels[v]]
        if not seen_in:
            continue

        first_v = seen_in[0]
        last_v  = seen_in[-1]
        record  = version_rels[last_v][stix_id]


        t_start_date = version_to_date(first_v)
        if t_start_date is None:
            continue


        is_terminated = record["revoked"] or record["deprecated"]


        t_end_date = version_to_date(last_v)
        if t_end_date is None:
            t_end_date = collection_cutoff


        def months_between(d1: datetime, d2: datetime) -> float:
            return max(0.0, (d2 - d1).days / 30.44)

        lifetime_months = months_between(t_start_date, t_end_date)

        rel_cat = RELATION_CATEGORY.get(record["relation_type"], "other")

        lifetime_records.append({
            "stix_id":          stix_id,
            "subject":          record["subject"],
            "subject_type":     record["subject_type"],
            "relation_type":    record["relation_type"],
            "relation_category":rel_cat,
            "object":           record["object"],
            "object_type":      record["object_type"],
            "version_start":    first_v,
            "version_end":      last_v,
            "t_start_date":     t_start_date.isoformat(),
            "t_end_date":       t_end_date.isoformat(),
            "lifetime_months":  round(lifetime_months, 2),
            "observed":         int(is_terminated),
            "revoked":          int(record["revoked"]),
            "deprecated":       int(record["deprecated"]),
        })

    df = pd.DataFrame(lifetime_records)
    print(f"\nToplam gozlem: {len(df)}")
    print(df["relation_category"].value_counts())

    out_path = DATA_DIR / "lifetimes.csv"
    df.to_csv(out_path, index=False)
    print(f"\nKaydedildi: {out_path}")


    stats = {}
    for cat, grp in df.groupby("relation_category"):
        stats[cat] = {
            "n_total":    int(len(grp)),
            "n_event":    int(grp["observed"].sum()),
            "n_censored": int((grp["observed"] == 0).sum()),
            "median_lifetime_months": round(grp["lifetime_months"].median(), 2),
            "mean_lifetime_months":   round(grp["lifetime_months"].mean(), 2),
        }

    stats_path = DATA_DIR / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Istatistikler kaydedildi: {stats_path}")

    return df

if __name__ == "__main__":
    df = compute_lifetimes()
    print("\n--- Ilk 5 kayit ---")
    print(df.head())
