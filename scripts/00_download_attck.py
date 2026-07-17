import json
import re
import time
import requests
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data" / "attck"
DATA_DIR.mkdir(parents=True, exist_ok=True)

REPO_OWNER = "mitre-attack"
REPO_NAME  = "attack-stix-data"
BRANCH     = "master"
ENTERPRISE_PATH = "enterprise-attack"

HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "TempCTI-KG-Downloader/1.0"
}

def list_files_and_versions():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{ENTERPRISE_PATH}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    items = r.json()

    entries = []
    pattern = re.compile(r"enterprise-attack-([\d.]+)\.json$")
    for item in items:
        if item["type"] == "file":
            m = pattern.match(item["name"])
            if m:
                version = m.group(1)
                download_url = item["download_url"]
                entries.append((version, download_url))


    entries.sort(key=lambda x: [int(p) for p in x[0].split(".")])
    return entries

def download_version(version: str, download_url: str):
    out_dir = DATA_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"enterprise-attack-{version}.json"
    out_path = out_dir / filename

    if out_path.exists() and out_path.stat().st_size > 100_000:
        print(f"  [SKIP] v{version} zaten mevcut ({out_path.stat().st_size/1e6:.1f} MB)")
        return out_path

    print(f"  [DOWN] v{version} ...")
    r = requests.get(download_url, headers=HEADERS, timeout=180, stream=True)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=131072):
            f.write(chunk)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  [OK]   v{version} -> {out_path.name} ({size_mb:.1f} MB)")
    return out_path

def main():
    print("=== ATT&CK STIX Surum Indirici ===")
    print(f"Hedef dizin: {DATA_DIR}")

    print("\nGitHub API ile dosya listesi aliniyor...")
    entries = list_files_and_versions()
    versions = [v for v, _ in entries]
    print(f"Bulunan {len(entries)} surum: {versions}")

    failed = []
    for version, dl_url in entries:
        try:
            download_version(version, dl_url)
            time.sleep(0.3)
        except Exception as e:
            print(f"  [ERR] v{version}: {e}")
            failed.append(version)

    print(f"\nTamamlandi. Basari: {len(entries)-len(failed)}/{len(entries)}")
    if failed:
        print(f"Basarisiz: {failed}")

    manifest = {"versions": versions, "failed": failed}
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest kaydedildi: {DATA_DIR/'manifest.json'}")

if __name__ == "__main__":
    main()
