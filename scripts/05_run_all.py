import subprocess
import sys
import argparse
from pathlib import Path

SCRIPTS = {
    0: ("00_download_attck.py",       "ATT&CK versiyonlari indiriliyor"),
    1: ("01_parse_attck_lifetimes.py", "ATT&CK yasam sureleri hesaplaniyor"),
    2: ("02_fetch_abusech.py",         "abuse.ch IOC verileri cekiliyor"),
    3: ("02c_parse_urlhaus_full.py",   "URLhaus tam dump parse (survivorship-bias korumali)"),
    4: ("02b_rebuild_merged.py",       "Birlesik IOC dosyasi (sansur duzeltmeli)"),
    5: ("03_survival_analysis.py",     "Survival analysis + grafikler"),
    6: ("04_validity_prediction.py",   "Gecerlilik tahmini (temporal split + hibrit)"),
    7: ("05_downstream_fp.py",         "Downstream stale-fact simulasyonu"),
    8: ("06_ablation.py",              "Ablasyon calismasi"),
    9: ("07_versioning_scalability.py", "Versioning MAE + scalability"),
    10: ("08_replay_eval.py",          "Gercek-veri replay (URLhaus)"),
}

BASE_DIR = Path(__file__).resolve().parent.parent

def run_step(step_id: int):
    script, desc = SCRIPTS[step_id]
    script_path = BASE_DIR / "scripts" / script
    print(f"\n{'='*60}")
    print(f"  ADIM {step_id}: {desc}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR)
    )
    if result.returncode != 0:
        print(f"  [HATA] Adim {step_id} basarisiz oldu!")
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", type=int, nargs="*", default=[],
                        help="Atlanacak adim numaralari")
    parser.add_argument("--only", type=int, default=None,
                        help="Sadece bu adimi calistir")
    args = parser.parse_args()

    if args.only is not None:
        run_step(args.only)
        return

    for step_id in sorted(SCRIPTS.keys()):
        if step_id in args.skip:
            print(f"\n[SKIP] Adim {step_id} atlaniyor...")
            continue
        success = run_step(step_id)
        if not success:
            print(f"\nPipeline adim {step_id}'de durdu.")
            sys.exit(1)

    print("\n\n=== TUM PIPELINE TAMAMLANDI ===")
    print("Sonuclar: results/")
    print("Grafikler: results/figures/")

if __name__ == "__main__":
    main()
