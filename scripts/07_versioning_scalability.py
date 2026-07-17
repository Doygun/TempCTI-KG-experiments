import json
import time
import platform
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RES_DIR  = BASE_DIR / "results"

THETA      = 0.50
MONTH_DAYS = 30.44


def task4_versioning_mae():
    print("=== Task 4: Versioning MAE ===")
    params = json.loads((RES_DIR / "survival_params.json").read_text())

    rows = []
    a = pd.read_csv(DATA_DIR / "attck" / "lifetimes.csv")
    a = a[a["observed"] == 1]
    for _, r in a.iterrows():
        rows.append({"category": r["relation_category"],
                     "true_days": r["lifetime_months"] * MONTH_DAYS})
    b = pd.read_csv(DATA_DIR / "abusech" / "ioc_lifetimes.csv")
    b = b[(b["observed"] == 1) & (b["lifetime_months"] >= 0.01)]
    for _, r in b.iterrows():
        rows.append({"category": r["ioc_type"],
                     "true_days": r["lifetime_months"] * MONTH_DAYS})
    ev = pd.DataFrame(rows)
    print(f"Acik olay sayisi: {len(ev)}")

    out_cat = {}
    for cat, g in ev.groupby("category"):
        p = params.get(cat, {})
        lam  = p.get("exponential", {}).get("lambda", 0.01)
        wb   = p.get("weibull", {})
        eta  = wb.get("scale_eta_months", 1.0 / max(lam, 1e-9))
        beta = wb.get("shape_beta", 1.0)

        t_exp_d = np.log(1.0 / THETA) / max(lam, 1e-9) * MONTH_DAYS
        t_wb_d  = eta * np.power(np.log(1.0 / THETA), 1.0 / beta) * MONTH_DAYS

        err_exp = np.abs(t_exp_d - g["true_days"].values)
        err_wb  = np.abs(t_wb_d - g["true_days"].values)
        out_cat[cat] = {
            "n": int(len(g)),
            "retract_at_exp_days": round(float(t_exp_d), 1),
            "retract_at_weibull_days": round(float(t_wb_d), 1),
            "mae_exp_days": round(float(err_exp.mean()), 1),
            "mae_weibull_days": round(float(err_wb.mean()), 1),
            "median_ae_weibull_days": round(float(np.median(err_wb)), 1),
        }
        print(f"  {cat:22s} n={len(g):6d} "
              f"MAE(exp)={err_exp.mean():9.1f}d "
              f"MAE(wb)={err_wb.mean():9.1f}d "
              f"medAE(wb)={np.median(err_wb):7.1f}d")

    return {"theta": THETA, "n_events": int(len(ev)),
            "note": "signal-free (kappa-only) regime; with explicit "
                    "signals retraction time equals signal time",
            "per_category": out_cat}


def task5_scalability():
    print("\n=== Task 5: Scalability micro-benchmark ===")
    rng = np.random.default_rng(0)
    results = []
    for n_edges in [5_000, 50_000, 500_000]:

        lam = rng.choice([0.345, 0.001, 0.004, 0.0004], size=n_edges)
        age_mo = rng.uniform(0, 24, size=n_edges)


        _ = np.exp(-lam * age_mo) >= THETA

        n_rep = 100
        t0 = time.perf_counter()
        for _ in range(n_rep):
            mask = np.exp(-lam * age_mo) >= THETA
        t1 = time.perf_counter()
        ms = (t1 - t0) / n_rep * 1000
        results.append({"n_edges": n_edges,
                        "mean_query_ms": round(ms, 3),
                        "n_active": int(mask.sum())})
        print(f"  {n_edges:>7,} kenar: {ms:.3f} ms/sorgu")

    return {"hardware": platform.processor() or platform.machine(),
            "n_repeats": 100, "results": results}


def main():
    out = {
        "task4_versioning": task4_versioning_mae(),
        "task5_scalability": task5_scalability(),
    }
    p = RES_DIR / "versioning_scalability.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nKaydedildi: {p}")


if __name__ == "__main__":
    main()
