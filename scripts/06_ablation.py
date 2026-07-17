import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss
from lifelines import WeibullFitter

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

vp = __import__("04_validity_prediction")

RES_DIR = BASE_DIR / "results"


def main():
    print("=== Ablation (temporal split) ===\n")
    df = vp.load_facts()
    params, train_df = vp.fit_train_params(df)
    ev = vp.build_eval_frame(df, "query_dates")
    y = ev["y_true"].values
    t = ev["elapsed"].values
    cats = ev["category"].values


    T = train_df["lifetime_months"].values
    E = train_df["observed"].values.astype(int)
    wf = WeibullFitter().fit(T, event_observed=E)
    g_eta, g_beta = float(wf.lambda_), float(wf.rho_)
    print(f"Global Weibull: eta={g_eta:.3f} beta={g_beta:.3f}")

    lam  = np.array([params.get(c, {}).get("lambda", 0.01) for c in cats])
    eta  = np.array([params.get(c, {}).get("weibull_eta", 100.0) for c in cats])
    beta = np.array([params.get(c, {}).get("weibull_beta", 1.0) for c in cats])

    configs = {
        "Full model (TempCTI-KG, Weibull)":
            np.exp(-np.power(np.clip(t, 0, None) / eta, beta)),
        "w/o Weibull shape (Exponential)":
            np.exp(-lam * t),
        "w/o relation-type conditioning":
            np.exp(-np.power(np.clip(t, 0, None) / g_eta, g_beta)),
        "w/o temporal modeling (Static KG)":
            np.ones(len(ev)),
    }

    results = {}
    for name, p in configs.items():
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
        brier = brier_score_loss(y, np.clip(p, 0, 1))
        results[name] = {"auc": round(auc, 4), "brier": round(brier, 4)}
        print(f"  {name:38s} AUC={auc:.4f} Brier={brier:.4f}")

    out = RES_DIR / "ablation_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nKaydedildi: {out}")


if __name__ == "__main__":
    main()
