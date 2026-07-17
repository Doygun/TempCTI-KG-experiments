import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
vp = __import__("04_validity_prediction")

RES_DIR = BASE_DIR / "results"


def main():
    df = vp.load_facts()
    params, train_df = vp.fit_train_params(df)
    enc = sorted(df["category"].unique())
    logreg = vp.train_logreg(train_df, enc)
    hybrid = vp.train_hybrid(train_df, enc, params)

    rows = []
    for grp, cfg in vp.SPLIT.items():
        sub = df[df["src_group"] == grp]
        for t_q in cfg["query_dates"]:
            m = sub[sub["first_seen"] <= t_q].copy()
            known = ~((m["observed"] == 0) & (m["end_date"] <= t_q))
            m = m[known]
            m["y_true"] = (m["end_date"] > t_q).astype(int)
            m["elapsed"] = ((t_q - m["first_seen"]).dt.total_seconds()
                            / 86400.0 / 30.44)
            m["t_q"] = t_q
            m["grp"] = grp
            rows.append(m[["category", "elapsed", "y_true",
                           "t_q", "grp", "fact_id"]])
    ev = pd.concat(rows, ignore_index=True)
    preds = vp.predict_all(ev, params, logreg, hybrid, enc)
    y = ev["y_true"].values

    out = {}
    for grp in ["attck", "abusech"]:
        mask = (ev["grp"] == grp).values
        yt = y[mask]
        out[grp] = {"n_pairs": int(mask.sum()),
                    "positive_rate": round(float(yt.mean()), 4)}
        for m in ["static_kg", "fixed_ttl", "global_decay",
                  "logreg_temporal", "tempcti_exp", "tempcti_weibull"]:
            pt = preds[m][mask]
            entry = {"brier": round(float(
                brier_score_loss(yt, np.clip(pt, 0, 1))), 4)}
            if len(np.unique(yt)) > 1:
                entry["auc"] = round(float(roc_auc_score(yt, pt)), 4)
            out[grp][m] = entry
            print(grp, m, entry)

    (RES_DIR / "per_source_auc.json").write_text(json.dumps(out, indent=2))
    print("Kaydedildi:", RES_DIR / "per_source_auc.json")


if __name__ == "__main__":
    main()
