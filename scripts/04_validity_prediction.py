import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from lifelines import ExponentialFitter, WeibullFitter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RES_DIR  = BASE_DIR / "results"


SPLIT = {
    "attck": {
        "cutoff": pd.Timestamp("2024-12-31"),
        "query_dates": [pd.Timestamp(d) for d in
                        ["2025-03-31", "2025-06-30", "2025-09-30",
                         "2025-12-31", "2026-03-31", "2026-06-24"]],
        "train_query_dates": [pd.Timestamp(d) for d in
                              ["2022-06-30", "2022-12-31", "2023-06-30",
                               "2023-12-31", "2024-06-30", "2024-12-31"]],
    },
    "abusech": {
        "cutoff": pd.Timestamp("2026-05-31"),
        "query_dates": [pd.Timestamp(d) for d in
                        ["2026-06-05", "2026-06-11", "2026-06-17",
                         "2026-06-23", "2026-06-29", "2026-07-05"]],
        "train_query_dates": [pd.Timestamp(d) for d in
                              ["2026-04-20", "2026-04-27", "2026-05-04",
                               "2026-05-11", "2026-05-18", "2026-05-25"]],
    },
}

FIXED_TTL_MO = {
    "network_ioc_url":      1.0,
    "file_hash":            3.0,
    "ttp_use":              12.0,
    "technique_hierarchy":  18.0,
    "mitigation":           18.0,
    "attribution":          24.0,
    "meta":                 18.0,
    "other":                6.0,
}
N_BOOT = 1000
MONTH_DAYS = 30.44


def load_facts() -> pd.DataFrame:
    frames = []

    a = pd.read_csv(DATA_DIR / "attck" / "lifetimes.csv",
                    parse_dates=["t_start_date"])
    a = a.rename(columns={"relation_category": "category",
                          "t_start_date": "first_seen"})
    a["end_date"] = a["first_seen"] + pd.to_timedelta(
        a["lifetime_months"] * MONTH_DAYS, unit="D")
    a["src_group"] = "attck"
    frames.append(a[["category", "first_seen", "end_date",
                     "lifetime_months", "observed", "src_group"]])

    b = pd.read_csv(DATA_DIR / "abusech" / "ioc_lifetimes.csv",
                    parse_dates=["first_seen"])
    b = b.rename(columns={"ioc_type": "category"})
    b["end_date"] = b["first_seen"] + pd.to_timedelta(
        b["lifetime_months"] * MONTH_DAYS, unit="D")
    b["src_group"] = "abusech"
    frames.append(b[["category", "first_seen", "end_date",
                     "lifetime_months", "observed", "src_group"]])

    df = pd.concat(frames, ignore_index=True)
    df = df[df["lifetime_months"] >= 0.01].reset_index(drop=True)
    df["fact_id"] = df.index
    return df


def fit_train_params(df: pd.DataFrame) -> dict:
    parts = []
    for grp, cfg in SPLIT.items():
        cutoff = cfg["cutoff"]
        tr = df[(df["src_group"] == grp) &
                (df["first_seen"] <= cutoff)].copy()
        after = tr["end_date"] > cutoff
        tr.loc[after, "observed"] = 0
        tr.loc[after, "lifetime_months"] = (
            (cutoff - tr.loc[after, "first_seen"]).dt.total_seconds()
            / 86400.0 / MONTH_DAYS)
        parts.append(tr)
    train = pd.concat(parts, ignore_index=True)
    train = train[train["lifetime_months"] >= 0.005]

    params = {}
    for cat, g in train.groupby("category"):
        T = g["lifetime_months"].values
        E = g["observed"].values.astype(int)
        entry = {"n_train": int(len(g)), "n_events": int(E.sum())}
        if E.sum() >= 1:
            ef = ExponentialFitter().fit(T, event_observed=E)
            entry["lambda"] = float(1.0 / ef.lambda_)
        else:
            entry["lambda"] = 1e-6
        if E.sum() >= 5:
            try:
                wf = WeibullFitter().fit(T, event_observed=E)
                entry["weibull_eta"]  = float(wf.lambda_)
                entry["weibull_beta"] = float(wf.rho_)
            except Exception:
                pass
        if "weibull_eta" not in entry:
            entry["weibull_eta"]  = 1.0 / max(entry["lambda"], 1e-9)
            entry["weibull_beta"] = 1.0

        entry["median_lifetime_mo"] = float(np.log(2) / max(entry["lambda"], 1e-9))
        params[cat] = entry
    return params, train


def build_eval_frame(df: pd.DataFrame, which: str = "query_dates") -> pd.DataFrame:
    rows = []
    for grp, cfg in SPLIT.items():
        sub = df[df["src_group"] == grp]
        for t_q in cfg[which]:
            m = sub[sub["first_seen"] <= t_q].copy()

            known = ~((m["observed"] == 0) & (m["end_date"] <= t_q))
            m = m[known]
            m["y_true"]  = (m["end_date"] > t_q).astype(int)
            m["elapsed"] = ((t_q - m["first_seen"]).dt.total_seconds()
                            / 86400.0 / MONTH_DAYS)
            m["t_q"] = t_q
            rows.append(m[["category", "elapsed", "y_true", "t_q", "fact_id"]])
    return pd.concat(rows, ignore_index=True)


def predict_all(ev: pd.DataFrame, params: dict,
                logreg, hybrid, enc_cats: list) -> dict:
    cats = ev["category"].values
    t    = ev["elapsed"].values

    lam = np.array([params.get(c, {}).get("lambda", 0.01) for c in cats])
    eta = np.array([params.get(c, {}).get("weibull_eta", 100.0) for c in cats])
    beta = np.array([params.get(c, {}).get("weibull_beta", 1.0) for c in cats])
    ttl = np.array([FIXED_TTL_MO.get(c, 6.0) for c in cats])

    lam_glob = np.mean([p["lambda"] for p in params.values()])

    preds = {
        "static_kg":      np.ones(len(ev)),
        "fixed_ttl":      (t <= ttl).astype(float),
        "global_decay":   np.exp(-lam_glob * t),
        "tempcti_exp":    np.exp(-lam * t),
        "tempcti_weibull": np.exp(-np.power(np.clip(t, 0, None) / eta, beta)),
    }
    X = logreg_features(cats, t, enc_cats)
    preds["logreg_temporal"] = logreg.predict_proba(X)[:, 1]
    Xh = hybrid_features(cats, t, enc_cats, params)
    preds["tempcti_hybrid"] = hybrid.predict_proba(Xh)[:, 1]
    return preds


def logreg_features(cats, t, enc_cats):
    X_cat = np.zeros((len(cats), len(enc_cats)))
    idx = {c: i for i, c in enumerate(enc_cats)}
    for j, c in enumerate(cats):
        if c in idx:
            X_cat[j, idx[c]] = 1.0
    t = np.asarray(t, dtype=float)
    return np.hstack([X_cat,
                      t.reshape(-1, 1),
                      np.log1p(t).reshape(-1, 1),
                      X_cat * t.reshape(-1, 1)])


def kappa_features(cats, t, params):
    t = np.asarray(t, dtype=float)
    lam  = np.array([params.get(c, {}).get("lambda", 0.01) for c in cats])
    eta  = np.array([params.get(c, {}).get("weibull_eta", 100.0) for c in cats])
    beta = np.array([params.get(c, {}).get("weibull_beta", 1.0) for c in cats])
    k_exp = np.exp(-lam * t)
    k_wb  = np.exp(-np.power(np.clip(t, 0, None) / eta, beta))

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    return np.column_stack([k_exp, k_wb, logit(k_exp), logit(k_wb)])


def hybrid_features(cats, t, enc_cats, params):
    return np.hstack([logreg_features(cats, t, enc_cats),
                      kappa_features(cats, t, params)])


def train_logreg(train_df: pd.DataFrame, enc_cats: list):
    ev = build_eval_frame(train_df, "train_query_dates")
    X = logreg_features(ev["category"].values, ev["elapsed"].values, enc_cats)
    y = ev["y_true"].values
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(X, y)
    return lr


def train_hybrid(train_df: pd.DataFrame, enc_cats: list, params: dict):
    ev = build_eval_frame(train_df, "train_query_dates")
    X = hybrid_features(ev["category"].values, ev["elapsed"].values,
                        enc_cats, params)
    y = ev["y_true"].values
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(X, y)
    return lr


def delong_test(y_true, p1, p2):
    def midrank(x):
        order = np.argsort(x)
        ranks = np.empty(len(x))
        i = 0
        sorted_x = x[order]
        while i < len(x):
            j = i
            while j < len(x) and sorted_x[j] == sorted_x[i]:
                j += 1
            ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
            i = j
        return ranks

    def auc_var(y, preds):
        pos = preds[y == 1]; neg = preds[y == 0]
        m, n = len(pos), len(neg)
        allv = np.concatenate([pos, neg])
        tx = midrank(pos); ty = midrank(neg); tz = midrank(allv)
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01 = (tz[:m] - tx) / n
        v10 = 1.0 - (tz[m:] - ty) / m
        return auc, v01, v10

    auc1, v01_1, v10_1 = auc_var(y_true, p1)
    auc2, v01_2, v10_2 = auc_var(y_true, p2)
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    var = (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / len(v01_1) + \
          (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / len(v10_1)
    z = (auc1 - auc2) / np.sqrt(max(var, 1e-12))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, p


def main():
    print("=== Task 1: Validity Prediction (temporal split) ===\n")
    df = load_facts()
    print(f"Toplam gercek: {len(df)}")

    params, train_df = fit_train_params(df)
    print(f"Train toplami: {len(train_df)}")
    for c, p in sorted(params.items()):
        print(f"  {c:22s} lambda={p['lambda']:.6f} "
              f"(n={p['n_train']}, events={p['n_events']})")

    enc_cats = sorted(df["category"].unique())

    logreg = train_logreg(train_df, enc_cats)
    hybrid = train_hybrid(train_df, enc_cats, params)


    ev = build_eval_frame(df, "query_dates")
    print(f"\nTest (fact, t_q) cifti: {len(ev)} "
          f"(pozitif orani: {ev['y_true'].mean():.3f})")

    preds = predict_all(ev, params, logreg, hybrid, enc_cats)
    y = ev["y_true"].values


    results = {}
    rng_master = np.random.default_rng(7)
    fact_ids = ev["fact_id"].values
    uniq_facts = np.unique(fact_ids)

    order = np.argsort(fact_ids, kind="stable")
    sorted_fids = fact_ids[order]
    starts = np.searchsorted(sorted_fids, uniq_facts, side="left")
    ends   = np.searchsorted(sorted_fids, uniq_facts, side="right")

    boot_idx = []
    for _ in range(N_BOOT):
        chosen = rng_master.integers(0, len(uniq_facts), size=len(uniq_facts))
        idx = np.concatenate([order[starts[c]:ends[c]] for c in chosen])
        boot_idx.append(idx)

    for method, p in preds.items():
        aucs, aps, briers = [], [], []
        for idx in boot_idx:
            yt, pt = y[idx], p[idx]
            if len(np.unique(yt)) < 2:
                continue
            aucs.append(roc_auc_score(yt, pt))
            aps.append(average_precision_score(yt, pt))
            briers.append(brier_score_loss(yt, np.clip(pt, 0, 1)))

        def ci(vals):
            vals = np.array(vals)
            m = vals.mean()
            if len(vals) < 10 or vals.std() == 0:
                return m, 0.0
            lo, hi = np.percentile(vals, [2.5, 97.5])
            return m, (hi - lo) / 2.0

        auc_m, auc_h = ci(aucs)
        ap_m,  ap_h  = ci(aps)
        br_m,  br_h  = ci(briers)


        per_tq = []
        for t_q in sorted(ev["t_q"].unique()):
            mask = ev["t_q"] == t_q
            yt, pt = y[mask.values], p[mask.values]
            if len(np.unique(yt)) < 2:
                continue
            per_tq.append({
                "t_q": str(pd.Timestamp(t_q).date()),
                "auc": round(roc_auc_score(yt, pt), 4),
                "ap": round(average_precision_score(yt, pt), 4),
                "brier": round(brier_score_loss(yt, np.clip(pt, 0, 1)), 4),
                "n": int(mask.sum()),
            })

        results[method] = {
            "auc_mean": round(auc_m, 4),  "auc_ci95": round(auc_h, 4),
            "ap_mean": round(ap_m, 4),    "ap_ci95": round(ap_h, 4),
            "brier_mean": round(br_m, 4), "brier_ci95": round(br_h, 4),
            "per_tq": per_tq,
        }
        print(f"  {method:18s} AUC={auc_m:.3f}±{auc_h:.3f} "
              f"AP={ap_m:.3f}±{ap_h:.3f} Brier={br_m:.3f}±{br_h:.3f}")


    comparisons = [
        ("tempcti_weibull", "logreg_temporal"),
        ("tempcti_hybrid", "logreg_temporal"),
        ("tempcti_hybrid", "tempcti_weibull"),
        ("tempcti_weibull", "fixed_ttl"),
        ("tempcti_weibull", "global_decay"),
        ("tempcti_weibull", "tempcti_exp"),
    ]
    results["delong"] = []
    for m1, m2 in comparisons:
        a1, a2, z, pval = delong_test(y, preds[m1], preds[m2])
        results["delong"].append({
            "comparison": f"{m1} vs {m2}",
            "auc_1": round(a1, 4), "auc_2": round(a2, 4),
            "delta_auc": round(a1 - a2, 4),
            "z": round(z, 3), "p_value": float(f"{pval:.2e}"),
        })
        print(f"DeLong: {m1} ({a1:.4f}) vs {m2} ({a2:.4f}) "
              f"z={z:.2f} p={pval:.2e}")

    results["protocol"] = {
        "split": {g: {"cutoff": str(c["cutoff"].date()),
                      "query_dates": [str(d.date()) for d in c["query_dates"]]}
                  for g, c in SPLIT.items()},
        "n_eval_pairs": int(len(ev)),
        "positive_rate": round(float(ev["y_true"].mean()), 4),
        "n_bootstrap": N_BOOT,
        "train_params": {c: {k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in p.items()}
                         for c, p in params.items()},
    }

    out = RES_DIR / "validity_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nKaydedildi: {out}")


if __name__ == "__main__":
    main()
