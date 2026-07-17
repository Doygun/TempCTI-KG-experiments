import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.stats import weibull_min
from lifelines import KaplanMeierFitter, WeibullFitter, ExponentialFitter, CoxPHFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
RES_DIR   = BASE_DIR / "results"
FIG_DIR   = RES_DIR / "figures"
RES_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


PALETTE = {
    "network_ioc_url": "#e63946",
    "file_hash":       "#f4a261",
    "ioc_domain_ip":   "#e76f51",
    "ttp_use":         "#2a9d8f",
    "technique_hierarchy": "#457b9d",
    "mitigation":      "#1d3557",
    "attribution":     "#6a4c93",
    "other":           "#adb5bd",
}
LABEL_MAP = {
    "network_ioc_url":      "Network IOC (URL)",
    "file_hash":            "File Hash IOC",
    "ttp_use":              "TTP/Technique Use",
    "technique_hierarchy":  "Technique (ATT&CK)",
    "mitigation":           "Mitigation",
    "attribution":          "Attribution",
    "other":                "Other",
}

def load_all_data() -> pd.DataFrame:
    dfs = []


    attck_path = DATA_DIR / "attck" / "lifetimes.csv"
    if attck_path.exists():
        df_a = pd.read_csv(attck_path)
        df_a = df_a[["relation_category", "lifetime_months", "observed"]].copy()
        df_a.rename(columns={"relation_category": "category"}, inplace=True)
        dfs.append(df_a)
        print(f"ATT&CK: {len(df_a)} kayit yuklendi")
    else:
        print("[WARN] ATT&CK lifetimes.csv bulunamadi")


    abusech_path = DATA_DIR / "abusech" / "ioc_lifetimes.csv"
    if abusech_path.exists():
        df_b = pd.read_csv(abusech_path)
        df_b = df_b[["ioc_type", "lifetime_months", "observed"]].copy()
        df_b.rename(columns={"ioc_type": "category"}, inplace=True)
        dfs.append(df_b)
        print(f"abuse.ch: {len(df_b)} kayit yuklendi")
    else:
        print("[WARN] abuse.ch ioc_lifetimes.csv bulunamadi")

    if not dfs:
        raise FileNotFoundError(
            "Veri bulunamadi. Once 00, 01, 02 scriptlerini calistirin.")

    df = pd.concat(dfs, ignore_index=True)

    df = df[df["lifetime_months"] >= 0.01].copy()
    print(f"\nToplam: {len(df)} gozlem")
    return df

def fit_exponential(durations: np.ndarray, events: np.ndarray) -> dict:
    ef = ExponentialFitter()
    ef.fit(durations, event_observed=events)
    lam = 1.0 / ef.lambda_
    half_life = np.log(2) / lam
    return {"lambda": round(float(lam), 6),
            "half_life_months": round(float(half_life), 3),
            "mean_lifetime_months": round(float(ef.lambda_), 3)}

def fit_weibull(durations: np.ndarray, events: np.ndarray) -> dict:
    wf = WeibullFitter()
    wf.fit(durations, event_observed=events)
    rho   = wf.rho_
    lam_w = wf.lambda_

    median = lam_w * (np.log(2) ** (1.0 / rho))
    return {
        "shape_beta":       round(float(rho), 4),
        "scale_eta_months": round(float(lam_w), 4),
        "median_months":    round(float(median), 3),
        "aic":              round(float(wf.AIC_), 2),
    }

def run_survival_analysis(df: pd.DataFrame) -> dict:
    categories = df["category"].value_counts()
    categories = categories[categories >= 10].index.tolist()
    print(f"\nAnaliz edilecek kategoriler: {categories}")

    results = {}
    km_records = []

    for cat in categories:
        sub = df[df["category"] == cat]
        T = sub["lifetime_months"].values
        E = sub["observed"].values

        print(f"\n  [{cat}] n={len(T)}, events={E.sum()}, censored={(E==0).sum()}")


        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=E, label=cat)


        km_df = kmf.survival_function_.reset_index()
        km_df.columns = ["timeline", "km_estimate"]
        km_df["category"] = cat
        km_df["ci_lower"] = kmf.confidence_interval_[f"{cat}_lower_0.95"].values
        km_df["ci_upper"] = kmf.confidence_interval_[f"{cat}_upper_0.95"].values
        km_records.append(km_df)


        median_km = kmf.median_survival_time_


        try:
            exp_params = fit_exponential(T, E)
        except Exception as e:
            print(f"    [WARN] Exp fit: {e}")
            exp_params = {}


        try:
            wb_params = fit_weibull(T, E)
        except Exception as e:
            print(f"    [WARN] Weibull fit: {e}")
            wb_params = {}

        results[cat] = {
            "n_total":           int(len(T)),
            "n_events":          int(E.sum()),
            "n_censored":        int((E == 0).sum()),
            "median_km_months":  round(float(median_km) if not np.isnan(median_km) else -1, 3),
            "exponential":       exp_params,
            "weibull":           wb_params,
        }

        print(f"    Median (KM): {median_km:.2f} ay")
        if exp_params:
            print(f"    Exp lambda={exp_params['lambda']:.6f}  "
                  f"tau_1/2={exp_params['half_life_months']:.2f} ay")


    km_combined = pd.concat(km_records, ignore_index=True)
    km_combined.to_csv(RES_DIR / "km_data.csv", index=False)
    print(f"\nKM verileri kaydedildi: {RES_DIR/'km_data.csv'}")

    return results

def plot_km_curves(km_df: pd.DataFrame):
    cats = [c for c in km_df["category"].unique()
            if c in LABEL_MAP]
    cats = sorted(cats, key=lambda c: LABEL_MAP.get(c, c))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")

    for cat in cats:
        sub = km_df[km_df["category"] == cat].copy()
        color = PALETTE.get(cat, "#888888")
        label = LABEL_MAP.get(cat, cat)

        sub = sub[sub["timeline"] >= 0.0]
        t = sub["timeline"].clip(lower=0.01)
        ax.step(t, sub["km_estimate"],
                where="post", color=color, lw=2.0, label=label)
        ax.fill_between(t, sub["ci_lower"], sub["ci_upper"],
                        step="post", color=color, alpha=0.12)

    ax.set_xscale("log")
    tick_pos = [0.033, 0.23, 1, 6, 12, 60, 120]
    tick_lab = ["1 d", "1 wk", "1 mo", "6 mo", "1 yr", "5 yr", "10 yr"]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlim(0.01, 130)

    ax.set_xlabel("Time since first observation (log scale)",
                  fontsize=13, labelpad=6)
    ax.set_ylabel("Survival Probability S(t)", fontsize=13, labelpad=6)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, ls="--", alpha=0.35)
    ax.axhline(0.5, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.text(0.0115, 0.52, "median", fontsize=9, color="black", alpha=0.7)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)

    out = FIG_DIR / "fig_survival.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_survival.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Kaydedildi: {out} (+.pdf)")

def plot_decay_curves(params: dict):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")

    t = np.logspace(np.log10(0.01), np.log10(120), 600)

    for cat, res in params.items():
        if cat not in LABEL_MAP:
            continue
        color = PALETTE.get(cat, "#888888")
        label = LABEL_MAP.get(cat, cat)

        exp_p = res.get("exponential", {})
        if exp_p and exp_p.get("lambda", 0) > 0:
            kappa = np.exp(-exp_p["lambda"] * t)
            ax.plot(t, kappa, color=color, lw=2.0, label=label)


            hl = exp_p["half_life_months"]
            if 0.01 < hl < 120:
                ax.plot([hl], [0.5], marker="o", ms=5, color=color)

    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5,
               label=r"$\kappa = 0.5$ (half-life)")
    ax.set_xscale("log")
    tick_pos = [0.033, 0.23, 1, 6, 12, 60, 120]
    tick_lab = ["1 d", "1 wk", "1 mo", "6 mo", "1 yr", "5 yr", "10 yr"]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlim(0.01, 120)

    ax.set_xlabel("Time since observation (log scale)",
                  fontsize=13, labelpad=6)
    ax.set_ylabel(r"Confidence $\kappa(t)$", fontsize=13, labelpad=6)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)

    out = FIG_DIR / "fig_decay.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_decay.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Kaydedildi: {out} (+.pdf)")

def print_latex_table(params: dict):
    print("\n\n=== LaTeX Tablosu (tab:decay) icin ===")
    rows = [
        ("network_ioc_url",     "Network IOC (IP/domain)", "URLhaus/OTX"),
        ("file_hash",           "File Hash IOC",           "MalwareBazaar"),
        ("ttp_use",             "TTP / technique use",     "ATT\\&CK"),
        ("technique_hierarchy", "Technique (deprecated)",  "ATT\\&CK"),
        ("attribution",         "Attribution",             "ATT\\&CK"),
    ]
    print(r"\begin{tabular}{llrr}")
    print(r"\toprule")
    print(r"\textbf{Relation type} & \textbf{Source} & $\hat{\lambda}$ & $\tau_{1/2}$ \\")
    print(r"\midrule")
    for cat, label, src in rows:
        if cat in params and params[cat].get("exponential"):
            ep = params[cat]["exponential"]
            lam = ep.get("lambda", 0)
            hl  = ep.get("half_life_months", 0)
            unit = "d" if hl < 3 else "mo"
            hl_disp = f"{hl*30:.0f}~d" if hl < 3 else f"{hl:.1f}~mo"
            print(f"  {label} & {src} & {lam:.6f} & {hl_disp} \\\\")
        else:
            print(f"  {label} & {src} & --- & --- \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

def main():
    print("=== Survival Analysis Pipeline ===\n")
    df = load_all_data()

    params = run_survival_analysis(df)


    out_json = RES_DIR / "survival_params.json"
    out_json.write_text(json.dumps(params, indent=2))
    print(f"\nParametreler kaydedildi: {out_json}")


    km_df = pd.read_csv(RES_DIR / "km_data.csv")
    plot_km_curves(km_df)
    plot_decay_curves(params)


    print_latex_table(params)

    print("\n=== Tamamlandi ===")

if __name__ == "__main__":
    main()
