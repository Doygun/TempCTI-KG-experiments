import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RES_DIR  = BASE_DIR / "results"
FIG_DIR  = RES_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

TAU        = 0.50
TAU_GRID   = [0.25, 0.50, 0.75]
IOC_TTL_D  = 30
TTP_TTL_D  = 365
SIM_DAYS   = 90
MONTH_DAYS = 30.44


def load_params():
    return json.loads((RES_DIR / "survival_params.json").read_text())


def sample_weibull_lifetime_days(rng, eta_mo, beta, size):
    u = rng.uniform(size=size)
    lifetime_mo = eta_mo * np.power(-np.log(u), 1.0 / beta)
    return lifetime_mo * MONTH_DAYS


def build_mixed_kg(params, n_ioc_per_day=50, n_ttp=500, sim_days=SIM_DAYS):
    rng = np.random.default_rng(42)
    records = []

    p_ttp = params["ttp_use"]["weibull"]
    p_ioc = params["network_ioc_url"]["weibull"]


    starts = rng.integers(0, sim_days // 2, size=n_ttp)
    lifetimes = sample_weibull_lifetime_days(
        rng, p_ttp["scale_eta_months"], p_ttp["shape_beta"], n_ttp)
    for s, lt in zip(starts, lifetimes):
        records.append({"fact_type": "ttp_use",
                        "start_day": int(s),
                        "end_day": int(s + lt)})


    for day in range(sim_days):
        lifetimes = sample_weibull_lifetime_days(
            rng, p_ioc["scale_eta_months"], p_ioc["shape_beta"],
            n_ioc_per_day)
        for lt in lifetimes:
            records.append({"fact_type": "network_ioc_url",
                            "start_day": day,
                            "end_day": day + int(lt)})

    return pd.DataFrame(records)


def kappa_exp(t_mo, lam):
    return np.exp(-lam * t_mo)


def kappa_weibull(t_mo, eta, beta):
    return np.exp(-np.power(np.clip(t_mo, 0, None) / eta, beta))


def confusion(yp, yt):
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    fpr  = fp / (fp + tn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "prec": prec, "rec": rec, "f1": f1, "fpr": fpr}


def evaluate(df, params, tau=TAU, sim_days=SIM_DAYS):
    lam_ioc  = params["network_ioc_url"]["exponential"]["lambda"]
    lam_ttp  = params["ttp_use"]["exponential"]["lambda"]
    wb_ioc   = params["network_ioc_url"]["weibull"]
    wb_ttp   = params["ttp_use"]["weibull"]

    is_ioc_all = (df["fact_type"] == "network_ioc_url").values

    metrics = []
    for t_q in range(1, sim_days + 1):
        mask = (df["start_day"] <= t_q).values
        if mask.sum() < 10:
            continue
        pool = df[mask]
        is_ioc = is_ioc_all[mask]

        y_true = (pool["end_day"].values >= t_q).astype(int)
        if len(np.unique(y_true)) < 2:
            continue

        t_el_d  = np.clip(t_q - pool["start_day"].values, 0, None)
        t_el_mo = t_el_d / MONTH_DAYS

        y_static = np.ones(len(pool), dtype=int)

        ttl_days = np.where(is_ioc, IOC_TTL_D, TTP_TTL_D)
        y_ttl = (t_el_d <= ttl_days).astype(int)

        lam = np.where(is_ioc, lam_ioc, lam_ttp)
        y_exp = (kappa_exp(t_el_mo, lam) >= tau).astype(int)

        eta  = np.where(is_ioc, wb_ioc["scale_eta_months"],
                        wb_ttp["scale_eta_months"])
        beta = np.where(is_ioc, wb_ioc["shape_beta"], wb_ttp["shape_beta"])
        y_wb = (kappa_weibull(t_el_mo, eta, beta) >= tau).astype(int)

        row = {"t_q_day": t_q, "n_facts": len(pool),
               "n_active_gt": int(y_true.sum())}
        for name, yp in [("static", y_static), ("ttl", y_ttl),
                         ("exp", y_exp), ("weibull", y_wb)]:
            c = confusion(yp, y_true)
            row.update({f"{k}_{name}": v for k, v in c.items()})
            row[f"alerts_{name}"] = int(yp.sum())
        metrics.append(row)

    return pd.DataFrame(metrics)


def summarize(res):
    summary = {}
    for m in ["static", "ttl", "exp", "weibull"]:
        tp_ret = res[f"tp_{m}"].mean() / (res["tp_static"].mean() + 1e-9) * 100
        summary[m] = {
            "mean_srr":    round(res[f"fpr_{m}"].mean(), 4),
            "mean_prec":   round(res[f"prec_{m}"].mean(), 4),
            "mean_rec":    round(res[f"rec_{m}"].mean(), 4),
            "mean_f1":     round(res[f"f1_{m}"].mean(), 4),
            "mean_alerts": round(res[f"alerts_{m}"].mean(), 1),
            "tp_retention_pct": round(tp_ret, 1),
        }
    return summary


def plot_timeseries(res):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.patch.set_facecolor("white")
    colors = {"static": "#e63946", "ttl": "#f4a261",
              "exp": "#457b9d", "weibull": "#2a9d8f"}
    labels = {"static": "Static KG", "ttl": "Fixed-TTL",
              "exp": "TempCTI-KG (Exp.)", "weibull": "TempCTI-KG (Weibull)"}

    for m in colors:
        axes[0].plot(res["t_q_day"], res[f"fpr_{m}"],
                     color=colors[m], lw=2, label=labels[m])
    axes[0].set_ylabel("Stale Retrieval Rate", fontsize=12)
    axes[0].legend(fontsize=10); axes[0].grid(True, ls="--", alpha=0.35)
    axes[0].set_ylim(-0.02, 1.05)

    for m in colors:
        axes[1].plot(res["t_q_day"], res[f"prec_{m}"],
                     color=colors[m], lw=2, label=labels[m])
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].legend(fontsize=10); axes[1].grid(True, ls="--", alpha=0.35)
    axes[1].set_ylim(-0.02, 1.05)

    for m in colors:
        axes[2].plot(res["t_q_day"], res[f"alerts_{m}"],
                     color=colors[m], lw=2, label=labels[m])
    axes[2].set_ylabel("Active Facts in KG", fontsize=12)
    axes[2].set_xlabel("Day of Operation", fontsize=12)
    axes[2].legend(fontsize=10); axes[2].grid(True, ls="--", alpha=0.35)

    plt.tight_layout()
    out = FIG_DIR / "fig_fp_timeseries.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_fp_timeseries.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Kaydedildi: {out} (+.pdf)")


def plot_summary_bars(summary):
    methods = ["Static KG", "Fixed-TTL", "TempCTI-KG\n(Exp.)",
               "TempCTI-KG\n(Weibull)"]
    keys    = ["static", "ttl", "exp", "weibull"]
    colors  = ["#e63946", "#f4a261", "#457b9d", "#2a9d8f"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor("white")

    metrics = [("mean_srr", "Mean Stale Retrieval Rate (↓)"),
               ("mean_f1", "Mean F1-score (↑)"),
               ("mean_alerts", "Mean Active Facts (↓)")]
    for ax, (key, title) in zip(axes, metrics):
        vals = [summary[k][key] for k in keys]
        bars = ax.bar(methods, vals, color=colors,
                      edgecolor="white", linewidth=1.2, alpha=0.9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", ls="--", alpha=0.35)
        ax.set_ylim(0, max(vals) * 1.25 + 0.01)
        ax.tick_params(axis="x", labelsize=8)
        for bar, v in zip(bars, vals):
            fmt = f"{v:.3f}" if v < 10 else f"{v:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.04,
                    fmt, ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "fig_fp.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Kaydedildi: {out}")


def main():
    print("=== Task 3: Downstream Stale-Fact Reduction ===\n")
    params = load_params()
    df = build_mixed_kg(params)
    print(f"Toplam fact: {len(df)} "
          f"(TTP: {(df.fact_type == 'ttp_use').sum()}, "
          f"IOC: {(df.fact_type == 'network_ioc_url').sum()})")

    res = evaluate(df, params, tau=TAU)
    res.to_csv(RES_DIR / "downstream_daily.csv", index=False)
    summary = summarize(res)

    for m, s in summary.items():
        print(f"  {m:8s} SRR={s['mean_srr']:.3f} P={s['mean_prec']:.3f} "
              f"R={s['mean_rec']:.3f} F1={s['mean_f1']:.3f} "
              f"alerts={s['mean_alerts']:.0f} TPret={s['tp_retention_pct']}%")


    print("\nSensitivity (Weibull):")
    sensitivity = {}
    for tau in TAU_GRID:
        r = evaluate(df, params, tau=tau)
        s = summarize(r)["weibull"]
        sensitivity[str(tau)] = s
        print(f"  tau={tau}: SRR={s['mean_srr']:.3f} R={s['mean_rec']:.3f} "
              f"F1={s['mean_f1']:.3f}")

    out = {
        "primary_tau": TAU,
        "methods": summary,
        "sensitivity_tau_weibull": sensitivity,
        "srr_reduction_vs_static_pct": round(
            (1 - summary["weibull"]["mean_srr"]) * 100, 1),
        "active_fact_reduction_pct": round(
            (1 - summary["weibull"]["mean_alerts"]
             / summary["static"]["mean_alerts"]) * 100, 1),
    }
    (RES_DIR / "downstream_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nKaydedildi: {RES_DIR / 'downstream_results.json'}")

    plot_timeseries(res)
    plot_summary_bars(summary)


if __name__ == "__main__":
    main()
