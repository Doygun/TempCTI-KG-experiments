import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import ExponentialFitter, WeibullFitter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "abusech"
RES_DIR  = BASE_DIR / "results"
FIG_DIR  = RES_DIR / "figures"

TRAIN_CUTOFF = pd.Timestamp("2026-05-31")
REPLAY_START = pd.Timestamp("2026-06-01")
REPLAY_END   = pd.Timestamp("2026-07-05")
MONTH_DAYS   = 30.44
IOC_TTL_D    = 30.0
TAUS         = [0.5, 0.25]


def load_urls():
    df = pd.read_csv(DATA_DIR / "urlhaus_full_parsed.csv",
                     parse_dates=["dateadded", "last_online"])
    dump_time = df["dateadded"].max()

    is_off = df["url_status"].str.lower().eq("offline")
    death = pd.Series(pd.NaT, index=df.index)
    death[is_off] = df.loc[is_off, "last_online"]
    death[is_off & df["last_online"].isna()] = df.loc[
        is_off & df["last_online"].isna(), "dateadded"]
    df["death"] = death
    print(f"URL: {len(df)} | olen: {is_off.sum()} | dump: {dump_time}")
    return df, dump_time


def fit_train(df):
    tr = df[df["dateadded"] <= TRAIN_CUTOFF].copy()
    died_before = tr["death"].notna() & (tr["death"] <= TRAIN_CUTOFF)
    end = tr["death"].where(died_before, TRAIN_CUTOFF)
    T = ((end - tr["dateadded"]).dt.total_seconds() / 86400.0
         / MONTH_DAYS).clip(lower=0.0164).values
    E = died_before.astype(int).values
    ef = ExponentialFitter().fit(T, event_observed=E)
    lam = float(1.0 / ef.lambda_)
    wf = WeibullFitter().fit(T, event_observed=E)
    eta, beta = float(wf.lambda_), float(wf.rho_)
    print(f"Train: n={len(tr)} events={E.sum()} | "
          f"lambda={lam:.4f}/ay | Weibull eta={eta:.4f} beta={beta:.4f}")
    return lam, eta, beta


def confusion(yp, yt):
    tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    return {"prec": prec, "rec": rec,
            "f1": 2 * prec * rec / (prec + rec + 1e-9),
            "srr": fp / (fp + tn + 1e-9), "tp": tp, "alerts": tp + fp}


def replay(df, dump_time, lam, eta, beta):
    days = pd.date_range(REPLAY_START, REPLAY_END, freq="D")
    daily = []
    for t_q in days:
        pool = df[df["dateadded"] <= t_q]

        y_true = (~(pool["death"].notna() & (pool["death"] <= t_q))).astype(int).values
        if len(np.unique(y_true)) < 2 or len(pool) < 50:
            continue
        t_el_d = ((t_q - pool["dateadded"]).dt.total_seconds() / 86400.0).values
        t_el_mo = t_el_d / MONTH_DAYS

        k_exp = np.exp(-lam * t_el_mo)
        k_wb  = np.exp(-np.power(np.clip(t_el_mo, 0, None) / eta, beta))

        row = {"t_q": str(t_q.date()), "n_pool": len(pool),
               "n_active": int(y_true.sum())}
        methods = {"static": np.ones(len(pool)),
                   "ttl": (t_el_d <= IOC_TTL_D).astype(float)}
        for tau in TAUS:
            methods[f"exp_tau{tau}"] = (k_exp >= tau).astype(float)
            methods[f"wb_tau{tau}"]  = (k_wb >= tau).astype(float)
        for name, yp in methods.items():
            c = confusion(yp.astype(int), y_true)
            row.update({f"{k}_{name}": v for k, v in c.items()})
        daily.append(row)

    res = pd.DataFrame(daily)
    res.to_csv(RES_DIR / "replay_daily.csv", index=False)
    return res


def summarize(res):
    out = {}
    for m in ["static", "ttl", "exp_tau0.5", "wb_tau0.5",
              "exp_tau0.25", "wb_tau0.25"]:
        out[m] = {
            "mean_srr":  round(res[f"srr_{m}"].mean(), 4),
            "mean_prec": round(res[f"prec_{m}"].mean(), 4),
            "mean_rec":  round(res[f"rec_{m}"].mean(), 4),
            "mean_f1":   round(res[f"f1_{m}"].mean(), 4),
            "mean_alerts": round(res[f"alerts_{m}"].mean(), 1),
        }
    return out


def plot(res):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.patch.set_facecolor("white")
    series = {"static": ("#e63946", "Static KG"),
              "ttl": ("#f4a261", "Fixed-TTL (30d)"),
              "exp_tau0.5": ("#457b9d", "TempCTI-KG (Exp., $\\tau$=0.5)"),
              "wb_tau0.5": ("#2a9d8f", "TempCTI-KG (Weibull, $\\tau$=0.5)"),
              "wb_tau0.25": ("#6a4c93", "TempCTI-KG (Weibull, $\\tau$=0.25)")}
    x = pd.to_datetime(res["t_q"])
    for i, (metric, label) in enumerate(
            [("srr", "Stale Retrieval Rate"),
             ("prec", "Precision"), ("f1", "F1-score")]):
        for m, (c, lab) in series.items():
            axes[i].plot(x, res[f"{metric}_{m}"], color=c, lw=2, label=lab)
        axes[i].set_ylabel(label, fontsize=12)
        axes[i].grid(True, ls="--", alpha=0.35)
        axes[i].set_ylim(-0.02, 1.05)
    axes[0].legend(fontsize=9, ncol=2)
    axes[2].set_xlabel("Query date", fontsize=12)
    fig.autofmt_xdate()
    plt.tight_layout()
    out = FIG_DIR / "fig_replay.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_replay.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Kaydedildi: {out} (+.pdf)")


def main():
    print("=== Task 3b: Real-Data Replay (URLhaus) ===\n")
    df, dump_time = load_urls()
    lam, eta, beta = fit_train(df)
    res = replay(df, dump_time, lam, eta, beta)
    summary = summarize(res)
    for m, s in summary.items():
        print(f"  {m:12s} SRR={s['mean_srr']:.3f} P={s['mean_prec']:.3f} "
              f"R={s['mean_rec']:.3f} F1={s['mean_f1']:.3f}")
    out = {"train_cutoff": str(TRAIN_CUTOFF.date()),
           "replay_window": [str(REPLAY_START.date()), str(REPLAY_END.date())],
           "train_params": {"lambda": lam, "weibull_eta": eta,
                            "weibull_beta": beta},
           "methods": summary}
    (RES_DIR / "replay_results.json").write_text(json.dumps(out, indent=2))
    print(f"Kaydedildi: {RES_DIR / 'replay_results.json'}")
    plot(res)


if __name__ == "__main__":
    main()
