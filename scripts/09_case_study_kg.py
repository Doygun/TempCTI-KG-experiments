import io
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RES_DIR  = BASE_DIR / "results"
FIG_DIR  = RES_DIR / "figures"

MONTH_DAYS = 30.44
THETA = 0.5
MALWARE = "Amadey"


def load_urlhaus_full():
    z = zipfile.ZipFile(DATA_DIR / "abusech" / "urlhaus_full.zip")
    raw = z.read("csv.txt").decode("utf-8", errors="replace")
    lines = [l for l in raw.splitlines() if l and not l.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)),
                     names=["id", "dateadded", "url", "url_status",
                            "last_online", "threat", "tags",
                            "urlhaus_link", "reporter"],
                     on_bad_lines="skip")
    df["dateadded"] = pd.to_datetime(df["dateadded"], errors="coerce")
    df["last_online"] = pd.to_datetime(df["last_online"], errors="coerce")
    df["tags"] = df["tags"].fillna("")
    return df[df["dateadded"] >= "2026-04-07 20:59:08"]


def kg_stats(url_win):
    a = pd.read_csv(DATA_DIR / "attck" / "lifetimes.csv")
    stats = {}


    subj = a[["subject", "subject_type"]].rename(
        columns={"subject": "name", "subject_type": "type"})
    obj = a[["object", "object_type"]].rename(
        columns={"object": "name", "object_type": "type"})
    nodes = pd.concat([subj, obj]).drop_duplicates("name")
    stats["attck_nodes_total"] = int(len(nodes))
    stats["attck_node_types"] = {
        k: int(v) for k, v in
        nodes["type"].value_counts().head(8).items()}
    stats["attck_edges"] = int(len(a))
    stats["attck_relation_types"] = int(a["relation_type"].nunique())


    fam = url_win["tags"].str.split(",").explode().str.strip()
    fam = fam[fam.str.len() > 2]
    generic = {"elf", "exe", "zip", "apk", "ascii", "opendir", "32-bit",
               "64-bit", "mips", "arm", "x86", "sh", "bat", "ps1", "vbs",
               "dll", "jar", "doc", "xls", "js", "encrypted", "ua-wget"}
    fam = fam[~fam.str.lower().isin(generic)]
    stats["urlhaus_url_nodes"] = int(url_win["url"].nunique())
    stats["urlhaus_family_nodes"] = int(fam.str.lower().nunique())
    stats["urlhaus_url_family_edges"] = int(len(fam))
    return stats


def kappa(t_mo, model):
    if t_mo < 0:
        return 1.0
    eta, beta = model["scale_eta_months"], model["shape_beta"]
    return float(np.exp(-((max(t_mo, 0.0) / eta) ** beta)))


def build_case(url_win, params):
    a = pd.read_csv(DATA_DIR / "attck" / "lifetimes.csv",
                    parse_dates=["t_start_date"])
    tech = a[(a["subject"].astype(str).str.lower() == MALWARE.lower())
             & (a["relation_category"] == "ttp_use")
             & (a["observed"] == 0)].copy()
    tech = tech.sort_values("t_start_date").head(5)

    urls = url_win[url_win["tags"].str.contains(MALWARE, case=False)].copy()
    urls["death"] = urls["last_online"]
    urls.loc[urls["url_status"] == "online", "death"] = pd.NaT

    tq1, tq2 = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01")
    cand = urls[urls["dateadded"] <= tq1].copy()
    alive_tq1 = cand[(cand["death"].isna()) | (cand["death"] > tq1)]


    die_between = alive_tq1[(alive_tq1["death"].notna())
                            & (alive_tq1["death"] <= tq2)]
    still_alive = alive_tq1[(alive_tq1["death"].isna())
                            | (alive_tq1["death"] > tq2)]
    dead_before = cand[cand["death"].notna() & (cand["death"] <= tq1)]
    picks = pd.concat([
        die_between.sort_values("death").head(3),
        still_alive.sort_values("dateadded").tail(2),
        dead_before.tail(1),
    ]).drop_duplicates("url").head(6)
    return tech, picks


def short_url(u, i):
    from urllib.parse import urlparse
    try:
        host = urlparse(u).netloc.split(":")[0]
        parts = host.split(".")
        if all(p.isdigit() for p in parts):
            host = f"{parts[0]}.{parts[1]}.x.x"
        elif len(parts) > 2:
            host = ".".join(parts[-2:])
    except Exception:
        host = f"url-{i}"
    return f"hxxp://{host}/…"


def draw_case(tech, picks, params):
    wb_ttp = params["ttp_use"]["weibull"]
    wb_ioc = params["network_ioc_url"]["weibull"]

    tq1 = pd.Timestamp("2026-06-01")
    tq2 = pd.Timestamp("2026-07-01")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    fig.patch.set_facecolor("white")

    for ax, tq in zip(axes, [tq1, tq2]):
        G = nx.Graph()
        G.add_node(MALWARE)
        pos = {MALWARE: (0, 0)}


        edges = []
        n_t = len(tech)
        for i, (_, row) in enumerate(tech.iterrows()):
            name = row["object"]
            label = name if len(name) <= 32 else name[:30] + "…"
            y = (i - (n_t - 1) / 2) * 1.15
            pos[label] = (-3.1, y)
            G.add_node(label)
            age_mo = (tq - row["t_start_date"]).days / MONTH_DAYS
            k = kappa(age_mo, wb_ttp)
            edges.append((MALWARE, label, k, k >= THETA, "ttp"))


        n_u = len(picks)
        for i, (_, row) in enumerate(picks.reset_index().iterrows()):
            label = short_url(row["url"], i)
            if label in pos:
                label += f" ({i})"
            y = (i - (n_u - 1) / 2) * 1.15
            pos[label] = (3.1, y)
            G.add_node(label)
            if pd.notna(row["death"]) and row["death"] <= tq:
                k, active = 0.0, False
            else:
                age_mo = (tq - row["dateadded"]).days / MONTH_DAYS
                k = kappa(age_mo, wb_ioc)
                active = k >= THETA
            edges.append((MALWARE, label, k, active, "ioc"))


        for (u, v, k, active, kind) in edges:
            color = "#2a9d8f" if kind == "ttp" else "#e63946"
            if active:
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color=color, lw=1.0 + 3.0 * k, alpha=0.35 + 0.6 * k,
                        zorder=1)
            else:
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color="#adb5bd", lw=1.2, ls=(0, (4, 3)), alpha=0.8,
                        zorder=1)
            xm = 0.45 * pos[u][0] + 0.55 * pos[v][0]
            ym = 0.45 * pos[u][1] + 0.55 * pos[v][1] + 0.14
            txt = f"$\\kappa$={k:.2f}" if active else "retracted"
            ax.text(xm, ym, txt, fontsize=9,
                    color="#1d3557" if active else "#868e96",
                    ha="center", zorder=3)


        for n, (x, y) in pos.items():
            if n == MALWARE:
                ax.scatter([x], [y], s=2600, marker="s", color="#1d3557",
                           edgecolor="white", linewidth=1.5, zorder=2)
                ax.text(x, y, MALWARE, color="white", fontsize=11,
                        fontweight="bold", ha="center", va="center", zorder=3)
                continue
            up = y > 0.01
            dy = 0.38 if up else -0.38
            va = "bottom" if up else "top"
            xin = 0.15 if x < 0 else -0.15
            if x < 0:
                ax.scatter([x], [y], s=420, marker="o", color="#d8f3dc",
                           edgecolor="#2a9d8f", linewidth=1.8, zorder=2)
                ax.text(x + xin, y + dy, n, fontsize=8.8, ha="center",
                        va=va, color="#1b4332", zorder=3)
            else:
                ax.scatter([x], [y], s=420, marker="o", color="#ffe3e3",
                           edgecolor="#e63946", linewidth=1.8, zorder=2)
                ax.text(x + xin, y + dy, n, fontsize=8.8, ha="center",
                        va=va, color="#7f1d1d", family="monospace",
                        zorder=3)

        n_act = sum(1 for e in edges if e[3])
        ax.set_title(f"$t_q$ = {tq.date()}   "
                     f"({n_act}/{len(edges)} facts active)",
                     fontsize=13, fontweight="bold")
        ax.set_xlim(-4.7, 4.7)
        ax.set_ylim(-3.9, 3.5)
        ax.axis("off")

    legend = [
        Line2D([0], [0], color="#2a9d8f", lw=3,
               label="uses technique (ATT&CK)"),
        Line2D([0], [0], color="#e63946", lw=3,
               label="distributes via URL (URLhaus)"),
        Line2D([0], [0], color="#adb5bd", lw=2, ls=(0, (4, 3)),
               label="retracted fact"),
        mpatches.Patch(facecolor="#d8f3dc", edgecolor="#2a9d8f",
                       label="ATT&CK technique"),
        mpatches.Patch(facecolor="#ffe3e3", edgecolor="#e63946",
                       label="malware-distribution URL"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = FIG_DIR / "fig_case_kg.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    out_pdf = FIG_DIR / "fig_case_kg.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Kaydedildi: {out}")
    print(f"Kaydedildi: {out_pdf}")


def main():
    params = json.loads((RES_DIR / "survival_params.json").read_text())
    url_win = load_urlhaus_full()

    stats = kg_stats(url_win)
    (RES_DIR / "kg_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    tech, picks = build_case(url_win, params)
    print(f"\nVaka: {MALWARE} | teknik={len(tech)} URL={len(picks)}")
    print(picks[["url", "dateadded", "url_status", "death"]].to_string())
    draw_case(tech, picks, params)


if __name__ == "__main__":
    main()
