"""Figures for the report, from results/games.csv (written by analyze.py).

Run with the system Python (needs numpy and matplotlib):
    python make_figures.py
Writes results/fig1_target_challenges_ally.png, fig2_G.png, fig3_allies_discuss_U.png
and results/figures_data.csv (the numbers behind every figure, for a table view).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CONDS = ["control", "inform", "warn", "expose", "provoke"]
MODELS = [("deepseek-v4.1-flash", "DeepSeek V4.1 Flash", "#2a78d6"),   # categorical slot 1
          ("llama-3.1-8b", "Llama 3.1 8B", "#eb6834")]               # categorical slot 2
INK, INK2, GRID, SURFACE = "#1f1f1e", "#5f5e5a", "#e4e3df", "#fcfcfb"
RNG = np.random.default_rng(2026)
N_BOOT = 5000

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, mid - half, mid + half


def boot_g(rows):
    """Pooled G and its cluster-bootstrap 95% CI (resampling games)."""
    a = np.array([[float(r[k]) for k in ("al_pass", "al_n", "ot_pass", "ot_n")] for r in rows])
    g = a[:, 0].sum() / a[:, 1].sum() - a[:, 2].sum() / a[:, 3].sum()
    idx = RNG.integers(0, len(a), size=(N_BOOT, len(a)))
    s = a[idx].sum(axis=1)
    ok = (s[:, 1] > 0) & (s[:, 3] > 0)
    gb = s[ok, 0] / s[ok, 1] - s[ok, 2] / s[ok, 3]
    return g, *np.percentile(gb, [2.5, 97.5])


def axes_style(ax, ylabel):
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels(CONDS)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def dot_chart(table, ylabel, title, fname, pct=True, zero_line=False):
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    off = {MODELS[0][0]: -0.12, MODELS[1][0]: 0.12}
    for key, label, color in MODELS:
        xs = np.arange(len(CONDS)) + off[key]
        vals = [table[(key, c)] for c in CONDS]
        y = np.array([v[0] for v in vals]); lo = np.array([v[1] for v in vals]); hi = np.array([v[2] for v in vals])
        ax.errorbar(xs, y, yerr=[y - lo, hi - y], fmt="o", ms=7, color=color, ecolor=color, elinewidth=2,
                    capsize=0, markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        ax.text(xs[-1] + 0.1, y[-1], label.split()[0], color=INK2, va="center", fontsize=9)  # direct label
    if zero_line:
        ax.axhline(0, color=INK2, linewidth=1)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes_style(ax, ylabel)
    ax.set_xlim(-0.5, len(CONDS) - 0.2)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)  # off the data
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close(fig)


def main():
    rows = list(csv.DictReader(open(os.path.join(OUT, "games.csv"), encoding="utf-8")))
    by = {(m, c): [r for r in rows if r["model"] == m and r["condition"] == c] for m, _, _ in MODELS for c in CONDS}
    out_rows = []

    t1 = {}
    for k, rs in by.items():
        kk, nn = sum(int(r["tal_k"]) for r in rs), sum(int(r["tal_n"]) for r in rs)
        t1[k] = wilson(kk, nn)
        out_rows.append(dict(figure="fig1", model=k[0], condition=k[1], k=kk, n=nn, value=t1[k][0], lo=t1[k][1], hi=t1[k][2]))
    dot_chart(t1, "target challenges ally's lie", "Target challenges the ally's lies (round 3 on), Wilson 95% CI",
              "fig1_target_challenges_ally.png")

    t2 = {}
    for k, rs in by.items():
        t2[k] = boot_g(rs)
        out_rows.append(dict(figure="fig2", model=k[0], condition=k[1], k="", n=len(rs), value=t2[k][0], lo=t2[k][1], hi=t2[k][2]))
    dot_chart(t2, "G (ally lies passed - other lies passed)", "Favouritism G (round 3 on), cluster-bootstrap 95% CI",
              "fig2_G.png", pct=False, zero_line=True)

    # fig3: LLM-labelled chats about U (llm_label.py), per game; control = the labeller's false-positive baseline
    lab = list(csv.DictReader(open(os.path.join(OUT, "llm_labels_A.csv"), encoding="utf-8-sig")))
    t3 = {}
    for m, _, _ in MODELS:
        for c in CONDS:
            per_game = np.array([sum(1 for r in lab if r["model"] == m and r["condition"] == c and r["seed"] == str(s)
                                     and r["llm_label"] == "YES") for s in range(1, 31)], float)
            boots = per_game[RNG.integers(0, len(per_game), size=(N_BOOT, len(per_game)))].mean(axis=1)
            t3[(m, c)] = (per_game.mean(), *np.percentile(boots, [2.5, 97.5]))
            out_rows.append(dict(figure="fig3", model=m, condition=c, k=int(per_game.sum()), n=len(per_game),
                                 value=t3[(m, c)][0], lo=t3[(m, c)][1], hi=t3[(m, c)][2]))
    dot_chart(t3, "chats per game", "Allies' private chats about the undercover agent (LLM-labelled), per game",
              "fig3_allies_discuss_U.png", pct=False)

    with open(os.path.join(OUT, "figures_data.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print("written:", ", ".join(f for f in os.listdir(OUT) if f.startswith("fig")))


if __name__ == "__main__":
    main()
