"""
Figures for the report. Reads runs/, runs/shock/, runs_table/ and writes figures/*.png and *.pdf.

  python make_figures.py
"""
import glob, json, math, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

PN, PM = 1.4725, 1.925
COND = ["c1", "c2", "c3"]
LABEL = {"c1": "No label", "c2": "True label (long-run profit)", "c3": "False label (wants to beat you)"}
SHORT = {"c1": "No label", "c2": "True label", "c3": "False label"}
COLOR = {"c1": "#2a78d6", "c2": "#eb6834", "c3": "#1baf7a"}      # categorical slots 1-3 (validated all-pairs)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "legend.frameon": False, "lines.solid_capstyle": "round",
})


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


def games(folder, c):
    return [load(f) for f in sorted(glob.glob(os.path.join(HERE, folder, f"{c}_*_seed*.jsonl")))]


def ref_lines(ax, x_right):
    for y, name in ((PM, "monopoly 1.93"), (PN, "competitive 1.47")):
        ax.axhline(y, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.text(x_right, y, f" {name}", va="center", ha="left", fontsize=7, color=INK2, clip_on=False)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def mean_se(vals):
    m = st.mean(vals)
    se = st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0
    return m, se


# ---------- Figure 1: Experiment 1 price paths ----------
def fig_exp1_prices():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, c in zip(axes, COND):
        G = games("runs", c)
        for R in G:
            ax.plot([r["round"] for r in R], [(r["pA"] + r["pB"]) / 2 for r in R],
                    color=COLOR[c], lw=1.3, alpha=0.85)
        ax.set_title(f"{SHORT[c]} (n={len(G)})", loc="left")
        ax.set_xlim(0, 200)
        ax.set_ylim(1.3, 3.0)
        ax.set_xlabel("Round")
        ref_lines(ax, 200) if c == "c3" else [ax.axhline(y, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1) for y in (PM, PN)]
    axes[0].set_ylabel("Mean price of the two firms")
    fig.suptitle("Experiment 1: the false label does not change where prices settle", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.93, 0.98))
    save(fig, "fig1_exp1_prices")


# ---------- Figure 2: probes, say vs do ----------
def probe_shares(folder):
    res = {}
    for c in COND:
        pr = [r for R in games(folder, c) for r in R if "probeA" in r]
        nl = [r for r in pr if "probeA_nolabel" in r]
        b = lambda r, k: (r.get(k) or {}).get("answer") == "B"
        res[c] = (sum(b(r, "probeA") for r in pr) / len(pr) if pr else None,
                  sum(b(r, "probeA_nolabel") for r in nl) / len(nl) if nl else None, len(pr))
    return res


def fig_probes():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, (folder, title) in zip(axes, (("runs", "Experiment 1 (learn demand)"),
                                          ("runs_table", "Experiment 2 (profit table)"))):
        res = probe_shares(folder)
        x = range(len(COND))
        w = 0.36
        for i, c in enumerate(COND):
            withl, without, n = res[c]
            ax.bar(i - w / 2, withl * 100, w, color=COLOR[c], edgecolor=SURF, linewidth=1)
            ax.text(i - w / 2, withl * 100 + 2, f"{withl:.0%}", ha="center", fontsize=7.5, color=INK)
            if without is not None:
                ax.bar(i + w / 2, without * 100, w, color=SURF, edgecolor=COLOR[c], linewidth=1.4, hatch="////")
                ax.text(i + w / 2, without * 100 + 2, f"{without:.0%}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(list(x), [SHORT[c] for c in COND])
        ax.set_ylim(0, 112)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel('% of probes answering\n"competitor wants to beat us"')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=INK2, label="asked with the label in the prompt"),
                        Patch(facecolor=SURF, edgecolor=INK2, hatch="////", label="same prompt, label removed")],
               loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("What A says about its rival follows the label, not the data", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.2)
    save(fig, "fig2_probes")


# ---------- Figure 3: Experiment 1 shock ----------
def fig_shock():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, c in zip(axes, COND):
        files = sorted(glob.glob(os.path.join(HERE, "runs", "shock", f"{c}_*_at*.jsonl")))
        for f in files:
            br = {r["round"]: r for r in load(f)}
            at = next(r["round"] for r in br.values() if r.get("shock")) - 1
            main = {r["round"]: r for r in load(os.path.join(HERE, "runs", os.path.basename(f).split("_at")[0] + ".jsonl"))}
            pre = st.mean(br[t]["pA"] for t in range(at - 4, at + 1))
            ts = [t for t in range(at - 5, at + 31) if t in br]
            ax.plot([t - at - 1 for t in ts if t in main], [main[t]["pA"] - pre for t in ts if t in main],
                    color=MUTED, lw=0.9, ls=(0, (2, 2)), alpha=0.8)
            ax.plot([t - at - 1 for t in ts], [br[t]["pA"] - pre for t in ts], color=COLOR[c], lw=1.4, alpha=0.9)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.text(0.6, 0.97, "B cuts once", fontsize=7, color=INK2, va="top", transform=ax.get_xaxis_transform())
        ax.set_title(f"{SHORT[c]} (n={len(files)})", loc="left")
        ax.set_xlabel("Rounds after B's forced price cut")
        ax.set_xlim(-5, 30)
    axes[0].set_ylabel("A's price minus its\npre-shock price")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=INK2, lw=1.4, label="shock copy (one line per game)"),
                        Line2D([], [], color=MUTED, lw=0.9, ls=(0, (2, 2)), label="same game, no shock (control)")],
               loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Experiment 1 shock: A matches the cut in every condition; the label does not change the response", x=0.01,
                 ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.2)
    save(fig, "fig3_shock")


# ---------- Figure 4: Experiment 2 ----------
def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, centre - half, centre + half


def fig_exp2():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [2.2, 1]})
    ends = {}
    for c in COND:
        G = games("runs_table", c)
        T = min(len(R) for R in G)
        m, lo, hi = [], [], []
        for t in range(T):
            mu, se = mean_se([(R[t]["pA"] + R[t]["pB"]) / 2 for R in G])
            m.append(mu); lo.append(mu - se); hi.append(mu + se)
        xs = list(range(1, T + 1))
        ax.fill_between(xs, lo, hi, color=COLOR[c], alpha=0.15, linewidth=0)
        ax.plot(xs, m, color=COLOR[c], lw=2, label=f"{SHORT[c]} (n={len(G)})")
        ends[c] = m[-1]
        k = sum(any(r["pA"] >= 1.85 and r["pB"] >= 1.85 for r in R) for R in G)
        p, l, h = wilson(k, len(G))
        i = COND.index(c)
        ax2.plot([l * 100, h * 100], [i, i], color=COLOR[c], lw=2)
        ax2.plot(p * 100, i, "o", color=COLOR[c], ms=8, mec=SURF, mew=1.5)
        ax2.text(h * 100 + 3, i, f"{k}/{len(G)}", va="center", fontsize=7.5, color=INK)
    # direct labels at line ends, nudged apart
    order = sorted(COND, key=lambda c: ends[c])
    ys, last = {}, -9
    for c in order:
        ys[c] = max(ends[c], last + 0.05); last = ys[c]
    for c in COND:
        ax.text(20.4, ys[c], SHORT[c], color=INK, fontsize=7.5, va="center")
    ax.set_xlim(1, 20)
    ax.set_ylim(1.4, 2.05)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean price of the two firms\n(mean ± 1 SE across games)")
    ax.axhline(PM, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(19.8, PM + 0.012, "monopoly 1.93", fontsize=7, color=INK2, ha="right")
    ax.axhline(PN, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(19.8, PN + 0.012, "competitive 1.47", fontsize=7, color=INK2, ha="right")
    ax.set_title("Price path", loc="left")
    ax2.set_yticks(range(3), [SHORT[c] for c in COND])
    ax2.set_ylim(-0.6, 2.6)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 110)
    ax2.set_xlabel("% of games where both firms\nreached 1.85 (95% CI)")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Reached near-monopoly", loc="left")
    fig.suptitle("Experiment 2: with a profit table, the false label makes collusion less likely", x=0.01,
                 ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.2)
    save(fig, "fig4_exp2")


if __name__ == "__main__":
    fig_exp1_prices()
    fig_probes()
    fig_shock()
    fig_exp2()
