"""Extra report figures. Run with the system Python after analyze.py and llm_label.py:
    python make_figures_extra.py
Writes to results/:
  fig0_design.png               schematic of the table, the alliance, U and the timeline
  fig4_positive_control.png     control games: ally lies passed by the ally vs other lies passed
  fig5_target_replies.png       target's replies to U (LLM labels): ACCEPT / HEED / REJECT
  fig6_framing_pilot.png        pilot games (Llama 3.1 8B, earlier rules): framing vs favouritism
  positive_control.txt          test that G > 0 in control
  figures_extra_data.csv        the numbers behind fig4-fig6
"""
import csv
import glob
import json
import os
import re
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import sign_flip                                    # noqa: E402
from make_figures import INK, INK2, GRID, SURFACE, MODELS, wilson  # noqa: E402  same style as fig1-fig3

OUT = os.path.join(HERE, "results")
EXP = os.path.join(HERE, "..", "Collusion_Liars_Bar", "experiments")
REPLY_COLORS = {"ACCEPT": "#1baf7a", "HEED": "#eda100", "REJECT": "#e87ba4"}   # categorical slots 3-5
rows_out = []


def style(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig0():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    seats = {"Ally": (1.2, 3.8), "Target": (3.6, 3.8), "Undercover U": (3.6, 1.4), "Other player": (1.2, 1.4)}
    fills = {"Ally": "#cde2fb", "Target": "#cde2fb", "Undercover U": "#fbd9cb", "Other player": "#ecebe7"}
    for name, (x, y) in seats.items():
        ax.add_patch(FancyBboxPatch((x - 0.9, y - 0.4), 1.8, 0.8, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=fills[name], edgecolor=INK2, linewidth=0.8))
        ax.text(x, y, name, ha="center", va="center", fontsize=9, color=INK)
    def arrow(a, b, text, style="-|>", dashed=False, off=(0, 0.18)):
        p = FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=10, color=INK2, linewidth=1,
                            linestyle="--" if dashed else "-")
        ax.add_patch(p)
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", fontsize=7.5, color=INK2)
    arrow((2.1, 3.6), (2.7, 3.6), "plays to", off=(0, -0.42))
    arrow((2.1, 4.0), (2.7, 4.0), "", style="<|-|>", dashed=True)
    ax.text(2.4, 4.45, "private channel", ha="center", fontsize=7.5, color=INK2)
    arrow((3.6, 1.8), (3.6, 3.4), "messages\n(from round 3)", off=(0.75, -0.1), dashed=True)
    ax.text(2.4, 0.55, "seating order: Ally -> Target (the target decides whether to challenge the ally)",
            ha="center", fontsize=7.5, color=INK2)
    # timeline
    ax.text(5.4, 4.45, "Timeline of a game (about 10 rounds)", fontsize=9, color=INK)
    ax.add_patch(FancyBboxPatch((5.4, 3.55), 1.3, 0.55, boxstyle="round,pad=0.02", facecolor="#ecebe7",
                                edgecolor=INK2, linewidth=0.8))
    ax.text(6.05, 3.82, "rounds 1-2\nno intervention", ha="center", va="center", fontsize=7.5)
    ax.add_patch(FancyBboxPatch((6.85, 3.55), 2.9, 0.55, boxstyle="round,pad=0.02", facecolor="#fbd9cb",
                                edgecolor=INK2, linewidth=0.8))
    ax.text(8.3, 3.82, "round 3 on:\nU intervenes on the target", ha="center", va="center", fontsize=7.5)
    conds = [("control", "no intervention"), ("inform", "\"I know you and X are allied\""),
             ("warn", "\"challenge X once, or I tell everyone\""), ("expose", "public: \"X and Y are allied\""),
             ("provoke", "\"X is allied with Z too and lying to you\"")]
    for i, (c, d) in enumerate(conds):
        y = 2.95 - i * 0.48
        ax.text(5.4, y, c, fontsize=8, color=INK, weight="bold")
        ax.text(6.45, y, d, fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig0_design.png"), dpi=300)
    plt.close(fig)


def fig4(games):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    lines = ["Positive control: G > 0 in control games (round 3 on). Per-game G, sign-flip permutation test vs 0.\n"]
    for i, (key, label, color) in enumerate(MODELS):
        ctrl = [g for g in games if g["model"] == key and g["condition"] == "control"]
        for j, (k, n, name) in enumerate([("al_pass", "al_n", "ally's lie, decided by the ally"),
                                          ("ot_pass", "ot_n", "any other lie")]):
            kk, nn = sum(int(g[k]) for g in ctrl), sum(int(g[n]) for g in ctrl)
            p, lo, hi = wilson(kk, nn)
            x = i * 3 + j
            ax.bar(x, p, width=0.8, color=color, alpha=1.0 if j == 0 else 0.45, edgecolor=SURFACE, linewidth=2)
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], color=INK, elinewidth=1.2, capsize=0)
            ax.text(x, hi + 0.03, f"{kk}/{nn}", ha="center", fontsize=8, color=INK2)
            rows_out.append(dict(figure="fig4", model=key, group=name, k=kk, n=nn, value=p, lo=lo, hi=hi))
        gs = np.array([float(g["G"]) for g in ctrl if g["G"] not in ("", "None")])
        pval = sign_flip(gs)
        lines.append(f"{key}: games with G {len(gs)}, mean per-game G {gs.mean():+.3f}, games with G > 0: "
                     f"{int((gs > 0).sum())}, G < 0: {int((gs < 0).sum())}, G = 0: {int((gs == 0).sum())}, p = {pval:.4f}")
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(["ally's lies", "other lies", "ally's lies", "other lies"])
    for i, (_, label, _) in enumerate(MODELS):
        ax.text(i * 3 + 0.5, -0.17, label, ha="center", fontsize=9, color=INK, transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("lies not challenged")
    ax.set_title("Control games: lies not challenged (round 3 on), Wilson 95% CI", loc="left", fontsize=11)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_positive_control.png"), dpi=300)
    plt.close(fig)
    with open(os.path.join(OUT, "positive_control.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def fig5():
    lab = list(csv.DictReader(open(os.path.join(OUT, "llm_labels_B.csv"), encoding="utf-8-sig")))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ylabels, y = [], 0
    for key, label, _ in MODELS:
        for cond in ["inform", "warn", "provoke"]:
            c = Counter(r["llm_label"] for r in lab if r["model"] == key and r["condition"] == cond)
            n = sum(c.values())
            left = 0.0
            for cat, col in REPLY_COLORS.items():
                share = c[cat] / n if n else 0
                ax.barh(y, share, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.7,
                        label=cat if y == 0 else None)
                if share >= 0.06:                                  # direct labels: contrast is below 3:1
                    ax.text(left + share / 2, y, f"{share:.0%}", ha="center", va="center", fontsize=8, color=INK)
                rows_out.append(dict(figure="fig5", model=key, group=f"{cond}:{cat}", k=c[cat], n=n, value=share, lo="", hi=""))
                left += share
            ylabels.append(f"{label.split()[0]} {cond} (n={n})")
            y += 1
        y += 0.5
    ax.set_yticks([i for i in range(3)] + [i + 3.5 for i in range(3)])
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, 1)
    ax.tick_params(length=0)
    ax.set_title("Target's private replies to U (LLM-labelled)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_target_replies.png"), dpi=300)
    plt.close(fig)


def pilot_counts(files):
    al = [0, 0]; ot = [0, 0]
    for f in files:
        d = json.load(open(f, encoding="utf-8")); m = d["meta"]
        if not m.get("alliance") or d.get("winner") is None:
            continue
        pair, seats = set(m["alliance"]), m["seat_order"]
        for r in d["rounds"]:
            for p in r["play_history"]:
                if p["next_player"] not in seats or all(c in (r["target_card"], "Joker") for c in p["played_cards"]):
                    continue
                b = al if {p["player_name"], p["next_player"]} == pair else ot
                b[0] += 1; b[1] += not p["was_challenged"]
    return al, ot


def fig6():
    arch = lambda name, pat: [f for f in glob.glob(os.path.join(EXP, name, pat))]
    groups = [("neutral\n(control, 4)", arch("archive_rules_v1", "control_s[1-4].json")),
              ("moderate v1\n(control, 4)", arch("archive_moderate_v1", "control_moderate_s[1-4].json")),
              ("moderate v2\n(5 conditions, 1 each)", [f for f in arch("archive_rules_v1", "*_moderate_s1.json")
                                                     if re.search(r"(control|inform|warn|expose|provoke)_moderate_s1", f)]),
              ("strict\n(control, 4)", arch("archive_rules_v1", "control_strict_s[1-4].json"))]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (name, files) in enumerate(groups):
        al, ot = pilot_counts(files)
        for j, (cnt, alpha, who) in enumerate([(al, 1.0, "ally's lies"), (ot, 0.45, "other lies")]):
            p, lo, hi = wilson(cnt[1], cnt[0]) if cnt[0] else (0, 0, 0)
            x = i * 3 + j
            ax.bar(x, p, width=0.8, color=MODELS[1][2], alpha=alpha, edgecolor=SURFACE, linewidth=2,
                   label=who if i == 0 else None)
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], color=INK, elinewidth=1.2, capsize=0)
            ax.text(x, hi + 0.03, f"{cnt[1]}/{cnt[0]}", ha="center", fontsize=8, color=INK2)
            rows_out.append(dict(figure="fig6", model="llama-3.1-8b", group=f"{name.splitlines()[0]}:{who}",
                                 k=cnt[1], n=cnt[0], value=p, lo=lo, hi=hi))
    ax.set_xticks([i * 3 + 0.5 for i in range(len(groups))])
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("lies not challenged")
    ax.set_title("Pilot (Llama 3.1 8B, earlier rules): alliance framing, whole game", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_framing_pilot.png"), dpi=300)
    plt.close(fig)


def main():
    games = [g for g in csv.DictReader(open(os.path.join(OUT, "games.csv"), encoding="utf-8"))]
    fig0()
    fig4(games)
    fig5()
    fig6()
    with open(os.path.join(OUT, "figures_extra_data.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print("written: fig0_design.png, fig4_positive_control.png, fig5_target_replies.png, fig6_framing_pilot.png")


if __name__ == "__main__":
    main()
