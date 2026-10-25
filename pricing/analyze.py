"""
Analysis for the pricing experiments. Reads the results folders, prints tables, writes CSVs.

  python analyze.py main     # Experiment 1 (runs/): prices, probes, label mentions, early rounds
  python analyze.py shock    # Experiment 1 shock copies (runs/shock/)
  python analyze.py table    # Experiment 2 (runs_table/): pre-registered early-round predictions
  python analyze.py all

Permutation tests: two-sided, difference in means, 20,000 shuffles, seed 0.
"""
import csv, glob, json, os, random, re, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PN, PM = 1.4725, 1.925                       # Nash and monopoly price (Calvano logit, see experiment.py)
idx = lambda p: (p - PN) / (PM - PN)
NAMES = {"c1": "none", "c2": "told true (long-run)", "c3": "told false (beat you)"}
LABEL_WORDS = re.compile(r"instruct|beat us|beat our|outperform us|out-earn|make more profit than us|"
                         r"aggressive (competitor|rival|opponent)|zero-sum|relative profit", re.I)
TEXTBOOK = re.compile(r"prisoner|nash|game theor|equilibri|tit.for.tat|cartel|collu", re.I)


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return rows


def games(folder, cond):
    return [load(f) for f in sorted(glob.glob(os.path.join(HERE, folder, f"{cond}_*_seed*.jsonl")))]


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    return f"{st.mean(v):.2f}" + (f"±{st.stdev(v):.2f}" if len(v) > 1 else "")


def perm(x, y, n=20000):
    x, y = [a for a in x if a is not None], [a for a in y if a is not None]
    if len(x) < 2 or len(y) < 2:
        return None, None
    obs = st.mean(x) - st.mean(y)
    pool, k, rng, cnt = x + y, len(x), random.Random(0), 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:k]) - st.mean(pool[k:])) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, cnt / n


def answer(r, key):
    return (r.get(key) or {}).get("answer", "?")


def probe_stats(G, truth="A"):
    pr = [r for R in G for r in R if "probeA" in r]
    nl = [r for r in pr if "probeA_nolabel" in r]
    says_B = sum(answer(r, "probeA") == "B" for r in pr) / len(pr) if pr else None
    acc = sum(answer(r, "probeA") == truth for r in pr) / len(pr) if pr else None
    nl_A = sum(answer(r, "probeA_nolabel") == truth for r in nl) / len(nl) if nl else None
    return len(pr), acc, says_B, nl_A


def write_csv(name, rows):
    if not rows:
        return
    path = os.path.join(HERE, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {path}")


def per_game(R):
    fa, fb = R[-1]["pA"], R[-1]["pB"]
    lock = next(r["round"] for i, r in enumerate(R)
                if all(abs(x["pA"] - fa) <= 0.02 and abs(x["pB"] - fb) <= 0.02 for x in R[i:]))
    early = R[1:10]
    react = [R[i + 1]["pA"] - R[i]["pA"] for i in range(min(10, len(R) - 1)) if R[i]["pB"] < R[i]["pA"] - 0.02]
    return {
        "rounds": len(R),
        "round1_A": R[0]["pA"],
        "meanA_r1_5": st.mean(r["pA"] for r in R[:5]),
        "share_A_below_B_r2_10": sum(r["pA"] < r["pB"] - 1e-9 for r in early) / len(early),
        "A_change_after_B_undercut_r1_10": st.mean(react) if react else None,
        "reached_1.85": float(any(r["pA"] >= 1.85 and r["pB"] >= 1.85 for r in R)),
        "final_price_idx_last5": st.mean((idx(r["pA"]) + idx(r["pB"])) / 2 for r in R[-5:]),
        "total_profit": sum(r["piA"] + r["piB"] for r in R),
        "lock_in_round": lock,
        "label_mentions_in_notes": sum(bool(LABEL_WORDS.search(r["plansA"] + r["insightsA"] + r["obsA"])) for r in R),
        "textbook_word_rounds": sum(bool(TEXTBOOK.search(r["plansA"] + r["insightsA"] + r["obsA"])) for r in R),
        "parse_fail": sum((not r.get("parse_okA", True)) + (not r.get("parse_okB", True)) for r in R),
    }


def report(folder, title, metrics, tests=(("c3", "c1"), ("c3", "c2"), ("c2", "c1"))):
    print(f"\n===== {title} ({folder}) =====")
    G = {c: games(folder, c) for c in ("c1", "c2", "c3")}
    G = {c: g for c, g in G.items() if g}
    if not G:
        print("  no data")
        return
    P = {c: [per_game(R) for R in g] for c, g in G.items()}
    rows = []
    for c in P:
        for s, d in enumerate(P[c]):
            rows.append({"cond": c, "game": s, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}})
    print(f"{'metric':34s} " + " ".join(f"{c + ' (n=' + str(len(P[c])) + ')':>16s}" for c in P))
    for m in metrics:
        print(f"{m:34s} " + " ".join(f"{ms([d[m] for d in P[c]]):>16s}" for c in P))
    print("\n  probes: share answering B with label | accuracy with label | accuracy without label")
    for c in G:
        n, acc, b, nl = probe_stats(G[c])
        f = lambda x: "-" if x is None else f"{x:.0%}"
        print(f"  {c} {NAMES[c]:22s} n={n:3d}  says B {f(b):>5s} | acc {f(acc):>5s} | acc w/o label {f(nl):>5s}")
    print("\n  permutation tests (difference in means, two-sided p)")
    for m in metrics:
        out = []
        for a, b in tests:
            if a in P and b in P:
                d, p = perm([x[m] for x in P[a]], [x[m] for x in P[b]])
                if d is not None:
                    out.append(f"{a}-{b} {d:+.3f} (p={p:.3f})")
        print(f"  {m:34s} " + "   ".join(out))
    write_csv(f"analysis_{folder.replace('/', '_')}.csv", rows)


def shock_report():
    print("\n===== Experiment 1: shock copies (runs/shock) =====")
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "runs", "shock", "c*_seed*_at*.jsonl"))):
        R = {r["round"]: r for r in load(f)}
        s = next((r for r in R.values() if r.get("shock")), None)
        if not s:
            continue
        at = s["round"] - 1
        main = {r["round"]: r for r in load(os.path.join(HERE, "runs", os.path.basename(f).split("_at")[0] + ".jsonl"))}
        pre = st.mean(R[t]["pA"] for t in range(at - 4, at + 1))
        nxt = R[at + 2]["pA"]
        rec = next((t - (at + 1) for t in range(at + 2, max(R) - 1)
                    if all(R[u]["pA"] >= pre - 0.02 for u in (t, t + 1, t + 2))), None)
        win = [t for t in range(at + 1, at + 11) if t in R and t in main]
        post = [R[t] for t in (at + 2, at + 5) if t in R and "probeA" in R[t]]
        rows.append({
            "cond": f"c{s['cond']}", "seed": s["seed"], "A_pre": round(pre, 3), "B_shock_price": s["pB"],
            "A_next_round": nxt, "share_of_cut_matched": round((pre - nxt) / (pre - s["pB"]), 2),
            "rounds_to_recover": rec, "A_final_last5": round(st.mean(R[t]["pA"] for t in range(max(R) - 4, max(R) + 1)), 3),
            "B_profit_vs_control_10r": round(sum(R[t]["piB"] - main[t]["piB"] for t in win), 1),
            "post_probe_with_label": " ".join(answer(r, "probeA") for r in post),
            "post_probe_without_label": " ".join(answer(r, "probeA_nolabel") for r in post if "probeA_nolabel" in r),
        })
    for c in ("c1", "c2", "c3"):
        rs = [r for r in rows if r["cond"] == c]
        if not rs:
            continue
        print(f"  {c} {NAMES[c]:22s} n={len(rs)}  matched cut {ms([r['share_of_cut_matched'] for r in rs])}  "
              f"recovered {sum(r['rounds_to_recover'] is not None for r in rs)}/{len(rs)}  "
              f"B profit vs control {ms([r['B_profit_vs_control_10r'] for r in rs])}  "
              f"post-shock probes {' | '.join(r['post_probe_with_label'] + ('/' + r['post_probe_without_label'] if r['post_probe_without_label'] else '') for r in rs)}")
    write_csv("analysis_shock.csv", rows)


MAIN_METRICS = ["final_price_idx_last5", "total_profit", "lock_in_round", "round1_A",
                "share_A_below_B_r2_10", "A_change_after_B_undercut_r1_10", "label_mentions_in_notes"]
TABLE_METRICS = ["round1_A", "meanA_r1_5", "share_A_below_B_r2_10", "A_change_after_B_undercut_r1_10",
                 "reached_1.85", "final_price_idx_last5", "total_profit", "textbook_word_rounds"]

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("main", "all"):
        report("runs", "Experiment 1: main runs", MAIN_METRICS)
    if what in ("shock", "all"):
        shock_report()
    if what in ("table", "all"):
        report("runs_table", "Experiment 2: profit table", TABLE_METRICS)
