"""Descriptive tables for the Phase 4b final report, computed from saved
evaluation outputs (eval_report.json + per_query_rows.csv). Adds descriptive
statistics only (medians, percentiles, failure distributions, Spearman of
features vs grid-oracle ef); it never changes any gate input.

Usage: python python/phase4b_report.py <eval_dir> <evaluate.yaml> <mode> > tables.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ev_dir, cfg_path, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rep = json.loads((ev_dir / "eval_report.json").read_text())
rows = pd.read_csv(ev_dir / "per_query_rows.csv")
cfg = yaml.safe_load(cfg_path.read_text())
es = cfg["eval_sets"][mode]
out = []
p = out.append


def pct(x):
    return f"{100 * x:+.1f}%"


def ci(c, scale=100, sign=True):
    f = (lambda v: f"{scale * v:+.1f}") if sign else (lambda v: f"{scale * v:.1f}")
    return f"[{f(c[0])}, {f(c[1])}]"


# ---- integrity of the per-query table ---------------------------------------
num = rows.select_dtypes("number")
dupe = rows.duplicated(["seed", "query_id", "group", "contract", "level"]).sum()
cnt = rows.groupby(["group", "contract", "level", "seed"]).size()
p(f"- per-query table: {len(rows)} rows; NaN {int(num.isna().sum().sum())}; "
  f"Inf {int(np.isinf(num.to_numpy()).sum())}; duplicated keys {int(dupe)}; "
  f"rows per (group, contract, level, seed): min {cnt.min()}, max {cnt.max()} "
  f"(expected {rep['n_queries']}).\n")

# ---- gate tables ---------------------------------------------------------
for contract, levels in (("per_query", ["0.9", "0.95", "0.99"]), ("mean_recall", ["0.95", "0.97", "0.99"])):
    for grp in ("primary", "lid_ablation", "single_feature"):
        p(f"\n#### {grp} — {contract}\n")
        qn = "success" if contract == "per_query" else "mean recall"
        p(f"| level | router cost (mean / median) | B1 cost (mean / median) | abs. saving | saving vs B1 [95% CI] | "
          f"router {qn} | B1 {qn} | Δ{qn} [95% CI] | saving vs B2 [95% CI] | Wilcoxon p (r_rb) | "
          f"crit 1 | crit 2 | crit 3 | crit 4 | gate |")
        p("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for lv in levels:
            r = rep["results"][f"{grp}|{contract}|{float(lv)}"]
            pt, c, g, w = r["point"], r["ci95"], r["gate"], r["wilcoxon_seed_averaged"]
            rq, bq = ((pt["router_success"], pt["b1_success"]) if contract == "per_query"
                      else (pt["router_mean_recall"], pt["b1_mean_recall"]))
            dq, dqc = ((pt["success_diff"], c["success_diff"]) if contract == "per_query"
                       else (pt["mean_recall_diff"], c["mean_recall_diff"]))
            cr = list(g["criteria"].values())
            p(f"| {lv} | {pt['router_cost']:.0f} / {pt['router_median_cost']:.0f} | {pt['b1_cost']:.0f} / {pt['b1_median_cost']:.0f} | "
              f"{pt['abs_saving_vs_b1']:+.0f} | {pct(pt['saving_vs_b1'])} {ci(c['saving_vs_b1'])} | {rq:.4f} | {bq:.4f} | "
              f"{dq:+.4f} [{dqc[0]:+.4f}, {dqc[1]:+.4f}] | {pct(pt['saving_vs_b2'])} {ci(c['saving_vs_b2'])} | "
              f"{w['p_value']:.2e} ({w['rank_biserial']:+.3f}) | {'✓' if cr[0] else '✗'} | {'✓' if cr[1] else '✗'} | "
              f"{'✓' if cr[2] else '✗'} | {'✓' if cr[3] else '✗'} | {'**PASS**' if g['pass'] else '**FAIL**'} |")

# ---- per seed (primary) ----------------------------------------------------
p("\n#### Per seed — primary router\n")
p("| contract | level | seed | family | τ | B1 mix (lo/hi, w_hi) | router cost | B1 cost | B2 cost | saving vs B1 | router q | B1 q | Δq | McNemar exact p (router-only / B1-only) |")
p("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for contract, levels in (("per_query", ["0.9", "0.95", "0.99"]), ("mean_recall", ["0.95", "0.97", "0.99"])):
    for lv in levels:
        r = rep["results"][f"primary|{contract}|{float(lv)}"]
        for s, d in r["per_seed"].items():
            rq = d["router_success"] if contract == "per_query" else d["router_mean_recall"]
            bq = d["b1_success"] if contract == "per_query" else d["b1_mean_recall"]
            m = d["b1_mix"]
            p(f"| {contract} | {lv} | {s} | {d['family']} | {d['tau']:.3f} | {m['ef_lo']}/{m['ef_hi']}, {m['w_hi']:.3f} | "
              f"{d['router_cost']:.0f} | {d['b1_cost']:.0f} | {d['b2_cost']:.0f} | {pct(d['saving_vs_b1'])} | {rq:.4f} | {bq:.4f} | "
              f"{rq - bq:+.4f} | {d['mcnemar_exact_p']:.3g} ({d['mcnemar_router_only_success']}/{d['mcnemar_b1_only_success']}) |")

# ---- descriptive: quality, failures, cost distribution (primary) -----------
p("\n#### Descriptive — primary router vs B1 (pooled over seeds; query × seed rows)\n")
p("| contract | level | policy | success | mean R@10 | median R@10 | failures at R=0.9 / 0.8 / ≤0.7 | probe mean | search mean | total p50 / p90 / p99 | fallback |")
p("|---|---|---|---|---|---|---|---|---|---|---|")
for (contract, lv), g in rows[rows["group"] == "primary"].groupby(["contract", "level"]):
    for pol, rc, sc, cc in (("router", "router_recall", "router_success", "router_cost"),
                            ("B1", "b1_recall", "b1_success", "b1_cost")):
        r10 = g[rc].round(4)
        fails = r10[r10 < 1 - 1e-9]
        probe = g["router_probe"].mean() if pol == "router" else 0.0
        search = g["router_search"].mean() if pol == "router" else g["b1_cost"].mean()
        fb = f"{g['router_fallback'].mean():.4f}" if pol == "router" else "—"
        p(f"| {contract} | {lv} | {pol} | {g[sc].mean():.4f} | {g[rc].mean():.4f} | {g[rc].median():.2f} | "
          f"{int((fails == 0.9).sum())} / {int((fails == 0.8).sum())} / {int((fails <= 0.7).sum())} | {probe:.0f} | {search:.0f} | "
          f"{np.percentile(g[cc], 50):.0f} / {np.percentile(g[cc], 90):.0f} / {np.percentile(g[cc], 99):.0f} | {fb} |")

# ---- regret decomposition --------------------------------------------------
p("\n#### Regret decomposition (mean over seeds; primary router)\n")
p("| contract | level | avoidable failure | unavoidable failure | overspend on successes (dc) | underspend on failures (dc, not netted) | probe overhead (dc) | contract-oracle cost at realised quality | contract regret (dc) | legacy regret (dc) |")
p("|---|---|---|---|---|---|---|---|---|---|")
for contract, levels in (("per_query", ["0.9", "0.95", "0.99"]), ("mean_recall", ["0.95", "0.97", "0.99"])):
    for lv in levels:
        g = rep["results"][f"primary|{contract}|{float(lv)}"]["regret"]
        p(f"| {contract} | {lv} | {g['avoidable_failure_rate']:.4f} | {g['unavoidable_failure_rate']:.4f} | "
          f"{g['overspend_on_successes_mean']:.0f} | {g['underspend_on_avoidable_failures_mean']:.0f} | {g['mean_probe_cost']:.0f} | "
          f"{g['contract_oracle_cost_at_realised_quality']:.0f} | {g['contract_regret']:.0f} | {g['legacy_regret_total_minus_grid_oracle_dc']:.0f} |")

# ---- sensitivities & pure B1 -----------------------------------------------
p("\n#### Sensitivities (non-gating) and secondary pure-B1\n")
p("| contract | level | saving vs B1: additive (primary) | shared descent | probe-free | pure B1 ef | pure-B1 cost | pure-B1 quality | router saving vs pure B1 |")
p("|---|---|---|---|---|---|---|---|---|")
for contract, levels in (("per_query", ["0.9", "0.95", "0.99"]), ("mean_recall", ["0.95", "0.97", "0.99"])):
    for lv in levels:
        r = rep["results"][f"primary|{contract}|{float(lv)}"]
        pt = r["point"]
        pure_ef = r["per_seed"]["42"]["b1_mix"]
        bp = pt["b1_pure"]
        q = bp["success"] if contract == "per_query" else bp["mean_recall"]
        p(f"| {contract} | {lv} | {pct(pt['saving_vs_b1'])} | {pct(pt['sensitivity_shared_descent_saving_vs_b1'])} | "
          f"{pct(pt['sensitivity_probe_free_saving_vs_b1'])} | (per seed in specs) | {bp['cost']:.0f} | {q:.4f} | {pct(bp['saving_router_vs'])} |")

# ---- difficulty strata ------------------------------------------------------
p("\n#### Difficulty strata (frozen: easy ≤ 18 < medium ≤ 48 < hard; per-seed grid oracle) — primary router\n")
p("| contract | level | stratum | query×seed n | router cost | B1 cost | router success | B1 success |")
p("|---|---|---|---|---|---|---|---|")
for contract, levels in (("per_query", ["0.9", "0.95", "0.99"]),):
    for lv in levels:
        for gname, d in rep["results"][f"primary|{contract}|{float(lv)}"]["difficulty"].items():
            p(f"| {contract} | {lv} | {gname} | {d['n_query_seed']} | {d['router_cost']:.0f} | {d['b1_cost']:.0f} | "
              f"{d['router_success']:.4f} | {d['b1_success']:.4f} |")

# ---- oracles ----------------------------------------------------------------
p("\n#### Oracle references (kept separate; per seed)\n")
p("| seed | 1. grid min-ef oracle: reached / censored / mean search dc | 2. candidate oracle: achievable / mean total dc | 3. contract oracle per-query 0.90 / 0.95 / 0.99 (dc) | 3. contract oracle mean-recall 0.95 / 0.97 / 0.99 (dc) | difficulty counts e/m/h |")
p("|---|---|---|---|---|---|")
for s, o in rep["oracles"].items():
    g1, g2, g3 = o["1_grid_min_ef_oracle"], o["2_candidate_success_oracle"], o["3_contract_oracle"]
    pq = " / ".join(f"{g3[f'per_query_{x}']['mean_cost']:.0f}" for x in (0.9, 0.95, 0.99))
    mr = " / ".join(f"{g3[f'mean_recall_{x}']:.0f}" for x in (0.95, 0.97, 0.99))
    dc = o["difficulty_counts"]
    p(f"| {s} | {g1['reached']} / {g1['censored']} / {g1['mean_search_cost_reached']:.0f} | {g2['achievable']} / "
      f"{g2['mean_total_cost_achievable']:.0f} | {pq} | {mr} | {dc['easy']}/{dc['medium']}/{dc['hard']} |")

# ---- Spearman (descriptive): features vs grid-oracle ef ---------------------
p("\n#### Spearman ρ, primary features vs grid-oracle ef on this query set (descriptive)\n")
p("| seed | knn_dist | centroid_dist | score_concentration | lid |")
p("|---|---|---|---|---|")
for s, sc in es["seeds"].items():
    f = pd.read_csv(Path(sc["features_run"]) / "features.csv")
    f = f[f["split"] == es["split_value"]]
    lab = pd.read_csv(Path(sc["oracle_run"]) / "oracle_labels.csv")
    lab = lab[(lab["split"] == es["split_value"]) & (lab["reached"] == 1)]
    m = f.merge(lab[["query_id", "oracle_ef"]], on="query_id")
    vals = [stats.spearmanr(m[c], m["oracle_ef"]).statistic for c in ("knn_dist", "centroid_dist", "score_concentration", "lid")]
    p(f"| {s} | " + " | ".join(f"{v:+.3f}" for v in vals) + " |")

pg = rep["phase4_gate"]
p(f"\n**Phase 4 gate (frozen rule: pass at all three S\\*):** {'PASS' if pg['PASS'] else 'FAIL'} — per S\\*: {pg['per_S_star']}")
print("\n".join(out))
