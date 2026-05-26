"""Validate and summarise a Phase 2 oracle run (design doc §21 checkpoint).

Checks (each recorded as pass/fail in oracle_report.json):
  split       sizes match config; each query exactly once; the split column in
              every output file agrees with split.csv; the stored split file is
              byte-identical to the run's copy.
  complete    every query has a full curve (one row per grid ef) and one label
              row; reached labels have every field; censored ones are listed.
  rederive    labels re-derived from oracle_curves.csv by an independent
              Python implementation of the label rule match the C++ labels.
  phase1      for ef values in both grids, the oracle curves (parallel
              SearchBatch) equal the Phase 1 rows (serial Search) exactly.
  nondegen    the oracle-ef distribution is non-degenerate (criteria below).
  beats_fixed the oracle uses fewer distance computations than the cheapest
              fixed ef reaching the same mean recall (§21: "oracle beats
              fixed-ef baselines at matched recall").

Distributions and comparisons are reported per split. The training split is
the one later used for router fitting; test-split figures are descriptive
only — nothing in Phase 2 is selected or tuned from them.

Usage: python python/analyze_oracle.py <oracle_run_dir> [--phase1 <run_dir>]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

TOL = 1e-9

# Non-degeneracy criteria, fixed before looking at the SIFT1M distribution.
MIN_DISTINCT_LABELS = 10      # oracle uses at least 10 different ef values
MAX_SINGLE_LABEL_SHARE = 0.5  # no one ef value labels more than half the set
MIN_P90_OVER_P10 = 2.0        # 90th/10th percentile oracle ef ratio


def stable_reach_index(recall: np.ndarray, target: float):
    """Python reimplementation of ars::DeriveOracleLabel (stable reach)."""
    ok = recall >= target - TOL
    if not ok[-1]:
        return None, None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    first = int(np.argmax(ok))
    return i, first


def check(report, name, passed, **details):
    report["checks"][name] = {"pass": bool(passed), **details}


def matched_recall(curves_split, labels_split, grid, rcol):
    """Oracle policy vs the fixed-ef family on one query subset."""
    eff_ef = labels_split["oracle_ef"].fillna(grid[-1]).astype(int)
    pick = curves_split.merge(
        pd.DataFrame({"query_id": labels_split["query_id"], "ef": eff_ef}),
        on=["query_id", "ef"])
    oracle_recall = pick[rcol].mean()
    oracle_cost = pick["distance_computations"].mean()
    fixed = curves_split.groupby("ef").agg(
        recall=(rcol, "mean"), cost=("distance_computations", "mean"))
    reaching = fixed[fixed["recall"] >= oracle_recall - TOL]
    out = {
        "oracle_mean_recall": float(oracle_recall),
        "oracle_mean_distance_computations": float(oracle_cost),
    }
    if len(reaching):
        ef_m = int(reaching.index.min())
        out.update({
            "cheapest_fixed_ef_matching_recall": ef_m,
            "fixed_mean_recall": float(fixed.loc[ef_m, "recall"]),
            "fixed_mean_distance_computations": float(fixed.loc[ef_m, "cost"]),
            "fixed_over_oracle_cost": float(fixed.loc[ef_m, "cost"] / oracle_cost),
        })
    else:
        out["cheapest_fixed_ef_matching_recall"] = None
    # Reverse view: best fixed-ef recall within the oracle's mean budget.
    within = fixed[fixed["cost"] <= oracle_cost]
    if len(within):
        out["fixed_recall_at_oracle_budget"] = float(within["recall"].max())
        out["fixed_ef_at_oracle_budget"] = int(within["recall"].idxmax())
    return out, fixed


def describe(labels_split, grid):
    ef = labels_split.loc[labels_split["reached"] == 1, "oracle_ef"]
    dc = labels_split.loc[labels_split["reached"] == 1,
                          "oracle_distance_computations"]
    q = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    counts = ef.value_counts()
    return {
        "n": int(len(labels_split)),
        "reached": int(len(ef)),
        "censored": int(len(labels_split) - len(ef)),
        "oracle_ef_quantiles": {str(p): float(ef.quantile(p)) for p in q},
        "oracle_ef_mean": float(ef.mean()),
        "oracle_distance_computations_quantiles":
            {str(p): float(dc.quantile(p)) for p in q},
        "distinct_oracle_ef_values": int(ef.nunique()),
        "largest_single_ef_share": float(counts.max() / len(ef)),
        "most_common_oracle_ef": int(counts.idxmax()),
        "share_at_min_grid_ef": float((ef == grid[0]).mean()),
        "p90_over_p10": float(ef.quantile(0.9) / ef.quantile(0.1)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--phase1", help="Phase 1 run dir for the cross-check")
    args = ap.parse_args()
    run = Path(args.run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    meta = json.loads((run / "metadata.json").read_text())
    target = float(cfg["oracle"]["target_recall"])
    rdef = cfg["oracle"]["recall_definition"]
    rcol = "recall_tie_aware" if rdef == "tie_aware" else "recall"
    grid = meta["oracle"]["ef_grid"]

    split = pd.read_csv(run / "split.csv")
    labels = pd.read_csv(run / "oracle_labels.csv")
    curves = pd.read_csv(run / "oracle_curves.csv")
    nq = len(split)
    report = {"run": run.name, "target_recall": target,
              "recall_definition": rdef, "grid_size": len(grid),
              "grid_min": grid[0], "grid_max": grid[-1], "checks": {}}

    # --- split ---------------------------------------------------------------
    n_test = int((split["split"] == "test").sum())
    stored = Path(cfg["split"]["path"])
    stored_same = stored.exists() and stored.read_bytes() == (run / "split.csv").read_bytes()
    label_split_ok = (labels.set_index("query_id")["split"]
                      .equals(split.set_index("query_id")["split"]))
    curve_split = curves.groupby("query_id")["split"].agg(lambda s: s.unique().tolist())
    curve_split_ok = all(len(v) == 1 for v in curve_split) and (
        curve_split.map(lambda v: v[0]).equals(split.set_index("query_id")["split"]))
    check(report, "split",
          n_test == cfg["split"]["test_size"]
          and (split["query_id"] == np.arange(nq)).all()
          and set(split["split"]) == {"train", "test"}
          and stored_same and label_split_ok and curve_split_ok,
          seed=cfg["split"]["seed"], train=nq - n_test, test=n_test,
          stored_file=str(stored), stored_file_identical=bool(stored_same),
          fnv1a64=meta["split"]["fnv1a64_of_csv"])

    # --- completeness --------------------------------------------------------
    per_q = curves.groupby("query_id")["ef"].apply(list)
    full_curves = len(per_q) == nq and all(v == grid for v in per_q)
    reached = labels[labels["reached"] == 1]
    fields = ["oracle_ef", "oracle_ef_first_reach",
              "oracle_distance_computations", "oracle_recall",
              "oracle_recall_tie_aware"]
    censored_ids = labels.loc[labels["reached"] == 0, "query_id"].tolist()
    check(report, "complete",
          full_curves and len(labels) == nq
          and labels["query_id"].is_unique
          and not reached[fields].isna().any().any()
          and (reached["oracle_recall_tie_aware" if rdef == "tie_aware"
                       else "oracle_recall"] >= target - TOL).all(),
          curve_rows=len(curves), label_rows=len(labels),
          censored=len(censored_ids), censored_query_ids=censored_ids,
          censored_splits=labels.loc[labels["reached"] == 0, "split"]
          .value_counts().to_dict())

    # --- independent re-derivation -------------------------------------------
    rmat = curves.pivot(index="query_id", columns="ef", values=rcol)[grid].to_numpy()
    mismatches = []
    first_differs = 0
    for q in range(nq):
        idx, first = stable_reach_index(rmat[q], target)
        row = labels.iloc[q]
        if idx is None:
            ok = row["reached"] == 0
        else:
            ok = (row["reached"] == 1 and int(row["oracle_ef"]) == grid[idx]
                  and int(row["oracle_ef_first_reach"]) == grid[first])
            first_differs += int(first != idx)
        if not ok:
            mismatches.append(q)
    check(report, "rederive", not mismatches, mismatches=mismatches[:20],
          queries_where_first_reach_differs_from_stable=first_differs)

    # --- Phase 1 cross-check -------------------------------------------------
    if args.phase1:
        p1 = pd.read_csv(Path(args.phase1) / "results.csv")
        common = sorted(set(p1["ef"]) & set(grid))
        m = p1[p1["ef"].isin(common)].merge(
            curves, on=["query_id", "ef"], suffixes=("_p1", "_p2"))
        same = all((m[c + "_p1"] == m[c + "_p2"]).all() for c in
                   ["recall", "recall_tie_aware", "distance_computations"])
        check(report, "phase1", same and len(m) == len(common) * nq,
              common_ef_values=common, rows_compared=len(m))

    # --- distributions -------------------------------------------------------
    report["distribution"] = {
        s: describe(labels[labels["split"] == s], grid) for s in ("train", "test")}
    tr = report["distribution"]["train"]
    check(report, "nondegen",
          tr["distinct_oracle_ef_values"] >= MIN_DISTINCT_LABELS
          and tr["largest_single_ef_share"] <= MAX_SINGLE_LABEL_SHARE
          and tr["p90_over_p10"] >= MIN_P90_OVER_P10,
          evaluated_on="train",
          criteria={"min_distinct": MIN_DISTINCT_LABELS,
                    "max_single_share": MAX_SINGLE_LABEL_SHARE,
                    "min_p90_over_p10": MIN_P90_OVER_P10})

    # --- oracle vs fixed ef at matched recall --------------------------------
    report["matched_recall"] = {}
    fixed_curves = {}
    for s in ("train", "test"):
        res, fixed = matched_recall(curves[curves["split"] == s],
                                    labels[labels["split"] == s], grid, rcol)
        report["matched_recall"][s] = res
        fixed_curves[s] = fixed
    mtr = report["matched_recall"]["train"]
    check(report, "beats_fixed",
          mtr.get("fixed_over_oracle_cost") is not None
          and mtr["fixed_over_oracle_cost"] > 1.0, evaluated_on="train")

    # --- sensitivity (descriptive, not used for any choice) -----------------
    sens = {}
    for alt_def, col in (("tie_aware", "recall_tie_aware"), ("id", "recall")):
        m_alt = curves.pivot(index="query_id", columns="ef", values=col)[grid].to_numpy()
        for alt_t in (0.9, 1.0):
            idxs = [stable_reach_index(m_alt[q], alt_t)[0] for q in range(nq)]
            efs = pd.Series([grid[i] for i in idxs if i is not None])
            sens[f"{alt_def}@{alt_t}"] = {
                "censored": int(sum(i is None for i in idxs)),
                "median_oracle_ef": float(efs.median()),
                "p90_oracle_ef": float(efs.quantile(0.9)),
                "distinct": int(efs.nunique())}
    report["sensitivity_all_queries"] = sens

    report["all_checks_pass"] = all(c["pass"] for c in report["checks"].values())
    (run / "oracle_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # --- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    bins = np.array(grid + [grid[-1] * 1.05])
    for s, c in (("train", "C0"), ("test", "C1")):
        ef = labels.loc[(labels["split"] == s) & (labels["reached"] == 1), "oracle_ef"]
        axes[0].hist(ef, bins=bins, density=True, histtype="step", color=c,
                     label=f"{s} (n={len(ef)})")
        xs = np.sort(ef)
        axes[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, label=s)
    for ax in axes[:2]:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("oracle efSearch")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("density")
    axes[0].set_title(f"Oracle ef (recall@{cfg['oracle']['k']} {rdef} >= {target})")
    axes[1].set_ylabel("fraction of queries")
    axes[1].set_title("Oracle ef CDF")
    f = fixed_curves["train"]
    axes[2].plot(f["cost"], f["recall"], marker=".", label="fixed ef (train)")
    axes[2].scatter([mtr["oracle_mean_distance_computations"]],
                    [mtr["oracle_mean_recall"]], color="C3", zorder=5,
                    label="oracle (train)")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("mean distance computations / query")
    axes[2].set_ylabel(f"mean {rcol}")
    axes[2].set_title("Oracle vs fixed ef (train split)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    fig.suptitle(run.name, fontsize=9)
    fig.tight_layout()
    fig.savefig(run / "oracle_distribution.png", dpi=130)

    print(json.dumps({k: report[k] for k in
                      ("checks", "distribution", "matched_recall",
                       "sensitivity_all_queries", "all_checks_pass")},
                     indent=1, default=str)[:12000])
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
