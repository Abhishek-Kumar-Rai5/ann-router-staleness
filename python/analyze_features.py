"""Validate a Phase 3 feature run (design doc §21 checkpoint).

Checkpoint (§21): "No NaNs or degenerate constants; at least one feature
visibly correlates with the oracle label."

Label-free checks (all queries — features are inputs, not outcomes):
  finite      no NaN/inf in any feature (edge-case convention counts reported)
  nonconst    every feature varies (std > 0, >= MIN_DISTINCT distinct values)
  probe_cost  probe distance counts equal the Phase 1 rows at (k=10, ef=10)
              — the probe is the same deterministic search
  centroid    centroid_dist matches a NumPy recomputation of the base centroid
  knn_bound   probe d_k >= exact d_k from the verified ground truth (an
              approximate search can only overestimate)

Label check — TRAINING SPLIT ONLY. Test-split oracle labels are never read:
the labels file is filtered to split == "train" before it is joined.
  correlates  at least one core feature has |Spearman rho| with the oracle ef
              >= MIN_ABS_RHO and a bootstrap 95% CI excluding 0. Criteria fixed
              before the correlation was computed.

Usage:
  python python/analyze_features.py <features_run> --oracle <oracle_run>
      --phase1 <phase1_run> --gt <gt_cache_prefix> --base <base.fvecs>
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

CORE = ["knn_dist", "centroid_dist", "score_concentration"]
ABLATION = ["lid"]
ALL = CORE + ABLATION
MIN_DISTINCT = 100
MIN_ABS_RHO = 0.2
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20261002


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return raw.reshape(-1, dim + 1)[:, 1:].copy().view(dtype)


def spearman(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def bootstrap_ci(x, y, rng):
    n = len(x)
    stats = []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, n, n)
        # Re-rank inside each resample so ties are handled per resample.
        stats.append(spearman(x[i], y[i]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def check(report, name, passed, **details):
    report["checks"][name] = {"pass": bool(passed), **details}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--phase1", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    feats = pd.read_csv(run / "features.csv")
    nq = len(feats)
    report = {"run": run.name, "num_queries": nq, "checks": {}}

    # --- label-free checks (all queries) ------------------------------------
    finite = np.isfinite(feats[ALL].to_numpy()).all()
    check(report, "finite", finite,
          lid_zero=int((feats["lid"] == 0).sum()),
          lid_inf=int(np.isinf(feats["lid"]).sum()),
          concentration_eq_1=int((feats["score_concentration"] == 1).sum()),
          knn_dist_zero=int((feats["knn_dist"] == 0).sum()))
    distinct = {f: int(feats[f].nunique()) for f in ALL}
    std = {f: float(feats[f].std()) for f in ALL}
    check(report, "nonconst",
          all(distinct[f] >= MIN_DISTINCT and std[f] > 0 for f in ALL),
          distinct=distinct, std=std)

    p1 = pd.read_csv(Path(args.phase1) / "results.csv")
    p1 = p1[(p1["ef"] == 10) & (p1["k"] == 10)].set_index("query_id")
    same_cost = (feats.set_index("query_id")["probe_distance_computations"]
                 == p1["distance_computations"]).all()
    check(report, "probe_cost", bool(same_cost) and len(p1) == nq,
          mean_probe_distance_computations=float(
              feats["probe_distance_computations"].mean()),
          mean_lid_probe_distance_computations=float(
              feats["lid_probe_distance_computations"].mean()))

    base = np.memmap(args.base, dtype=np.float32, mode="r").reshape(-1, 129)[:, 1:]
    mu = np.zeros(base.shape[1])
    for s in range(0, base.shape[0], 100000):
        mu += base[s:s + 100000].astype(np.float64).sum(axis=0)
    mu /= base.shape[0]
    queries = read_vecs(str(Path(args.base).with_name("sift_query.fvecs")),
                        np.float32)[:nq]
    cd = np.linalg.norm(queries.astype(np.float64) - mu, axis=1)
    centroid_err = float(np.abs(cd - feats["centroid_dist"].to_numpy()).max())
    check(report, "centroid", centroid_err < 1e-6, max_abs_error=centroid_err)

    gt_d = read_vecs(args.gt + ".fvecs", np.float32)[:nq]
    exact_dk = np.sqrt(gt_d[:, 9].astype(np.float64))
    slack = feats["knn_dist"].to_numpy() - exact_dk
    check(report, "knn_bound", bool((slack >= -1e-4).all()),
          probe_dk_equals_exact_fraction=float((np.abs(slack) < 1e-4).mean()),
          median_relative_overestimate=float(np.median(slack / exact_dk)))

    # --- label check: TRAINING SPLIT ONLY -----------------------------------
    labels = pd.read_csv(Path(args.oracle) / "oracle_labels.csv")
    labels = labels[labels["split"] == "train"]          # test never read
    assert set(labels["split"]) == {"train"}
    train = feats[feats["split"] == "train"].merge(
        labels[["query_id", "reached", "oracle_ef",
                "oracle_distance_computations"]], on="query_id")
    assert len(train) == (feats["split"] == "train").sum()
    assert train["reached"].all(), "train split has no censored labels"
    y = train["oracle_ef"].to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    corr = {}
    for f in ALL:
        x = train[f].to_numpy()
        lo, hi = bootstrap_ci(x, y, rng)
        corr[f] = {"spearman_vs_oracle_ef": spearman(x, y),
                   "ci95": [lo, hi],
                   "spearman_vs_oracle_distance_computations":
                       spearman(x, train["oracle_distance_computations"]
                                .to_numpy())}
    report["train_correlation"] = corr
    ok = [f for f in CORE if abs(corr[f]["spearman_vs_oracle_ef"]) >= MIN_ABS_RHO
          and (corr[f]["ci95"][0] > 0 or corr[f]["ci95"][1] < 0)]
    check(report, "correlates", bool(ok), evaluated_on="train",
          criteria={"min_abs_rho": MIN_ABS_RHO, "ci_excludes_zero": True},
          core_features_meeting_criteria=ok)
    report["train_feature_intercorrelation"] = (
        train[ALL].rank().corr().round(4).to_dict())

    # Descriptive feature distributions (inputs only; per split).
    report["feature_quantiles"] = {
        s: {f: {str(p): float(feats.loc[feats["split"] == s, f].quantile(p))
                for p in (0.01, 0.25, 0.5, 0.75, 0.99)} for f in ALL}
        for s in ("train", "test")}

    report["all_checks_pass"] = all(c["pass"] for c in report["checks"].values())
    (run / "feature_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # Figure: each feature vs oracle ef, training split only.
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, f in zip(axes, ALL):
        ax.hexbin(train[f], train["oracle_ef"], yscale="log", gridsize=40,
                  bins="log", cmap="viridis", mincnt=1)
        ax.set_xlabel(f + (" (ablation)" if f in ABLATION else ""))
        ax.set_ylabel("oracle ef (train)")
        ax.set_title(f"Spearman {corr[f]['spearman_vs_oracle_ef']:+.3f}")
    fig.suptitle(f"{run.name} — features vs S0 oracle ef, TRAIN split only",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(run / "features_vs_oracle_train.png", dpi=130)

    print(json.dumps({k: report[k] for k in
                      ("checks", "train_correlation", "all_checks_pass")},
                     indent=1))
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
