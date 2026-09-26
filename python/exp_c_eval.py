"""Analyses the single frozen DARTH evaluation (Experiment C) against the frozen
B1 baseline. The gate criteria are the pre-registered ones; nothing is tuned here.

Usage: python python/exp_c_eval.py configs/exp_c/darth_s0.yaml <eval_run> <repeat_eval_run>
"""

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

POLICY_SHA256 = "bcbe2720044b51edff1b493b956e413958f8503a6a76bfe86f52317cec8ac75d"
MODEL_SHA256 = "d291a2f1f8cbd64f9fa2e744dd0cfc8bcc1b4261aba13915c88d1b94430eb041"
B1_ROWS = "results/phase4b_eval_old_test_second_look_20261002T175740Z/per_query_rows.csv"
ORACLE_RUN = "results/8e605bfc5769_20261001T201040Z"
B, BOOT_SEED = 2000, 20261002
TIMING = ["predictor_seconds", "wall_seconds", "plain_ef50_seconds", "plain_ef53_seconds"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def q(a):
    return {f"p{p}": float(np.percentile(a, p)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)} | {
        "mean": float(np.mean(a)), "min": float(np.min(a)), "max": float(np.max(a))}


def main() -> int:
    C = yaml.safe_load(Path(sys.argv[1]).read_text())
    run, rerun = Path(sys.argv[2]), Path(sys.argv[3])
    checks = {}

    pol = json.loads(Path(C["policy_path"]).read_text())
    seed = int(C.get("index_seed", 42))
    oracle_run = C.get("s0_oracle_run", ORACLE_RUN)
    checks["policy_hash_unchanged"] = sha(C["policy_path"]) == C.get("expected_policy_sha256", POLICY_SHA256)
    checks["model_hash_unchanged"] = (sha(pol["model_path"]) == C.get("expected_model_sha256", MODEL_SHA256)
                                      == pol["model_sha256"])
    checks["index_hash_matches_policy"] = sha(C["index_path"]) == pol["index"]["sha256"]
    for r in (run, rerun):
        res = json.loads((r / "metadata.json").read_text())["result"]
        checks[f"eval_used_frozen_policy_{r.name}"] = (res["ipi"] == pol["ipi"] and res["mpi"] == pol["mpi"]
                                                       and res["ef_cap"] == pol["ef_cap"])

    D = pd.read_csv(run / "darth_eval_rows.csv")
    D2 = pd.read_csv(rerun / "darth_eval_rows.csv")
    keep = [c for c in D.columns if c not in TIMING]
    checks["deterministic_repeat_non_timing_identical"] = D[keep].equals(D2[keep])

    split = pd.read_csv(C["split_path"])
    test_ids = split.loc[split.split == "test", "query_id"].to_numpy()
    train_ids = set(split.loc[split.split == "train", "query_id"])
    checks["eval_queries_are_exactly_test_split"] = np.array_equal(np.sort(D.query_id), np.sort(test_ids))
    checks["no_train_test_overlap"] = not (set(D.query_id) & train_ids)

    b = pd.read_csv(B1_ROWS)
    b = b[(b.seed == seed) & (b.contract == "mean_recall") & (b.level == 0.95) & (b.variant == "primary")]
    b = b[["query_id", "b1_ef", "b1_cost", "b1_recall"]].set_index("query_id").loc[D.query_id].reset_index()
    checks["b1_rows_complete"] = len(b) == len(D) == 2000 and set(b.b1_ef) <= {50, 53}
    cur = pd.read_csv(Path(oracle_run) / "oracle_curves.csv")
    cur = cur[cur.query_id.isin(test_ids)].set_index(["query_id", "ef"])
    oc = cur.loc[list(zip(b.query_id, b.b1_ef))]
    checks["b1_cost_equals_oracle_curve_cost"] = np.array_equal(oc.distance_computations.to_numpy(), b.b1_cost.to_numpy())
    checks["b1_recall_equals_oracle_tie_aware"] = np.allclose(oc.recall_tie_aware.to_numpy(), b.b1_recall.to_numpy())
    plain_dc = np.where(b.b1_ef == 50, D.plain_ef50_dc, D.plain_ef53_dc)
    checks["live_plain_search_cost_equals_b1_cost"] = np.array_equal(plain_dc, b.b1_cost.to_numpy())
    cap = cur.xs(int(pol["ef_cap"]), level="ef").loc[D.query_id]

    dc = D.distance_computations.to_numpy().astype(float)
    checks["recall_in_0_1"] = bool(D.recall_tie_aware.between(0, 1).all() and D.recall_id.between(0, 1).all())
    checks["dc_positive_and_le_plain_cap_plus_one_node"] = bool((dc >= 1).all() and
                                                               (dc <= cap.distance_computations.to_numpy() + 32).all())
    checks["predictor_called_every_query"] = bool((D.predictor_calls >= 1).all())
    stopped = float(D.stopped_early.mean())
    checks["most_queries_terminate_early"] = stopped > 0.5

    bc = b.b1_cost.to_numpy().astype(float)
    rd = D.recall_tie_aware.to_numpy()
    rb = b.b1_recall.to_numpy()
    saving = 1 - dc.mean() / bc.mean()
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(D), size=(B, len(D)))
    sav_bs = 1 - dc[idx].mean(1) / bc[idx].mean(1)
    diff_bs = (bc[idx] - dc[idx]).mean(1)
    qual_bs = (rd[idx] - rb[idx]).mean(1)
    ci = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]  # noqa: E731
    w = stats.wilcoxon(bc, dc)
    diff = bc - dc
    rank_biserial = float((stats.rankdata(np.abs(diff[diff != 0]))[diff[diff != 0] > 0].sum()
                           - stats.rankdata(np.abs(diff[diff != 0]))[diff[diff != 0] < 0].sum())
                          / stats.rankdata(np.abs(diff[diff != 0])).sum())
    g = {
        "G1_saving_ge_10pct": {"saving": float(saving), "pass": bool(saving >= 0.10)},
        "G2_cost_diff_ci_lb_gt_0": {"mean_cost_diff_b1_minus_darth": float(diff.mean()),
                                    "ci95": ci(diff_bs), "saving_ci95": ci(sav_bs),
                                    "pass": bool(ci(diff_bs)[0] > 0)},
        "G3_wilcoxon_p_lt_0.01": {"statistic": float(w.statistic), "p": float(w.pvalue),
                                  "rank_biserial_b1_minus_darth": rank_biserial,
                                  "pass": bool(w.pvalue < 0.01 and diff.mean() > 0)},
        "G4_recall_noninferiority": {"mean_recall_diff_darth_minus_b1": float((rd - rb).mean()),
                                     "ci95": ci(qual_bs), "pass": bool(ci(qual_bs)[0] >= -0.01)},
    }
    mechanism_ok = all(checks.values())
    c1_pass = mechanism_ok and all(v["pass"] for v in g.values())

    b1_wall = np.where(b.b1_ef == 50, D.plain_ef50_seconds, D.plain_ef53_seconds)
    rep = {
        "checks": checks, "gate": g, "C1_PASS": c1_pass,
        "darth": {"mean_recall_tie_aware": float(rd.mean()), "mean_recall_id": float(D.recall_id.mean()),
                  "recall_tie_aware_distribution": {str(k): int(v) for k, v in D.recall_tie_aware.round(1).value_counts().sort_index().items()},
                  "frac_queries_recall_lt_0.9": float((rd < 0.9).mean()),
                  "distance_computations": q(dc), "frac_stopped_early": stopped,
                  "base_dists_after_decision_mean": float((D.base_dists - D.dists_at_decision)[D.stopped_early == 1].mean()),
                  "predictor_calls": q(D.predictor_calls), "predictor_inferences": q(D.predictor_inferences),
                  "predictor_seconds_per_query": q(D.predictor_seconds),
                  "wall_seconds_per_query": q(D.wall_seconds)},
        "b1": {"mean_recall_tie_aware": float(rb.mean()), "distance_computations": q(bc),
               "ef_share_53": float((b.b1_ef == 53).mean()), "wall_seconds_per_query": q(b1_wall)},
        "paired_cost_diff_b1_minus_darth": q(diff),
        "frac_queries_darth_cheaper": float((diff > 0).mean()),
        "overhead": {"predictor_share_of_darth_wall": float(D.predictor_seconds.sum() / D.wall_seconds.sum()),
                     "darth_wall_over_b1_wall": float(D.wall_seconds.sum() / b1_wall.sum()),
                     "note": "single-threaded, sequential, warm cache; same process, same machine"},
        f"cap_plain_ef{int(pol['ef_cap'])}": {"mean_recall_tie_aware": float(cap.recall_tie_aware.mean()),
                            "distance_computations_mean": float(cap.distance_computations.mean())},
        "runs": {"eval": str(run), "repeat": str(rerun)}, "policy": C["policy_path"],
    }
    od = Path(C["output_dir"]) / f"{C.get('eval_report_prefix', 'exp_c_eval')}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    od.mkdir(parents=True)
    (od / "c1_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    out = D.merge(b, on="query_id")
    out.to_csv(od / "per_query_darth_vs_b1.csv", index=False)
    print(json.dumps({"checks": checks, "gate": g, "C1_PASS": c1_pass}, indent=1))
    print(od)
    return 0


if __name__ == "__main__":
    sys.exit(main())
