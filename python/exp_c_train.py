"""Experiment C (design_doc.md Addendum C1, C1.9): train the DARTH recall
predictor on the S0 training traces and freeze the complete policy.

Independent re-implementation (THIRD_PARTY.md). Frozen settings only:
LGBMRegressor(objective="regression", n_estimators=100, random_state=42,
verbose=-1), all other parameters default; all observation rows; intervals
from the published heuristic (ipi = int(dists_Rt / 2), mpi = int(dists_Rt / 10)).

Usage: python python/exp_c_train.py configs/exp_c/darth_s0.yaml <trace_run_dir>
"""

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

FEATURES = ["step", "dists", "inserts", "first_nn_dist", "nn_dist", "furthest_dist",
            "avg_dist", "variance", "percentile_25", "percentile_50", "percentile_75"]
TARGET_RECALL = 0.95
SEED = 42
INDEX_SHA256 = "4282e2e2"  # prefix recorded in notes (S0 rebuild check); full hash computed below


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fit(X, y):
    m = lgb.LGBMRegressor(objective="regression", n_estimators=100, random_state=SEED, verbose=-1)
    m.fit(X, y)
    return m


def main() -> int:
    cfg_path, trace_run = Path(sys.argv[1]), Path(sys.argv[2])
    C = yaml.safe_load(cfg_path.read_text())
    tmeta = json.loads((trace_run / "metadata.json").read_text())["result"]
    assert tmeta["columns"] == ["query_id", *FEATURES, "recall_id"], "trace column order != C1.9"
    trace = Path(tmeta["trace_file"])
    raw = np.fromfile(trace, dtype=np.float64).reshape(-1, len(FEATURES) + 2)
    assert len(raw) == tmeta["observations"]
    qid = raw[:, 0].astype(np.int64)

    # ---- leakage check: every observation comes from a TRAIN-split query -----
    split = pd.read_csv(C["split_path"])
    train_ids = set(split.loc[split.split == "train", "query_id"])
    assert len(train_ids) == 8000
    if not set(np.unique(qid)) <= train_ids or len(np.unique(qid)) != 8000:
        raise SystemExit("STOP: trace contains non-training queries or misses some")

    X = pd.DataFrame(raw[:, 1:-1], columns=FEATURES)
    y = raw[:, -1]
    if not np.isfinite(raw).all() or y.min() < 0 or y.max() > 1:
        raise SystemExit("STOP: non-finite values or labels outside [0, 1]")

    # ---- published interval heuristic --------------------------------------------
    df = pd.DataFrame({"qid": qid, "dists": X["dists"].to_numpy(), "r": y})
    reached = df[df.r >= TARGET_RECALL].groupby("qid", sort=False)["dists"].min()
    dists_rt = float(reached.mean())
    ipi, mpi = int(dists_rt / 2), int(dists_rt / 10)

    # ---- trace statistics ------------------------------------------------------------
    per_q = df.groupby("qid").agg(n=("r", "size"), final_r=("r", "last"), max_d=("dists", "max"))
    tq = pd.read_csv(trace_run / "trace_queries.csv")
    stats = {
        "queries": int(len(per_q)), "observations": int(len(df)),
        "observations_per_query": {k: float(v) for k, v in per_q.n.describe().items()},
        "distance_computations_per_query (total, incl. upper layers)":
            {k: float(v) for k, v in tq.distance_computations.describe().items()},
        "queries_with_zero_observations": int(8000 - len(per_q)),
        "final_recall_id_mean": float(tq.final_recall_id.mean()),
        "final_recall_tie_aware_mean": float(tq.final_recall_tie_aware.mean()),
        "queries_not_reaching_Rt_in_trace": int(8000 - len(reached)),
        "feature_ranges": {f: {"min": float(X[f].min()), "mean": float(X[f].mean()),
                               "max": float(X[f].max())} for f in FEATURES},
        "label_mean": float(y.mean()),
    }

    # ---- train (frozen settings) twice: determinism check -----------------------------
    # Optional per-seed keys (Experiment E replication); defaults = seed-42 C1.
    model_path = Path(C.get("model_out", "derived/exp_c/darth_lgbm_s0.txt"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = dt.datetime.now()
    m = fit(X, y)
    train_s = (dt.datetime.now() - t0).total_seconds()
    m.booster_.save_model(str(model_path))
    m2 = fit(X, y)
    rep_path = Path(C["trace_dir"]) / "darth_lgbm_s0_refit.txt"
    m2.booster_.save_model(str(rep_path))
    deterministic = sha(model_path) == sha(rep_path)
    booster = lgb.Booster(model_file=str(model_path))
    assert booster.feature_name() == FEATURES, "model feature order != C1.9"
    params = {k: v for k, v in m.get_params().items()}

    policy = {
        "frozen": "design_doc.md Addendum C1 / C1.9; immutable before evaluation",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model_path": str(model_path), "model_sha256": sha(model_path),
        "feature_order": FEATURES, "k": int(C["k"]), "ef_cap": int(C["ef_cap"]),
        "target_recall": TARGET_RECALL, "ipi": ipi, "mpi": mpi, "dists_Rt": dists_rt,
        "interval_rule": "ipi = int(dists_Rt/2), mpi = int(dists_Rt/10); "
                         "pi = int(mpi + (ipi - mpi)*(Rt - Rp))",
        "lightgbm_version": lgb.__version__, "lightgbm_params": params,
        "seeds": {"lightgbm_random_state": SEED, "bootstrap": 20261002, "b1_hash_seed": 20261006},
        "training": {"split_path": C["split_path"], "split_sha256": sha(C["split_path"]),
                     "split_value": "train", "query_ids_sha256": hashlib.sha256(
                         ",".join(map(str, sorted(train_ids))).encode()).hexdigest(),
                     "trace_run": str(trace_run), "trace_sha256": sha(trace),
                     "observations": int(len(df)), "train_seconds": train_s,
                     "refit_identical_model": deterministic},
        "index": {"path": C["index_path"], "sha256": sha(C["index_path"])},
        "ground_truth": {"prefix": C["gt_prefix"], "ivecs_sha256": sha(C["gt_prefix"] + ".ivecs")},
        "queries": {"path": C["queries"], "sha256": sha(C["queries"])},
    }
    if not policy["index"]["sha256"].startswith(C.get("expected_index_sha256_prefix", INDEX_SHA256)):
        raise SystemExit("STOP: S0 index hash differs from the recorded S0 hash for this seed")
    pol_path = Path(C["policy_path"])
    pol_path.write_text(json.dumps(policy, indent=1) + "\n")
    rep = Path(C["output_dir"]) / f"exp_c_train_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    rep.mkdir(parents=True)
    (rep / "train_report.json").write_text(json.dumps({"policy": policy, "trace_stats": stats}, indent=1) + "\n")
    print(json.dumps({"ipi": ipi, "mpi": mpi, "dists_Rt": dists_rt, "model_sha256": policy["model_sha256"],
                      "deterministic_refit": deterministic, "train_seconds": train_s,
                      "observations": len(df)}, indent=1))
    print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
