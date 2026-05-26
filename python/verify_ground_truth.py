"""Independently verify our brute-force ground truth for SIFT1M.

Two independent checks, neither sharing code with the C++ implementation:
  1. Compare our top-k ids against the official TEXMEX sift_groundtruth.ivecs.
  2. For every row where the id lists differ, recompute the distances of both
     candidate lists in NumPy (float64) and confirm the disagreement is only
     a tie (equal distances) rather than a wrong answer.

Usage:
  python python/verify_ground_truth.py --ours data/cache/gt_S0_<key> \
      --official data/sift/sift_groundtruth.ivecs \
      --base data/sift/sift_base.fvecs --queries data/sift/sift_query.fvecs \
      [--out results/<exp>/gt_verification.json]
"""

import argparse
import json
import sys

import numpy as np


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    assert (rows[:, 0] == dim).all(), f"inconsistent dims in {path}"
    return rows[:, 1:].copy().view(dtype)


def sq_dists(base, query, ids):
    diff = base[ids].astype(np.float64) - query.astype(np.float64)
    return (diff * diff).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="cache prefix (no extension)")
    ap.add_argument("--official", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    ours_ids = read_vecs(args.ours + ".ivecs", np.int32)
    ours_d = read_vecs(args.ours + ".fvecs", np.float32)
    official = read_vecs(args.official, np.int32)
    base = np.memmap(args.base, dtype=np.float32, mode="r").reshape(-1, 129)[:, 1:]
    queries = read_vecs(args.queries, np.float32)
    assert ours_ids.shape == official.shape, (ours_ids.shape, official.shape)
    nq, depth = ours_ids.shape

    report = {"num_queries": nq, "depth": depth}
    for k in (1, 10, 100):
        if k > depth:
            continue
        exact_rows = int((ours_ids[:, :k] == official[:, :k]).all(axis=1).sum())
        set_overlap = np.mean([
            len(np.intersect1d(ours_ids[q, :k], official[q, :k])) / k
            for q in range(nq)
        ])
        report[f"top{k}_rows_identical_order"] = exact_rows
        report[f"top{k}_mean_set_overlap"] = float(set_overlap)

    # Tie analysis on every row whose full id list differs.
    differing = np.where((ours_ids != official).any(axis=1))[0]
    max_abs_err_ours = 0.0
    non_tie_rows = []
    for q in differing:
        d_ours = sq_dists(base, queries[q], ours_ids[q])
        d_off = sq_dists(base, queries[q], official[q])
        max_abs_err_ours = max(max_abs_err_ours,
                               float(np.abs(d_ours - ours_d[q]).max()))
        # Same sorted distance profile => the lists differ only by permuting
        # or swapping equidistant points.
        if not np.allclose(np.sort(d_ours), np.sort(d_off), rtol=0, atol=1e-6):
            non_tie_rows.append(int(q))
    # Our stored distances must equal float64 recomputation on a sample too.
    for q in range(0, nq, max(1, nq // 200)):
        d = sq_dists(base, queries[q], ours_ids[q])
        max_abs_err_ours = max(max_abs_err_ours, float(np.abs(d - ours_d[q]).max()))

    report["rows_with_any_id_difference"] = int(len(differing))
    report["rows_differing_beyond_ties"] = len(non_tie_rows)
    report["non_tie_row_examples"] = non_tie_rows[:20]
    report["max_abs_error_stored_vs_float64_distance"] = max_abs_err_ours
    report["verdict"] = "PASS" if not non_tie_rows else "FAIL"

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
