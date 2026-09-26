"""Independent NumPy check of the confirmation-set ground truth. It shares no code
with the C++ ground truth.

Usage: python python/phase4b_verify_confirm_gt.py <oracle_run_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml


def rv(p, dtype):
    raw = np.fromfile(p, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].copy().view(dtype)


run = Path(sys.argv[1])
meta = json.loads((run / "metadata.json").read_text())
cfg = yaml.safe_load((run / "config.yaml").read_text())
gtp = meta["ground_truth"]["cache_path"]
ids, dists = rv(gtp + ".ivecs", np.int32), rv(gtp + ".fvecs", np.float32)
Q = rv(cfg["dataset"]["queries"], np.float32).astype(np.float64)
B = rv(cfg["dataset"]["base"], np.float32)
bn = (B.astype(np.float64) ** 2).sum(1)
qn = (Q ** 2).sum(1)
best_d = np.full((len(Q), 100), np.inf)
best_i = np.zeros((len(Q), 100), dtype=np.int64)
for s in range(0, len(B), 100000):
    blk = B[s:s + 100000].astype(np.float64)
    d = qn[:, None] + bn[None, s:s + 100000] - 2 * Q @ blk.T
    cat_d = np.concatenate([best_d, d], 1)
    cat_i = np.concatenate([best_i, np.arange(s, s + len(blk))[None, :].repeat(len(Q), 0)], 1)
    o = np.lexsort((cat_i, cat_d), axis=1)[:, :100]
    best_d = np.take_along_axis(cat_d, o, 1)
    best_i = np.take_along_axis(cat_i, o, 1)
prof_equal = bool(np.array_equal(np.round(best_d), dists.astype(np.float64)))
id_equal_rows = int((best_i == ids).all(1).sum())
bad = 0
for q in range(len(Q)):
    a, b = set(best_i[q, :10]), set(ids[q, :10])
    if a != b:
        kth = best_d[q, 9]
        for x in a ^ b:
            if abs(((B[x].astype(np.float64) - Q[q]) ** 2).sum() - kth) > 1e-6:
                bad += 1
                break
rep = {"gt_cache": gtp, "queries": len(Q), "top100_distance_profile_identical": prof_equal,
       "rows_with_identical_top100_ids": id_equal_rows,
       "rows_whose_top10_set_differs_beyond_ties": bad,
       "verdict": "PASS" if prof_equal and bad == 0 else "FAIL"}
print(json.dumps(rep, indent=1))
sys.exit(0 if rep["verdict"] == "PASS" else 1)
