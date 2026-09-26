"""Integrity and distribution-shift checks for the Phase 4b confirmation set.
Never reads ground truth, oracle curves or labels.

Usage:
  python python/phase4b_confirmation_checks.py integrity configs/phase4b/confirmation_set.yaml
  python python/phase4b_confirmation_checks.py features  configs/phase4b/confirmation_set.yaml \
      <orig_features_run_s42> <conf_features_run_s42> ... (3 pairs: seeds 42, 43, 44)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    assert (rows[:, 0] == dim).all()
    return rows[:, 1:].copy().view(np.float32)


class MT64:
    def __init__(self, seed):
        self.mt = [0] * 312
        self.mt[0] = seed & 0xFFFFFFFFFFFFFFFF
        for i in range(1, 312):
            self.mt[i] = (6364136223846793005 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 62)) + i) & 0xFFFFFFFFFFFFFFFF
        self.i = 312

    def __call__(self):
        if self.i >= 312:
            for k in range(312):
                x = (self.mt[k] & 0xFFFFFFFF80000000) | (self.mt[(k + 1) % 312] & 0x7FFFFFFF)
                xa = x >> 1
                if x & 1:
                    xa ^= 0xB5026F5AA96619E9
                self.mt[k] = self.mt[(k + 156) % 312] ^ xa
            self.i = 0
        x = self.mt[self.i]
        self.i += 1
        x ^= (x >> 29) & 0x5555555555555555
        x ^= (x << 17) & 0x71D67FFFEDA60000
        x ^= (x << 37) & 0xFFF7EEE000000000
        x ^= x >> 43
        return x & 0xFFFFFFFFFFFFFFFF


def independent_selection(n, k, seed):
    r = MT64(seed)
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        b = i + 1
        th = ((1 << 64) - b) % b
        while True:
            v = r()
            if v >= th:
                j = v % b
                break
        p[i], p[j] = p[j], p[i]
    return sorted(p[:k])


def row_keys(m):
    return {r.tobytes() for r in np.ascontiguousarray(m)}


def integrity(cfg):
    learn = read_fvecs(cfg["source"])
    sub = read_fvecs(cfg["output_fvecs"])
    ids = pd.read_csv(cfg["output_ids"])
    tags = pd.read_csv(cfg["output_tagging"])
    base = read_fvecs("data/sift/sift_base.fvecs")
    query = read_fvecs("data/sift/sift_query.fvecs")
    excl = set()
    for path in cfg.get("exclude_exact_duplicates_of", []):
        excl |= row_keys(read_fvecs(path))
    population, seen = [], set()
    for r, row in enumerate(learn):
        k = row.tobytes()
        if k in excl:
            continue
        if cfg.get("deduplicate_population", False):
            if k in seen:
                continue
            seen.add(k)
        population.append(r)
    pos = independent_selection(len(population), cfg["n_select"], cfg["seed"])
    indep = sorted(population[p] for p in pos)
    sub_keys = [r.tobytes() for r in sub]
    base_keys, query_keys = row_keys(base), row_keys(query)
    rep = {
        "eligible_population_independent": len(population),
        "n_selected": int(len(sub)), "n_expected": cfg["n_select"],
        "source_rows": int(len(learn)),
        "ids_unique": bool(ids["source_row"].is_unique),
        "ids_match_independent_python_fisher_yates": ids["source_row"].tolist() == indep,
        "vectors_equal_source_rows": bool(np.array_equal(sub, learn[ids["source_row"].to_numpy()])),
        "tagging_rows": int(len(tags)), "tagging_all_test": bool((tags["split"] == "test").all()),
        "exact_duplicates_vs_base": int(sum(k in base_keys for k in sub_keys)),
        "exact_duplicates_vs_original_queries": int(sum(k in query_keys for k in sub_keys)),
        "exact_duplicates_within_set": int(len(sub_keys) - len(set(sub_keys))),
    }
    rep["PASS"] = (rep["n_selected"] == rep["n_expected"] and rep["ids_unique"]
                   and rep["ids_match_independent_python_fisher_yates"]
                   and rep["vectors_equal_source_rows"] and rep["tagging_rows"] == rep["n_selected"]
                   and rep["tagging_all_test"] and rep["exact_duplicates_vs_base"] == 0
                   and rep["exact_duplicates_vs_original_queries"] == 0
                   and rep["exact_duplicates_within_set"] == 0)
    def summ(m):
        nrm = np.linalg.norm(m.astype(np.float64), axis=1)
        return {"norm_quantiles": np.percentile(nrm, [5, 25, 50, 75, 95]).round(2).tolist(),
                "mean_coordinate": float(m.mean()), "zero_fraction": float((m == 0).mean()),
                "max_coordinate": float(m.max())}
    rep["shift_raw_vectors"] = {
        "original_queries": summ(query), "confirmation_set": summ(sub),
        "norm_ks": dict(zip(("statistic", "p_value"), map(float, stats.ks_2samp(
            np.linalg.norm(query.astype(np.float64), axis=1),
            np.linalg.norm(sub.astype(np.float64), axis=1))))),
        "per_dim_mean_abs_diff": float(np.abs(query.mean(0) - sub.mean(0)).mean()),
        "per_dim_mean_abs_diff_reference_train_vs_test": ref_dim_diff(query)}
    rep["near_duplicate_diagnostic_descriptive_only"] = nn_report(sub, query)
    return rep


def ref_dim_diff(query):
    sp = pd.read_csv("splits/sift1m_query_split_seed20261001.csv")
    tr = query[(sp["split"] == "train").to_numpy()]
    te = query[(sp["split"] == "test").to_numpy()]
    return float(np.abs(tr.mean(0) - te.mean(0)).mean())


def nn_sq(a, b, exclude_self=False):
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    out = np.empty(len(a))
    bn = (b64 ** 2).sum(1)
    for s0 in range(0, len(a), 500):
        blk = a64[s0:s0 + 500]
        d = (blk ** 2).sum(1)[:, None] + bn[None, :] - 2 * blk @ b64.T
        if exclude_self:
            d[np.arange(len(blk)), np.arange(s0, s0 + len(blk))] = np.inf
        out[s0:s0 + len(blk)] = np.maximum(d.min(1), 0)
    return np.sqrt(out)


def nn_report(sub, query):
    sp = pd.read_csv("splits/sift1m_query_split_seed20261001.csv")
    tr = query[(sp["split"] == "train").to_numpy()]
    te = query[(sp["split"] == "test").to_numpy()]
    q = [1, 5, 25, 50, 75, 95]
    def qs(x):
        return np.percentile(x, q).round(2).tolist()
    return {"quantiles": q,
            "confirmation_to_nearest_original_query": qs(nn_sq(sub, query)),
            "confirmation_to_nearest_train_query": qs(nn_sq(sub, tr)),
            "reference_old_test_to_nearest_train_query": qs(nn_sq(te, tr)),
            "reference_train_to_nearest_other_train_query": qs(nn_sq(tr, tr, exclude_self=True)),
            "confirmation_min_distance_to_any_original_query": float(nn_sq(sub, query).min())}


def features_shift(runs):
    out = {}
    for seed, (orig, conf) in zip((42, 43, 44), zip(runs[0::2], runs[1::2])):
        a = pd.read_csv(Path(orig) / "features.csv")
        b = pd.read_csv(Path(conf) / "features.csv")
        assert len(b) == 2000
        d = {}
        for f in ("knn_dist", "centroid_dist", "score_concentration", "lid",
                  "probe_distance_computations"):
            ks = stats.ks_2samp(a[f], b[f])
            d[f] = {"original_q05_q50_q95": np.percentile(a[f], [5, 50, 95]).round(4).tolist(),
                    "confirmation_q05_q50_q95": np.percentile(b[f], [5, 50, 95]).round(4).tolist(),
                    "ks_statistic": float(ks.statistic), "ks_p": float(ks.pvalue)}
        out[str(seed)] = d
    return out


if __name__ == "__main__":
    stage, cfgp = sys.argv[1], sys.argv[2]
    cfg = yaml.safe_load(Path(cfgp).read_text())
    rep = integrity(cfg) if stage == "integrity" else features_shift(sys.argv[3:])
    print(json.dumps(rep, indent=1))
    sys.exit(0 if (stage != "integrity" or rep["PASS"]) else 1)
