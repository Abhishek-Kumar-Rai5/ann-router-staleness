"""Second, read-only validation of the Phase 4 methodology audit.
TRAINING SPLIT ONLY. No model is fitted, nothing canonical is modified, no
test-split row is read (asserted).

Sections (see docs/phase4_methodology_validation.md):
  V1 seed / variance inventory + query-bootstrap CIs + CV-fold spread
  V2 recall contract of every audit-table row
  V3 probe-cost accounting checks
  V4 fixed-ef baseline provenance (bracketing rows, interpolation)
  V5 independent distance-count check: a pure-Python re-implementation of
     hnswlib v0.8.0 searchKnn (bare-bone path) that reads the saved index file
     directly and counts every distance evaluation; ground truth by NumPy
     brute force. Shares no code with the C++ search / evaluation path.
  V6 reconstructed five-row table

Usage: python python/validate_audit_train.py configs/phase4_router_sift1m.yaml <train_run>
"""

import datetime as dt
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_lib as rl  # noqa: E402

BOOT = 200
BOOT_SEED = 20261004


# ----------------------------------------------------------------------------
# Lagrangian allocation (same method as audit_phase4_train.py, but returns the
# per-query choice so that the recall contract can be inspected).
def lagrange(rec, cost, target, iters=100):
    idx = np.arange(len(rec))

    def pick(lam):
        j = np.argmin(cost - lam * rec, axis=1)
        return rec[idx, j].mean(), j

    lo, hi = 0.0, 1e9
    if pick(hi)[0] < target - 1e-12:
        return None
    for _ in range(iters):
        mid = (lo + hi) / 2
        if pick(mid)[0] >= target - 1e-12:
            hi = mid
        else:
            lo = mid
    return pick(hi)[1]


def contract(rec_chosen, cost_chosen, target):
    return {"mean_recall": float(rec_chosen.mean()),
            "per_query_failure_rate": float((rec_chosen < target - 1e-9).mean()),
            "share_10_of_10": float((rec_chosen >= 1 - 1e-9).mean()),
            "mean_cost": float(cost_chosen.mean())}


# ----------------------------------------------------------------------------
# Independent hnswlib replica (V5).
class HnswFile:
    """Reads an hnswlib v0.8.0 saveIndex() file (layout: hnswalg.h:685-713)."""

    def __init__(self, path):
        with open(path, "rb") as f:
            hdr = f.read(96)
        (self.offset_level0, self.max_elements, self.count, self.size_per_el,
         self.label_offset, self.offset_data, self.maxlevel, self.entry,
         self.maxM, self.maxM0, self.M, self.mult, self.efc) = struct.unpack(
            "<QQQQQQiIQQQdQ", hdr)
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        base = 96
        self.level0 = self.mm[base: base + self.count * self.size_per_el].reshape(
            self.count, self.size_per_el)
        self.size_links = self.maxM * 4 + 4
        pos = base + self.count * self.size_per_el
        self.ll_off = np.zeros(self.count, dtype=np.int64)
        self.ll_size = np.zeros(self.count, dtype=np.int64)
        raw = self.mm
        for i in range(self.count):
            sz = int(raw[pos:pos + 4].view(np.uint32)[0])
            self.ll_off[i] = pos + 4
            self.ll_size[i] = sz
            pos += 4 + sz
        assert pos == len(raw), "file layout mismatch"
        dim = (self.label_offset - self.offset_data) // 4
        self.vec = self.level0[:, self.offset_data:self.label_offset].view(
            np.float32).reshape(self.count, dim)
        self.labels = self.level0[:, self.label_offset:self.label_offset + 8].view(
            np.uint64).reshape(-1)

    def links0(self, i):
        row = self.level0[i]
        n = int(row[0:2].view(np.uint16)[0])          # getListCount: low 16 bits
        return row[4:4 + 4 * n].view(np.uint32)

    def links(self, i, level):
        start = int(self.ll_off[i]) + (level - 1) * self.size_links
        n = int(self.mm[start:start + 2].view(np.uint16)[0])
        return self.mm[start + 4:start + 4 + 4 * n].view(np.uint32)


# libstdc++ heap algorithms (bits/stl_heap.h) with comparator a[0] < b[0]
# (hnswlib CompareByFirst), so tie order matches std::priority_queue exactly.
def _push_heap_hole(h, hole, top, value):
    parent = (hole - 1) // 2
    while hole > top and h[parent][0] < value[0]:
        h[hole] = h[parent]
        hole = parent
        parent = (hole - 1) // 2
    h[hole] = value


def pq_push(h, value):
    h.append(value)
    _push_heap_hole(h, len(h) - 1, 0, value)


def pq_pop(h):
    n = len(h)
    if n > 1:
        value = h[n - 1]
        h[n - 1] = h[0]
        length, hole, top, second = n - 1, 0, 0, 0
        while second < (length - 1) // 2:
            second = 2 * (second + 1)
            if h[second][0] < h[second - 1][0]:
                second -= 1
            h[hole] = h[second]
            hole = second
        if (length & 1) == 0 and second == (length - 2) // 2:
            second = 2 * (second + 1)
            h[hole] = h[second - 1]
            hole = second - 1
        _push_heap_hole(h, hole, top, value)
    h.pop()


def replica_search(idx: HnswFile, q: np.ndarray, k: int, ef: int):
    """hnswlib searchKnn (hnswalg.h:1271) + searchBaseLayerST<true>
    (hnswalg.h:309-440), counting every distance evaluation."""
    count = 0
    qd = q.astype(np.float64)

    def dist(i):
        nonlocal count
        count += 1
        d = idx.vec[i].astype(np.float64) - qd
        return float(np.dot(d, d))      # exact for integer-valued SIFT

    cur = int(idx.entry)
    curdist = dist(cur)
    for level in range(idx.maxlevel, 0, -1):
        changed = True
        while changed:
            changed = False
            for c in idx.links(cur, level):
                d = dist(int(c))
                if d < curdist:
                    curdist, cur, changed = d, int(c), True
    ef_eff = max(ef, k)
    visited = {cur}
    d0 = dist(cur)
    top, cand = [], []
    pq_push(top, (d0, cur))
    pq_push(cand, (-d0, cur))
    lower = d0
    while cand:
        cdist = -cand[0][0]
        if cdist > lower:               # bare_bone_search stop rule
            break
        node = cand[0][1]
        pq_pop(cand)
        for c in idx.links0(node):
            c = int(c)
            if c in visited:
                continue
            visited.add(c)
            d = dist(c)
            if len(top) < ef_eff or lower > d:
                pq_push(cand, (-d, c))
                pq_push(top, (d, c))
                while len(top) > ef_eff:
                    pq_pop(top)
                if top:
                    lower = top[0][0]
    while len(top) > k:
        pq_pop(top)
    res = sorted(top)
    return [int(idx.labels[i]) for _, i in res], [d for d, _ in res], count


def main() -> int:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    run = Path(sys.argv[2])
    inp, target = cfg["inputs"], cfg["target_recall"]
    tiers = json.loads(Path(inp["tiers"]).read_text())["fixed_ef_baselines"]
    tier_efs = [tiers["low"], tiers["med"], tiers["high"]]
    train = rl.load_split_frame(Path(inp["features_run"]) / "features.csv",
                                inp["strata"], "train")
    curves = rl.load_curves(Path(inp["oracle_run"]) / "oracle_curves.csv", "train")
    assert set(train["split"]) == {"train"} and len(train) == 8000
    oof = pd.read_csv(run / "oof_predictions.csv")
    train = train.merge(oof, on=["query_id", "tier_label"], validate="1:1")
    qid = train["query_id"].to_numpy()
    probe = train["probe_distance_computations"].to_numpy(float)
    grid = list(curves.recall.columns)
    R = curves.recall.loc[qid].to_numpy()
    D = curves.dc.loc[qid].to_numpy().astype(float)
    T = [grid.index(e) for e in tier_efs]
    out = {"n_train": len(train), "test_rows_read": 0}

    # ---------------- V3/V6 core quantities (full train set) ----------------
    dt2 = rl.route(train["oof_DT2"], qid, probe, curves, tiers, target)
    Rt = float(dt2["recall"].mean())
    curve = rl.fixed_curve(curves, qid, target)
    rows = {}
    rows["fixed_ef_interp"] = {"cost": rl.interp_cost_at_recall(curve, Rt)}
    j_grid = lagrange(R, D, Rt)
    j_t = lagrange(R[:, T], D[:, T], Rt)
    ii = np.arange(len(qid))
    rows["omniscient_any_ef"] = contract(R[ii, j_grid], D[ii, j_grid], target)
    rows["omniscient_3_tiers"] = contract(R[ii, [T[j] for j in j_t]],
                                          D[ii, [T[j] for j in j_t]], target)
    r3 = rows["omniscient_3_tiers"]
    rows["omniscient_3_tiers_plus_probe"] = {
        **r3, "mean_cost": r3["mean_cost"] + float(probe.mean())}
    rows["dt2_oof"] = contract(dt2["recall"].to_numpy(),
                               dt2["total_dc"].to_numpy(), target)
    rows["dt2_oof"]["mean_search_cost"] = float(dt2["search_dc"].mean())
    # fixed: realised contract at the bracketing grid ef values (V4)
    c = curve.sort_values("ef").reset_index(drop=True)
    i_hi = int(np.where(c["mean_recall"] >= Rt - 1e-12)[0][0])
    lo_row, hi_row = c.loc[i_hi - 1], c.loc[i_hi]
    w = (Rt - lo_row["mean_recall"]) / (hi_row["mean_recall"] - lo_row["mean_recall"])
    out["V4_fixed_baseline"] = {
        "target_mean_recall": Rt,
        "bracket_low": {"ef": int(lo_row["ef"]), "mean_recall": float(lo_row["mean_recall"]),
                        "mean_dc": float(lo_row["mean_dc"]),
                        "failure_rate": float(lo_row["failure_rate"])},
        "bracket_high": {"ef": int(hi_row["ef"]), "mean_recall": float(hi_row["mean_recall"]),
                         "mean_dc": float(hi_row["mean_dc"]),
                         "failure_rate": float(hi_row["failure_rate"])},
        "weight_on_high": float(w),
        "interpolated_cost": rows["fixed_ef_interp"]["cost"],
        "direct_cost_at_cheapest_reaching_ef": float(hi_row["mean_dc"]),
        "max_interpolation_error_bound": float(hi_row["mean_dc"] - lo_row["mean_dc"]),
        "interpolated_failure_rate_equiv": float(
            lo_row["failure_rate"] + w * (hi_row["failure_rate"] - lo_row["failure_rate"])),
    }
    rows["fixed_ef_interp"].update({
        "mean_recall": Rt, "per_query_failure_rate":
            out["V4_fixed_baseline"]["interpolated_failure_rate_equiv"]})
    out["V6_rows"] = rows

    # ---------------- V3 probe accounting checks ----------------------------
    out["V3_probe"] = {
        "dt2_total_minus_search_equals_probe_per_query": bool(np.array_equal(
            dt2["total_dc"].to_numpy() - dt2["search_dc"].to_numpy(), probe)),
        "probe_equals_ef10_curve_dc": bool(np.array_equal(probe, D[:, grid.index(10)])),
        "mean_probe": float(probe.mean()),
        "plus_probe_row_equals_restricted_plus_mean_probe": True,
        "note": "The probe is identical for every option of a query, so adding "
                "it does not change the Lagrangian allocation; adding the mean "
                "probe to the mean cost is exact.",
    }

    # ---------------- V1 variance: query bootstrap + CV folds ---------------
    rng = np.random.default_rng(BOOT_SEED)
    bs = {k: [] for k in ("fixed", "omni_any", "omni_3", "omni_3_probe", "dt2",
                          "gap_fixed_minus_omni3probe", "gap_dt2_minus_fixed",
                          "gap_fixed_minus_omni_any")}
    for _ in range(BOOT):
        s = rng.integers(0, len(qid), len(qid))
        Rs, Ds, ps = R[s], D[s], probe[s]
        dt2s = dt2.iloc[s]
        rt = float(dt2s["recall"].mean())
        cs = pd.DataFrame({"ef": grid, "mean_recall": Rs.mean(0), "mean_dc": Ds.mean(0)})
        f = rl.interp_cost_at_recall(cs, rt)
        jg = lagrange(Rs, Ds, rt, iters=60)
        jt = lagrange(Rs[:, T], Ds[:, T], rt, iters=60)
        k_ = np.arange(len(s))
        oa = Ds[k_, jg].mean()
        o3 = Ds[k_, [T[j] for j in jt]].mean()
        bs["fixed"].append(f)
        bs["omni_any"].append(oa)
        bs["omni_3"].append(o3)
        bs["omni_3_probe"].append(o3 + ps.mean())
        bs["dt2"].append(dt2s["total_dc"].mean())
        bs["gap_fixed_minus_omni3probe"].append(f - (o3 + ps.mean()))
        bs["gap_dt2_minus_fixed"].append(dt2s["total_dc"].mean() - f)
        bs["gap_fixed_minus_omni_any"].append(f - oa)
    out["V1_query_bootstrap"] = {
        k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
            "ci95": [float(x) for x in np.percentile(v, [2.5, 97.5])]}
        for k, v in bs.items()}
    out["V1_query_bootstrap"]["resamples"] = BOOT
    out["V1_query_bootstrap"]["note"] = (
        "Query-sampling variance only (operating point re-matched per resample). "
        "Does NOT capture index-construction (HNSW seed) or CV-seed variance.")
    skf = StratifiedKFold(cfg["router"]["cv_folds"], shuffle=True,
                          random_state=cfg["router"]["seed"])
    folds = []
    for f_i, (_, va) in enumerate(skf.split(train[rl.CORE_FEATURES].to_numpy(),
                                            train["tier_label"].to_numpy())):
        sub = dt2.iloc[va]
        cf = rl.fixed_curve(curves, qid[va], target)
        r_f = float(sub["recall"].mean())
        fc = rl.interp_cost_at_recall(cf, r_f)
        folds.append({"fold": f_i, "n": int(len(va)), "mean_recall": r_f,
                      "dt2_cost": float(sub["total_dc"].mean()),
                      "fixed_cost": fc, "ratio_fixed_over_dt2":
                          float(fc / sub["total_dc"].mean())})
    out["V1_cv_folds"] = folds
    out["V1_seed_inventory"] = {
        "hnsw_level_seed_values_with_built_indexes": [42],
        "split_seeds": [20261001], "cv_seeds": [20261002],
        "independent_index_replicates": 0,
        "design_requirement": "§13: core conditions repeated across 2-3 seeds",
    }

    # ---------------- V5 independent distance-count check -------------------
    meta = json.loads((Path(inp["oracle_run"]) / "metadata.json").read_text())
    idx_path = meta["index"]["cache_path"]
    idx = HnswFile(idx_path)
    picks = {}
    for name, (t, p) in {"low_correct": ("low", "low"),
                         "low_to_high_conservative": ("low", "high"),
                         "high_to_low_aggressive": ("high", "low"),
                         "med_correct": ("med", "med"),
                         "high_correct": ("high", "high")}.items():
        m = train[(train["tier_label"] == t) & (train["oof_DT2"] == p)]
        picks[name] = int(m["query_id"].min())
    fcfg = yaml.safe_load((Path(inp["features_run"]) / "config.yaml").read_text())
    raw_q = np.fromfile(fcfg["dataset"]["queries"], dtype=np.float32).reshape(
        -1, 129)[:, 1:]
    base = idx.vec  # vectors as stored in the index file (label == internal id)
    assert np.array_equal(idx.labels, np.arange(idx.count, dtype=np.uint64))
    checks = []
    for name, q in picks.items():
        qv = raw_q[q]
        diff = base.astype(np.float64) - qv.astype(np.float64)
        gt_d = np.einsum("ij,ij->i", diff, diff)
        kth = np.partition(gt_d, 9)[9]                      # NumPy ground truth
        row = train[train["query_id"] == q].iloc[0]
        rec = {"query_id": q, "case": name, "true_tier": row["tier_label"],
               "oof_tier": row["oof_DT2"], "results": {}}
        for ef in (10, *tier_efs):
            labels, dists, cnt = replica_search(idx, qv, 10, ef)
            tie_rec = float(np.mean(np.array(dists) <= kth * (1 + 1e-6)))
            rec["results"][str(ef)] = {
                "replica_dc": cnt,
                "pipeline_dc": int(curves.dc.loc[q, ef]),
                "replica_tie_aware_recall": tie_rec,
                "pipeline_tie_aware_recall": float(curves.recall.loc[q, ef]),
                "match": cnt == int(curves.dc.loc[q, ef])
                and abs(tie_rec - float(curves.recall.loc[q, ef])) < 1e-9}
        pr = rec["results"]["10"]["replica_dc"]
        ef_route = tiers[row["oof_DT2"]]
        rec["probe_replica_dc"] = pr
        rec["probe_pipeline_dc"] = int(row["probe_distance_computations"])
        rec["policy_total_replica"] = pr + rec["results"][str(ef_route)]["replica_dc"]
        rec["policy_total_pipeline"] = float(
            dt2.loc[dt2["query_id"] == q, "total_dc"].iloc[0])
        rec["policy_match"] = rec["policy_total_replica"] == rec["policy_total_pipeline"]
        checks.append(rec)
    out["V5_independent_check"] = {
        "index_file": idx_path,
        "index_header": {"count": int(idx.count), "maxlevel": int(idx.maxlevel),
                         "entry": int(idx.entry), "M": int(idx.M),
                         "maxM0": int(idx.maxM0)},
        "queries": checks,
        "all_match": all(r["policy_match"] and all(v["match"] for v in
                                                   r["results"].values())
                         for r in checks)}

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    od = Path(cfg["output_dir"]) / f"audit_validation_{ts}"
    od.mkdir(parents=True)
    (od / "validation_train.json").write_text(rl.to_json(out) + "\n")
    print(rl.to_json(out))
    print(od)
    return 0


if __name__ == "__main__":
    sys.exit(main())
