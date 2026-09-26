"""Read-only audit diagnostics for Phase 4, using the training split only.
No model is fitted and no test row is read.

Usage: python python/audit_phase4_train.py configs/phase4_router_sift1m.yaml <train_run>
"""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_lib as rl  # noqa: E402


def lagrange_alloc(rec: np.ndarray, cost: np.ndarray, target: float,
                   extra_cost: float = 0.0) -> dict:
    def pick(lam):
        j = np.argmin(cost - lam * rec, axis=1)
        idx = np.arange(len(rec))
        return rec[idx, j].mean(), cost[idx, j].mean(), j
    lo, hi = 0.0, 1e9
    if pick(hi)[0] < target - 1e-12:
        return {"feasible": False}
    for _ in range(200):
        mid = (lo + hi) / 2
        if pick(mid)[0] >= target - 1e-12:
            hi = mid
        else:
            lo = mid
    r, c, j = pick(hi)
    return {"feasible": True, "mean_recall": float(r),
            "mean_cost": float(c + extra_cost), "option_share":
            np.bincount(j, minlength=rec.shape[1]).tolist()}


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
    assert set(oof["query_id"]) == set(train["query_id"])
    train = train.merge(oof, on=["query_id", "tier_label"], validate="1:1")
    qid = train["query_id"].to_numpy()
    probe = train["probe_distance_computations"].to_numpy(float)
    grid = list(curves.recall.columns)
    R = curves.recall.loc[qid].to_numpy()
    D = curves.dc.loc[qid].to_numpy().astype(float)
    T = [grid.index(e) for e in tier_efs]
    out = {"source_train_run": run.name, "n_train": int(len(train)),
           "test_rows_read": 0, "tiers": tiers}

    cvt = pd.read_csv(run / "cv_candidates.csv")
    prim = cvt[cvt["role"] == "primary_candidate"].copy()
    prim["feasible_recomputed"] = (prim["oof_failure_rate"]
                                   <= prim["matched_fixed_failure_rate"] + 1e-9)
    out["A_selection"] = {
        "table": prim[["name", "score", "oof_failure_rate",
                       "matched_fixed_failure_rate", "feasible",
                       "feasible_recomputed"]].to_dict(orient="records"),
        "chosen_by_rule_recomputed": rl.select_candidate(
            prim.assign(feasible=prim["feasible_recomputed"]), 0.02),
        "best_score_ignoring_filter": prim.loc[prim["score"].idxmax(), "name"],
    }

    curve = rl.fixed_curve(curves, qid, target)
    dt2 = rl.route(train["oof_DT2"], qid, probe, curves, tiers, target)
    R_oof = float(dt2["recall"].mean())
    e53 = grid.index(53)
    vals, cnts = np.unique(R[:, e53], return_counts=True)
    out["B_objective"] = {
        "dt2_oof_mean_recall": R_oof,
        "dt2_oof_failure_rate": float(dt2["failure"].mean()),
        "dt2_oof_mean_total_dc": float(dt2["total_dc"].mean()),
        "dt2_oof_mean_search_dc": float(dt2["search_dc"].mean()),
        "fixed_cost_at_dt2_recall": rl.interp_cost_at_recall(curve, R_oof),
        "fixed_ef53_recall_value_shares": {f"{v:.1f}": float(c / len(qid))
                                           for v, c in zip(vals, cnts)},
        "oracle_10of10_mean_cost": float(D[np.arange(len(qid)),
                                           [grid.index(int(e)) for e in
                                            train["oracle_ef"]]].mean()),
        "note": "Labels encode the cost of 10/10 per query; the matched "
                "comparison is on MEAN recall, where 9/10 counts 0.9.",
    }
    ok = curve[curve["failure_rate"] <= dt2["failure"].mean() + 1e-12]
    out["B_objective"]["fixed_at_matched_failure_rate"] = {
        "ef": int(ok["ef"].iloc[0]), "mean_dc": float(ok["mean_dc"].iloc[0]),
        "failure_rate": float(ok["failure_rate"].iloc[0])}

    targets = sorted({round(R_oof, 6), 0.97, 0.99, 0.999})
    alloc = {}
    for r in targets:
        tier_rec, tier_dc = R[:, T], D[:, T]
        alloc[str(r)] = {
            "fixed_ef_interp": rl.interp_cost_at_recall(curve, r),
            "omniscient_any_grid_ef": lagrange_alloc(R, D, r),
            "omniscient_3_tiers": lagrange_alloc(tier_rec, tier_dc, r),
            "omniscient_3_tiers_plus_probe": lagrange_alloc(
                tier_rec, tier_dc, r, extra_cost=float(probe.mean())),
        }
    out["C_omniscient_allocation"] = alloc
    perfect = rl.route(train["tier_label"], qid, probe, curves, tiers, target)
    out["C_perfect_tier_classification"] = {
        "mean_recall": float(perfect["recall"].mean()),
        "mean_search_dc": float(perfect["search_dc"].mean()),
        "mean_total_dc": float(perfect["total_dc"].mean()),
        "fixed_cost_at_that_recall": rl.interp_cost_at_recall(
            curve, perfect["recall"].mean())}

    cells = []
    for t_true in rl.TIERS:
        for t_pred in rl.TIERS:
            m = (train["tier_label"] == t_true) & (train["oof_DT2"] == t_pred)
            if not m.any():
                continue
            ids = qid[m.to_numpy()]
            r_p, _, d_p = curves.lookup(ids, [tiers[t_pred]] * len(ids))
            r_t, _, d_t = curves.lookup(ids, [tiers[t_true]] * len(ids))
            cells.append({"true": t_true, "pred": t_pred, "n": int(m.sum()),
                          "mean_dc_at_pred": float(d_p.mean()),
                          "extra_dc_vs_true_tier": float((d_p - d_t).mean()),
                          "mean_recall_at_pred": float(r_p.mean()),
                          "recall_lost_vs_true_tier": float((r_t - r_p).mean()),
                          "failure_rate": float((r_p < target - 1e-9).mean())})
    cdf = pd.DataFrame(cells)
    cons = cdf[cdf["true"].map(rl.TIER_INDEX) < cdf["pred"].map(rl.TIER_INDEX)]
    aggr = cdf[cdf["true"].map(rl.TIER_INDEX) > cdf["pred"].map(rl.TIER_INDEX)]
    n = len(train)
    out["D_error_prices"] = {
        "cells": cells,
        "conservative_total_extra_dc_per_query_of_all":
            float((cons["n"] * cons["extra_dc_vs_true_tier"]).sum() / n),
        "aggressive_total_recall_lost_per_query_of_all":
            float((aggr["n"] * aggr["recall_lost_vs_true_tier"]).sum() / n),
        "aggressive_total_dc_saved_per_query_of_all":
            float(-(aggr["n"] * aggr["extra_dc_vs_true_tier"]).sum() / n),
    }

    rows = []
    for t in rl.TIERS:
        m = (train["tier_label"] == t).to_numpy()
        oe = train.loc[m, "oracle_ef"].to_numpy()
        d_t = D[m, T[rl.TIER_INDEX[t]]]
        d_o = D[np.where(m)[0], [grid.index(int(e)) for e in oe]]
        rows.append({"tier": t, "ef": tiers[t], "n": int(m.sum()),
                     "oracle_ef_median": float(np.median(oe)),
                     "oracle_ef_p10_p90": [float(np.quantile(oe, .1)),
                                           float(np.quantile(oe, .9))],
                     "mean_dc_at_tier": float(d_t.mean()),
                     "mean_dc_at_oracle_ef": float(d_o.mean()),
                     "quantisation_overspend_dc": float((d_t - d_o).mean())})
    out["E_within_tier"] = rows

    rng = np.random.default_rng(20261003)
    sim = {}
    for rho in (0.36, 0.46, 0.56, 0.65, 0.8):
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 200000)
        a = np.digitize(z[:, 0], np.quantile(z[:, 0], [1 / 3, 2 / 3]))
        b = np.digitize(z[:, 1], np.quantile(z[:, 1], [1 / 3, 2 / 3]))
        sim[str(rho)] = {"tertile_accuracy": float((a == b).mean()),
                         "two_step_error_rate": float((np.abs(a - b) == 2).mean())}
    out["F_expected_accuracy_from_rho"] = sim
    out["F_observed_oof_accuracy"] = {
        c: float((train[f"oof_{c}"] == train["tier_label"]).mean())
        for c in ("DT2", "DT3", "LR", "DT4")}
    out["F_observed_two_step_errors_DT2"] = float(
        (np.abs(rl.tier_error(train["oof_DT2"], train["tier_label"])) == 2).mean())

    d18 = D[:, T[0]]
    out["G_probe"] = {
        "probe_mean_dc": float(probe.mean()),
        "fixed18_mean_dc": float(d18.mean()),
        "probe_over_ef18_search_median": float(np.median(probe / d18)),
        "probe_equals_ef10_search": bool(np.array_equal(
            probe, D[:, grid.index(10)])),
        "lid_probe_mean_dc": float(train["lid_probe_distance_computations"].mean()),
        "lid_probe_equals_ef20_search_dc": bool(np.array_equal(
            train["lid_probe_distance_computations"].to_numpy(float),
            D[:, grid.index(20)])),
    }

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    od = Path(cfg["output_dir"]) / f"audit_phase4_{ts}"
    od.mkdir(parents=True)
    (od / "audit_train.json").write_text(rl.to_json(out) + "\n")
    print(rl.to_json(out))
    print(od)
    return 0


if __name__ == "__main__":
    sys.exit(main())
