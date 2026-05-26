"""Phase 4b ex-ante feasibility report — TRAINING SPLIT ONLY (frozen contract,
docs/phase4_bias_audit.md §10 "Feasibility").

Per index seed and operating point: perfect-information mean cost over (a) the
full ef grid and (b) the frozen candidate ladder, both with the primary fully
additive probe accounting (probe result = ef 10 answer at probe cost only),
versus B1 = train-calibrated two-ef mix reaching the target exactly (expected
cost on train). Classification per the frozen rule:
  full-grid saving < 10%                  -> FAIL_NO_HEADROOM at that level
  ladder < 10% while full grid >= 10%     -> STOP_FOR_REVIEW (candidate-set defect)
  otherwise                               -> OK
No level is discarded. No model is fitted. No evaluation-set row is read.

Usage: python python/phase4b_feasibility.py <oracle_run_seed42> <features_run_seed42> \
          <oracle_run_seed43> <features_run_seed43> <oracle_run_seed44> <features_run_seed44>
"""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redesign_evidence_train as RE  # noqa: E402
import router_lib as rl  # noqa: E402
import validate_audit_train as V  # noqa: E402

LADDER = [20, 40, 83, 164, 327, 647, 1343, 2661]   # + "probe" (frozen)
S_LEVELS = [0.90, 0.95, 0.99]
R_LEVELS = [0.95, 0.97, 0.99]
TARGET = 0.95       # per-query: tie-aware recall@10 == 1.0 (>= 0.95 at k=10)


def two_ef_mix(level_by_ef: np.ndarray, cost_by_ef: np.ndarray, grid, target):
    """Expected cost of the fixed two-ef mix reaching `target` exactly."""
    ok = np.where(level_by_ef >= target - 1e-12)[0]
    j = int(ok[0])
    if j == 0 or level_by_ef[j] == target:
        return {"ef_lo": grid[j], "ef_hi": grid[j], "w_hi": 1.0,
                "cost": float(cost_by_ef[j])}
    lo, hi = j - 1, j
    w = (target - level_by_ef[lo]) / (level_by_ef[hi] - level_by_ef[lo])
    return {"ef_lo": grid[lo], "ef_hi": grid[hi], "w_hi": float(w),
            "cost": float(cost_by_ef[lo] + w * (cost_by_ef[hi] - cost_by_ef[lo]))}


def classify(full_saving, ladder_saving):
    if full_saving < 0.10:
        return "FAIL_NO_HEADROOM"
    if ladder_saving < 0.10:
        return "STOP_FOR_REVIEW"
    return "OK"


def main() -> int:
    runs = sys.argv[1:]
    assert len(runs) == 6
    out = {"test_rows_read": 0, "ladder": ["probe"] + LADDER, "seeds": {}}
    for seed, (orun, frun) in zip((42, 43, 44), zip(runs[0::2], runs[1::2])):
        ometa = json.loads((Path(orun) / "metadata.json").read_text())
        fmeta = json.loads((Path(frun) / "metadata.json").read_text())
        assert ometa["seeds"]["hnsw_level_seed"] == seed == fmeta["seeds"]["hnsw_level_seed"]
        feats = pd.read_csv(Path(frun) / "features.csv")
        feats = feats[feats["split"] == "train"].sort_values("query_id")
        curves = rl.load_curves(Path(orun) / "oracle_curves.csv", "train")
        q = feats["query_id"].to_numpy()
        assert len(q) == 8000 and set(curves.recall.index) == set(q)
        grid = [int(e) for e in curves.recall.columns]
        R = curves.recall.loc[q].to_numpy()
        D = curves.dc.loc[q].to_numpy().astype(float)
        S = R >= TARGET - 1e-9
        probe = feats["probe_distance_computations"].to_numpy(float)
        assert np.array_equal(probe, D[:, grid.index(10)]), "probe != ef10 search"

        def opts(cands):
            cost = np.stack([probe if c == "probe" else probe + D[:, grid.index(c)]
                             for c in cands], 1)
            succ = np.stack([S[:, grid.index(10 if c == "probe" else c)] for c in cands], 1)
            rec = np.stack([R[:, grid.index(10 if c == "probe" else c)] for c in cands], 1)
            return cost, succ, rec

        full = ["probe"] + [e for e in grid if e > 10]
        ladder = ["probe"] + LADDER
        res = {"per_query": {}, "mean_recall": {}}
        succ_rate, mean_rec, mean_dc = S.mean(0), R.mean(0), D.mean(0)
        for s in S_LEVELS:
            b1 = two_ef_mix(succ_rate, mean_dc, grid, s)
            f = RE.contract_oracle(*opts(full)[:2], s)
            l_ = RE.contract_oracle(*opts(ladder)[:2], s)
            fs, ls = 1 - f["mean_cost"] / b1["cost"], 1 - l_["mean_cost"] / b1["cost"]
            res["per_query"][str(s)] = {
                "B1_mix_train": b1, "full_grid_oracle_cost": f["mean_cost"],
                "ladder_oracle_cost": l_["mean_cost"], "full_grid_saving": fs,
                "ladder_saving": ls, "classification": classify(fs, ls)}
        for r in R_LEVELS:
            b1 = two_ef_mix(mean_rec, mean_dc, grid, r)
            vals = {}
            for name, cands in (("full", full), ("ladder", ladder)):
                cost, _, rec = opts(cands)
                j = V.lagrange(rec, cost, r)
                vals[name] = float(cost[np.arange(len(q)), j].mean())
            fs, ls = 1 - vals["full"] / b1["cost"], 1 - vals["ladder"] / b1["cost"]
            res["mean_recall"][str(r)] = {
                "B1_mix_train": b1, "full_grid_oracle_cost": vals["full"],
                "ladder_oracle_cost": vals["ladder"], "full_grid_saving": fs,
                "ladder_saving": ls, "classification": classify(fs, ls)}
        res["oracle_run"], res["features_run"] = orun, frun
        out["seeds"][str(seed)] = res

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    od = Path("results") / f"phase4b_feasibility_{ts}"
    od.mkdir(parents=True)
    (od / "feasibility_train.json").write_text(rl.to_json(out) + "\n")
    for seed, res in out["seeds"].items():
        for fam, levels in (("per_query", res["per_query"]), ("mean_recall", res["mean_recall"])):
            for lvl, v in levels.items():
                b = v["B1_mix_train"]
                print(f"seed {seed} {fam:11s} {lvl}: B1 mix ef {b['ef_lo']}/{b['ef_hi']} "
                      f"w={b['w_hi']:.3f} cost {b['cost']:.0f} | full {v['full_grid_oracle_cost']:.0f} "
                      f"({v['full_grid_saving']:+.1%}) | ladder {v['ladder_oracle_cost']:.0f} "
                      f"({v['ladder_saving']:+.1%}) -> {v['classification']}")
    print(od)
    return 0


if __name__ == "__main__":
    sys.exit(main())
