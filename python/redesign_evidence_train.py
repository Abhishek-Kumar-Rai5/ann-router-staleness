"""Training-split evidence for docs/phase4_redesign.md. Read-only; no model is
fitted and no test row is read.

Usage: python python/redesign_evidence_train.py configs/phase4_router_sift1m.yaml
"""

import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_lib as rl  # noqa: E402
import validate_audit_train as V  # noqa: E402

S_GRID = [0.80, 0.90, 0.95, 0.98, 0.99]
R_GRID = [0.95, 0.97, 0.99]


def stable_first(success: np.ndarray) -> np.ndarray:
    n, m = success.shape
    out = np.full(n, -1)
    ok = success[:, -1].copy()
    idx = np.where(ok, m - 1, -1)
    for j in range(m - 2, -1, -1):
        ok &= success[:, j]
        idx = np.where(ok, j, idx)
    out[:] = idx
    return out


def contract_oracle(cost: np.ndarray, success: np.ndarray, s_star: float):
    n = len(cost)
    first = stable_first(success)
    achievable = first >= 0
    fail_cost = cost.min(axis=1)
    succ_cost = np.where(achievable, cost[np.arange(n), np.maximum(first, 0)], np.inf)
    need = math.ceil(s_star * n - 1e-9)
    if achievable.sum() < need:
        return {"feasible": False, "achievable_rate": float(achievable.mean())}
    extra = succ_cost - fail_cost
    order = np.argsort(extra)
    chosen = np.zeros(n, bool)
    chosen[order[:need]] = True
    total = np.where(chosen, succ_cost, fail_cost)
    return {"feasible": True, "mean_cost": float(total.mean()),
            "success_rate": float(chosen.mean()),
            "achievable_rate": float(achievable.mean())}


def main() -> int:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    inp = cfg["inputs"]
    target = cfg["target_recall"]
    train = rl.load_split_frame(Path(inp["features_run"]) / "features.csv",
                                inp["strata"], "train")
    assert set(train["split"]) == {"train"} and len(train) == 8000
    curves = rl.load_curves(Path(inp["oracle_run"]) / "oracle_curves.csv", "train")
    q = train["query_id"].to_numpy()
    grid = [int(e) for e in curves.recall.columns]
    Rm = curves.recall.loc[q].to_numpy()
    Dm = curves.dc.loc[q].to_numpy().astype(float)
    Sm = Rm >= target - 1e-9
    probe = train["probe_distance_computations"].to_numpy(float)
    assert np.array_equal(probe, Dm[:, grid.index(10)])

    meta = json.loads((Path(inp["oracle_run"]) / "metadata.json").read_text())
    idx = V.HnswFile(meta["index"]["cache_path"])
    fcfg = yaml.safe_load((Path(inp["features_run"]) / "config.yaml").read_text())
    Q = np.fromfile(fcfg["dataset"]["queries"], dtype=np.float32).reshape(-1, 129)[:, 1:]
    vec = idx.vec.astype(np.float64)
    descent = np.empty(len(q))
    for t, qq in enumerate(q):
        qd = Q[qq].astype(np.float64)
        cur = int(idx.entry)
        best = float(((vec[cur] - qd) ** 2).sum())
        c = 1
        for level in range(idx.maxlevel, 0, -1):
            changed = True
            while changed:
                changed = False
                nb = idx.links(cur, level).astype(np.int64)
                d = ((vec[nb] - qd) ** 2).sum(axis=1)
                c += len(nb)
                for cand, dd in zip(nb, d):  # update one by one, as hnswlib does
                    if dd < best:
                        best, cur, changed = float(dd), int(cand), True
        descent[t] = c

    ladder = sorted({min(e for e in grid if e >= 10 * 2 ** i)
                     for i in range(1, 20) if any(e >= 10 * 2 ** i for e in grid)})
    sets = {
        "full_grid": ["probe"] + [e for e in grid if e > 10],
        "ladder_plus_probe_result": ["probe"] + ladder,
        "frozen_tiers": [18, 48, 327],
        "frozen_tiers_plus_probe_result": ["probe", 18, 48, 327],
    }

    def options(cands, accounting):
        cols, succ = [], []
        for c in cands:
            j = grid.index(10 if c == "probe" else c)
            succ.append(Sm[:, j])
            if accounting == "search_only":
                cols.append(np.zeros(len(q)) if c == "probe" else Dm[:, j])
            elif c == "probe":
                cols.append(probe.copy())
            elif accounting == "additive":
                cols.append(probe + Dm[:, j])
            else:
                cols.append(probe + Dm[:, j] - descent)
        return np.stack(cols, 1), np.stack(succ, 1)

    out = {"n_train": int(len(q)), "test_rows_read": 0, "ladder": ladder,
           "candidate_sets": {k: [str(c) for c in v] for k, v in sets.items()},
           "descent_dc": {"mean": float(descent.mean()),
                          "p10_p50_p90": np.percentile(descent, [10, 50, 90]).tolist(),
                          "share_of_probe_mean": float((descent / probe).mean())},
           "fixed_at_S": {}, "per_query_contract": {}, "mean_contract": {}}
    succ_rate = Sm.mean(0)
    for s in S_GRID:
        j = int(np.argmax(succ_rate >= s - 1e-12)) if (succ_rate >= s - 1e-12).any() else None
        out["fixed_at_S"][str(s)] = (None if j is None else
                                     {"ef": grid[j], "mean_dc": float(Dm[:, j].mean()),
                                      "success_rate": float(succ_rate[j])})
    for name, cands in sets.items():
        for acc in ("additive", "descent_shared", "search_only"):
            cost, succ = options(cands, acc)
            for s in S_GRID:
                r = contract_oracle(cost, succ, s)
                f = out["fixed_at_S"][str(s)]
                if r["feasible"] and f:
                    r["saving_vs_fixed"] = 1 - r["mean_cost"] / f["mean_dc"]
                out["per_query_contract"].setdefault(name, {}).setdefault(
                    acc, {})[str(s)] = r
    curve = rl.fixed_curve(curves, q, target)
    for name, cands in sets.items():
        cost, _ = options(cands, "additive")
        rec = np.stack([Rm[:, grid.index(10 if c == "probe" else c)] for c in cands], 1)
        for r_t in R_GRID:
            j = V.lagrange(rec, cost, r_t)
            fc = rl.interp_cost_at_recall(curve, r_t)
            if j is None:
                res = {"feasible": False}
            else:
                mc = float(cost[np.arange(len(q)), j].mean())
                res = {"feasible": True, "mean_cost": mc, "fixed_interp": fc,
                       "saving_vs_fixed": 1 - mc / fc}
            out["mean_contract"].setdefault(name, {})[str(r_t)] = res

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    od = Path(cfg["output_dir"]) / f"redesign_evidence_{ts}"
    od.mkdir(parents=True)
    (od / "redesign_train.json").write_text(rl.to_json(out) + "\n")
    print(od)
    return 0


if __name__ == "__main__":
    sys.exit(main())
