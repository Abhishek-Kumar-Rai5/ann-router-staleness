"""Experiment E orchestration (docs/exp_e_design_freeze_v2.md §18).

  prepare   write one frozen-DARTH eval config per (seed, state incl. S0); reads only
            the A1 manifest and run metadata (no outcomes)
  precheck  §18 step 4, S0 only:
              - REF(σ, S0) ≡ frozen B1(σ) (mix and per-query ef);
              - frozen-B1 test costs/recalls from the S0 oracle curves equal the
                Phase-4b per-query rows;
              - DARTH via the E pipeline at S0 reproduces the Stage-1/C1 S0 rows
                (non-timing columns).
  run       §18 steps 5–7: DARTH on every (seed, state) incl. S0 (33 cells); B1 and
            REF per-query rows from the oracle curves; §16 completeness checks.
            Writes results/exp_e_rows_<ts>/. No analysis happens here: the analysis
            is python/exp_e_analysis.py, run separately on that directory.

Usage: python python/exp_e_run.py {prepare|precheck|run} configs/exp_e/experiment_e.yaml
"""

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_e_lib as E  # noqa: E402
import router_lib as rl  # noqa: E402

TIMING = ["predictor_seconds", "wall_seconds", "plain_ef50_seconds", "plain_ef53_seconds"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def stop(msg):
    raise SystemExit(f"STOP: {msg}")


def cells(P):
    return [(int(s), st) for s in P["seeds"] for st in ["S0", *P["states"]]]


def cell_info(P, M, seed, state):
    """Index path, ground-truth prefix, oracle run and deletion list of one (seed, state)."""
    if state == "S0":
        c = P["s0"][seed]
        return {"index": c["index"], "gt_prefix": P["s0_gt_prefix"], "oracle_run": c["oracle_run"],
                "deleted_labels": "", "inserted": 0}
    st = M["seeds"][str(seed)]["states"][state]
    meta = json.loads((Path(st["oracle_run"]) / "metadata.json").read_text())
    return {"index": st["spec"]["index_path"], "gt_prefix": meta["ground_truth"]["cache_path"],
            "oracle_run": st["oracle_run"], "deleted_labels": st["spec"].get("deleted_labels", ""),
            "inserted": int(st["spec"].get("inserted_count", 0))}


def darth_cfg_path(P, seed, state):
    return Path(P["cell_config_dir"]) / f"darth_seed{seed}_{state}.yaml"


def prepare(P, M):
    verify_frozen(P)
    out = Path(P["cell_config_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for seed, state in cells(P):
        c = cell_info(P, M, seed, state)
        pol_path = P["s0"][seed]["darth_policy"]
        pol = json.loads(Path(pol_path).read_text())
        for f in (c["index"], c["gt_prefix"] + ".ivecs", c["gt_prefix"] + ".fvecs"):
            if not Path(f).exists():
                stop(f"missing input {f}")
        darth_cfg_path(P, seed, state).write_text(f"""# Experiment E cell: frozen DARTH seed {seed} on state {state} (design freeze v2 §3).
experiment_name: exp_e_darth_seed{seed}_{state}
index_path: {c['index']}
queries: {P['queries']}
split_path: {P['split_path']}
gt_prefix: {c['gt_prefix']}
k: {pol['k']}
ef_cap: {pol['ef_cap']}
verify_efs: [{pol['ef_cap']}]   # unused in eval mode
trace_dir: data/exp_e/unused     # unused in eval mode
policy_path: {pol_path}
output_dir: {P['output_dir']}
threads: 1
""")
    print(f"prepared {len(cells(P))} cell configs in {out}")


def test_ids(P):
    s = pd.read_csv(P["split_path"])
    return np.sort(s.loc[s.split == "test", "query_id"].to_numpy())


def b1_mix(P, seed):
    return json.loads(Path(P["b1_spec"].format(seed=seed)).read_text())["b1"]


def policy_rows(P, M, seed, state, qids):
    """Frozen B1 and REF per-query rows for one cell, from its oracle curves (§3/§5)."""
    c = cell_info(P, M, seed, state)
    curves = Path(c["oracle_run"]) / "oracle_curves.csv"
    train = rl.load_curves(curves, "train")
    if train.recall.shape[0] != 8000 or train.recall.shape[1] != 121 or train.recall.isna().any().any():
        stop(f"training oracle curves incomplete for seed {seed} {state}")
    ref = E.ref_mix(train)
    if ref is None:
        stop(f"two_ef_mix returned None for seed {seed} {state}")
    test = rl.load_curves(curves, "test")
    if not np.array_equal(np.sort(test.recall.index.to_numpy()), qids) or test.dc.isna().any().any():
        stop(f"test oracle rows incomplete for seed {seed} {state}")
    out = []
    for name, mix in (("B1", b1_mix(P, seed)), ("REF", ref)):
        ef = E.assign(mix, qids)
        cols = [list(test.recall.columns).index(e) for e in ef]
        r = np.arange(len(qids))
        out.append(pd.DataFrame({
            "policy": name, "seed": seed, "state": state, "query_id": qids, "ef": ef,
            "cost": test.dc.loc[qids].to_numpy()[r, cols].astype(float),
            "recall_tie_aware": test.recall.loc[qids].to_numpy()[r, cols],
            "recall_id": test.recall_id.loc[qids].to_numpy()[r, cols]}))
    return pd.concat(out), ref


def run_darth(P, seed, state):
    r = subprocess.run(["./build/exp_c_darth", "eval", str(darth_cfg_path(P, seed, state))],
                       capture_output=True, text=True)
    if r.returncode != 0:
        stop(f"DARTH eval failed for seed {seed} {state}: {r.stderr[-1000:]}")
    return r.stdout.strip().splitlines()[-1]


def darth_rows(run_dir, seed, state):
    d = pd.read_csv(Path(run_dir) / "darth_eval_rows.csv").sort_values("query_id")
    # Same cost accounting as B1/REF: total distance computations per query (§3).
    return d.assign(policy="DARTH", seed=seed, state=state, cost=d.distance_computations.astype(float))


def verify_frozen(P):
    """Frozen DARTH policy files and the models they reference must be byte-unchanged."""
    for seed, c in P["s0"].items():
        if sha(c["darth_policy"]) != c["darth_policy_sha256"]:
            stop(f"frozen DARTH policy hash mismatch for seed {seed}")
        pol = json.loads(Path(c["darth_policy"]).read_text())
        if sha(pol["model_path"]) != pol["model_sha256"]:
            stop(f"frozen DARTH model hash mismatch for seed {seed}")
        if not Path(P["b1_spec"].format(seed=seed)).exists():
            stop(f"frozen B1 spec missing for seed {seed}")


def precheck(P, M):
    verify_frozen(P)
    qids = test_ids(P)
    rep = {}
    old = pd.read_csv(P["b1_s0_rows"])
    old = old[(old.variant == "primary") & (old.contract == "mean_recall") & (old.level == 0.95)]
    for seed in P["seeds"]:
        rows, ref = policy_rows(P, M, seed, "S0", qids)
        b1 = rows[rows.policy == "B1"].set_index("query_id").loc[qids]
        rf = rows[rows.policy == "REF"].set_index("query_id").loc[qids]
        o = old[old.seed == seed].set_index("query_id").loc[qids]
        darth_run = run_darth(P, seed, "S0")
        T = [c for c in pd.read_csv(Path(darth_run) / "darth_eval_rows.csv").columns if c not in TIMING]
        a = pd.read_csv(Path(darth_run) / "darth_eval_rows.csv")[T]
        b = pd.read_csv(Path(P["s0"][seed]["darth_s0_reference_run"]) / "darth_eval_rows.csv")[T]
        rep[seed] = {
            "ref_mix_equals_frozen_b1": E.same_mix(ref, b1_mix(P, seed)),
            "ref_per_query_ef_equals_b1": bool(np.array_equal(rf.ef, b1.ef)),
            "b1_ef_equals_phase4b_rows": bool(np.array_equal(b1.ef.to_numpy(), o.b1_ef.to_numpy())),
            "b1_cost_equals_phase4b_rows": bool(np.array_equal(b1.cost.to_numpy(), o.b1_cost.to_numpy())),
            "b1_recall_equals_phase4b_rows": bool(np.allclose(b1.recall_tie_aware.to_numpy(), o.b1_recall.to_numpy())),
            "darth_e_pipeline_s0_equals_reference_run": bool(a.equals(b)),
            "darth_e_pipeline_s0_run": darth_run}
    ok = all(all(v for k, v in r.items() if k != "darth_e_pipeline_s0_run") for r in rep.values())
    od = Path(P["output_dir"]) / f"exp_e_precheck_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    od.mkdir(parents=True)
    (od / "precheck.json").write_text(json.dumps({"all_pass": ok, "seeds": rep}, indent=1, default=str) + "\n")
    print(json.dumps({"all_pass": ok}, indent=1))
    print(od)
    if not ok:
        stop("precheck failed")


def run(P, M):
    verify_frozen(P)
    for seed, state in cells(P):  # prepare must have produced every cell config
        if not darth_cfg_path(P, seed, state).exists():
            stop(f"missing cell config for seed {seed} {state}; run prepare first")
    qids = test_ids(P)
    frames, mixes, runs, checks = [], {}, {}, {}
    for seed, state in cells(P):
        rows, ref = policy_rows(P, M, seed, state, qids)
        mixes[f"{seed}|{state}"] = {"REF": ref, "B1": b1_mix(P, seed)}
        run_dir = run_darth(P, seed, state)
        runs[f"{seed}|{state}"] = run_dir
        d = darth_rows(run_dir, seed, state)
        # §16 completeness checks for the DARTH cell.
        c = cell_info(P, M, seed, state)
        labels = d["labels"].astype(str).str.split(" ").apply(lambda v: [int(x) for x in v])
        n_max = P["n0"] + c["inserted"]
        deleted = set(pd.read_csv(c["deleted_labels"])["label"]) if c["deleted_labels"] else set()
        bad = {
            "rows_ne_2000": len(d) != 2000 or not np.array_equal(d.query_id.to_numpy(), qids),
            "recall_out_of_range": not (d.recall_tie_aware.between(0, 1).all() and d.recall_id.between(0, 1).all()),
            "label_out_of_range": any(x >= n_max or x < 0 for v in labels for x in v),
            "deleted_label_returned": any(x in deleted for v in labels for x in v),
            "short_result": any(len(v) != 10 for v in labels)}
        checks[f"{seed}|{state}"] = bad
        if any(bad.values()):
            stop(f"§16 failed cell seed {seed} {state}: {bad}")
        frames += [rows, d.drop(columns=["labels"])]
        print(f"[exp_e] seed {seed} {state} done", flush=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    od = Path(P["output_dir"]) / f"exp_e_rows_{ts}"
    od.mkdir(parents=True)
    pd.concat(frames, ignore_index=True).to_csv(od / "rows.csv.gz", index=False)
    meta = {"plan": sys.argv[2], "plan_sha256": sha(sys.argv[2]),
            "design_doc_sha256": sha(P["design_doc"]),
            "scripts_sha256": {p: sha(p) for p in ("python/exp_e_lib.py", "python/exp_e_run.py",
                                                    "python/exp_e_analysis.py", "python/router4b_lib.py",
                                                    "python/router_lib.py")},
            "darth_binary_sha256": sha("build/exp_c_darth"),
            "policy_sha256": {s: P["s0"][s]["darth_policy_sha256"] for s in P["s0"]},
            "submodules": subprocess.run(["git", "submodule", "status"], capture_output=True,
                                         text=True).stdout.strip().splitlines(),
            "mixes": mixes, "darth_runs": runs, "s16_checks": checks}
    (od / "metadata.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    print(od)


def main() -> int:
    mode, plan = sys.argv[1], sys.argv[2]
    P = yaml.safe_load(Path(plan).read_text())
    M = json.loads(Path(P["manifest"]).read_text())
    {"prepare": prepare, "precheck": precheck, "run": run}[mode](P, M)
    return 0


if __name__ == "__main__":
    sys.exit(main())
