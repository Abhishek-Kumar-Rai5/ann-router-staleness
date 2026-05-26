"""Phase 5'(b) real-data orchestration (design_doc.md Addendum A1; pre-declared in
docs/notes.md). Same constructions and C++ tools as the validated synthetic run
(phase5_run.py), plus:
- the A1.4(1) SIFT insertion pool, rebuilt and checked against the stored file;
- S0 per seed = the canonical Phase 2/3 (seed 42) and Phase 4b (43, 44) runs, which
  are only read; S0 through the state path is run for the V2 reproduction check;
- ars_evolve runs in parallel (each is single-threaded, so deterministic), and is
  run a second time into repro_root for V6;
- feature re-runs on every state (V9) and the pre-declared oracle re-runs (V10).
Resumable: run_manifest.json is rewritten after every step, and finished steps are
skipped.

Usage: python python/phase5b_run.py configs/phase5/sift1m.yaml
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_lib as L  # noqa: E402
from phase5_run import features_cfg, oracle_cfg, run, s0_block, sha  # noqa: E402

TRAJ = (("id", "insert"), ("ood", "insert"), ("del", "delete"))


def save(root, man):
    (root / "run_manifest.json").write_text(json.dumps(man, indent=1) + "\n")


def evolve_cfg(P, seed, traj, kind, order, cnts, state_dir, s0_index):
    return f"""experiment_name: phase5_evolve_{traj}_seed{seed}
{s0_block(P, seed)}evolution:
  trajectory: {kind}
  name: {traj}
  pool: {P['pool']}
  order: {order}
  counts: [{', '.join(map(str, cnts))}]
  state_dir: {state_dir}
  verify_s0: {s0_index}
"""


def run_parallel(jobs):
    """jobs: list of (key, cfg_text, cfg_path). Returns {key: run_dir}."""
    procs = {}
    for key, text, path in jobs:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
        procs[key] = subprocess.Popen(["./build/ars_evolve", str(path)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = {}
    for key, p in procs.items():
        so, se = p.communicate()
        if p.returncode != 0:
            raise SystemExit(f"STOP: ars_evolve failed for {key}:\n{se[-2000:]}")
        out[key] = so.strip().splitlines()[-1]
    return out


def main() -> int:
    plan_path = Path(sys.argv[1])
    P = yaml.safe_load(plan_path.read_text())
    D = P["data"]
    root, rroot = Path(P["state_root"]), Path(P["repro_root"])
    root.mkdir(parents=True, exist_ok=True)
    P["base"], P["queries"], P["pool"] = D["base"], D["queries"], D["pool"]
    P["split_path"], P["split_fixed"] = D["split_path"], False
    P["split_seed"], P["split_test_size"] = D["split_seed"], D["split_test_size"]

    # ---- A1.4(1) pool: rebuild and require equality with the stored file -------
    learn, Q = L.read_fvecs(D["learn"]), L.read_fvecs(P["queries"])
    conf = pd.read_csv(D["confirm_ids"])["source_row"].to_numpy()
    pool, rows = L.sift_pool(learn, Q, conf)
    if len(pool) != D["expected_pool_size"]:
        raise SystemExit(f"STOP: pool size {len(pool)} != {D['expected_pool_size']}")
    if not Path(P["pool"]).exists():
        Path(P["pool"]).parent.mkdir(parents=True, exist_ok=True)
        L.write_fvecs(P["pool"], pool)
        pd.DataFrame({"pool_row": np.arange(len(pool)), "sift_learn_row": rows}).to_csv(D["pool_rows"], index=False)
    if not np.array_equal(L.read_fvecs(P["pool"]), pool):
        raise SystemExit("STOP: stored pool differs from the A1.4(1) construction")

    n0 = len(L.read_fvecs(P["base"]))
    counts = [int(round(f * n0)) for f in P["counts_fraction"]]
    dcounts = [int(round(f * n0)) for f in P["delete_fraction"]]
    S = P["seeds"]
    idp = L.id_order(len(pool), S["id_order"])
    oodp, anchor, _ = L.ood_order(pool, counts, idp, S["ood_anchor"])
    delp = L.fisher_yates(n0, S["deletion"])[: dcounts[-1]]
    orders = {"id": root / "order_id.csv", "ood": root / "order_ood.csv", "del": root / "order_del.csv"}
    L.write_order(orders["id"], idp[: counts[-1]])
    L.write_order(orders["ood"], oodp)
    L.write_order(orders["del"], delp)

    mpath = root / "run_manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"seeds": {}}
    man.update({"plan": str(plan_path), "plan_sha256": sha(plan_path), "n0": n0, "counts": counts,
                "delete_counts": dcounts, "anchor_pool_row": anchor, "pool_size": len(pool),
                "data_sha256": {k: sha(P[k]) for k in ("base", "pool", "queries")},
                "order_sha256": {k: sha(v) for k, v in orders.items()}})
    cnt = {"insert": counts, "delete": dcounts}

    # ---- S0 index per seed (from the canonical oracle run) --------------------
    s0_index = {}
    for seed in P["index"]["seeds"]:
        c = P["s0_canonical"][seed]
        s0_index[seed] = json.loads((Path(c["oracle"]) / "metadata.json").read_text())["index"]["cache_path"]
        man["seeds"].setdefault(str(seed), {"states": {}, "evolve": {}, "evolve_repro": {}})
        man["seeds"][str(seed)]["states"]["S0"] = {"oracle_run": c["oracle"], "features_run": c["features"],
                                                   "index": s0_index[seed], "canonical": True}

    # ---- state production (V3) and its reproduction (V6), in parallel ----------
    for tag, sroot in (("evolve", root), ("evolve_repro", rroot)):
        jobs = []
        for seed in P["index"]["seeds"]:
            for traj, kind in TRAJ:
                if traj in man["seeds"][str(seed)][tag]:
                    continue
                sd = sroot / f"seed{seed}"
                jobs.append(((seed, traj), evolve_cfg(P, seed, traj, kind, orders[traj], cnt[kind],
                                                      sd / traj, s0_index[seed]),
                             sd / "configs" / f"evolve_{traj}.yaml"))
        if jobs:
            print(f"[phase5b] {tag}: {len(jobs)} ars_evolve jobs in parallel", flush=True)
            for (seed, traj), erun in run_parallel(jobs).items():
                man["seeds"][str(seed)][tag][traj] = erun
            save(root, man)

    # ---- oracle + features on every state ----------------------------------------
    for seed in P["index"]["seeds"]:
        m = man["seeds"][str(seed)]
        cfgdir = root / f"seed{seed}" / "configs"
        if "s0_via_state" not in m:
            st = {"name": f"S0viaState_seed{seed}", "index_path": s0_index[seed]}
            m["s0_via_state"] = {
                "oracle_run": run("ars_oracle", oracle_cfg(P, seed, "S0viaState", st), cfgdir / "oracle_S0viaState.yaml"),
                "features_run": run("ars_features", features_cfg(P, seed, "S0viaState", st),
                                    cfgdir / "features_S0viaState.yaml")}
            save(root, man)
            print(f"[phase5b] seed {seed} S0viaState done", flush=True)
        for traj, kind in TRAJ:
            erun = m["evolve"][traj]
            emeta = json.loads((Path(erun) / "metadata.json").read_text())
            for stt in emeta["states"]:
                name = f"{traj}_{stt['count']}"
                if name in m["states"] and "oracle_run" in m["states"][name]:
                    continue
                spec = {"name": f"{name}_seed{seed}", "index_path": stt["index"]}
                if kind == "insert":
                    spec.update(inserted_vectors=stt["inserted_vectors"], inserted_count=stt["count"])
                else:
                    spec.update(deleted_labels=stt["deleted_labels"])
                m["states"][name] = {
                    "trajectory": traj, "count": stt["count"], "evolve_run": erun,
                    "in_process_s0_byte_identical": emeta["in_process_s0_byte_identical"],
                    "spec": spec,
                    "features_run": run("ars_features", features_cfg(P, seed, name, spec),
                                        cfgdir / f"features_{name}.yaml"),
                    "oracle_run": run("ars_oracle", oracle_cfg(P, seed, name, spec), cfgdir / f"oracle_{name}.yaml")}
                save(root, man)
                print(f"[phase5b] seed {seed} {name} done", flush=True)

    # ---- V9 feature re-runs (all states), V10 oracle re-runs (pre-declared) -------
    for seed in P["index"]["seeds"]:
        m = man["seeds"][str(seed)]
        cfgdir = root / f"seed{seed}" / "configs" / "repro"
        for name, st in m["states"].items():
            if name == "S0" or "features_repro" in st:
                continue
            spec = dict(st["spec"], name=st["spec"]["name"] + "_repro")
            st["features_repro"] = run("ars_features", features_cfg(P, seed, name + "_repro", spec),
                                       cfgdir / f"features_{name}.yaml")
            save(root, man)
    for seed, name in P["oracle_repro"]:
        st = man["seeds"][str(seed)]["states"][name]
        if "oracle_repro" in st:
            continue
        spec = dict(st["spec"], name=st["spec"]["name"] + "_repro")
        st["oracle_repro"] = run("ars_oracle", oracle_cfg(P, seed, name + "_repro", spec),
                                 root / f"seed{seed}" / "configs" / "repro" / f"oracle_{name}.yaml")
        save(root, man)
        print(f"[phase5b] oracle repro seed {seed} {name} done", flush=True)
    print(mpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
