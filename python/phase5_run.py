"""Runs the Phase 5' synthetic pipeline: builds S0, evolves it, then runs the oracle
and features on every state.

Usage: python python/phase5_run.py configs/phase5/synthetic.yaml
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_lib as L  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(app, cfg_text, cfg_path):
    Path(cfg_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg_path).write_text(cfg_text)
    r = subprocess.run([f"./build/{app}", str(cfg_path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"STOP: {app} failed on {cfg_path}:\n{r.stderr[-2000:]}")
    return r.stdout.strip().splitlines()[-1]


def s0_block(P, seed):
    return f"""dataset:
  base: {P['base']}
  queries: {P['queries']}
  max_base: 0
  max_queries: 0
index:
  m: {P['index']['m']}
  ef_construction: {P['index']['ef_construction']}
  seed: {seed}
  build_threads: 1
ground_truth:
  k: {P['oracle']['gt_k']}
paths:
  cache_dir: {P['cache_dir']}
  output_dir: {P['output_dir']}
"""


def state_block(st):
    if st is None:
        return ""
    return f"""state:
  name: {st['name']}
  index_path: {st['index_path']}
  inserted_vectors: "{st.get('inserted_vectors', '')}"
  inserted_count: {st.get('inserted_count', 0)}
  deleted_labels: "{st.get('deleted_labels', '')}"
"""


def oracle_cfg(P, seed, name, st):
    o = P["oracle"]
    return f"""experiment_name: phase5_oracle_{name}_seed{seed}
{s0_block(P, seed)}{state_block(st)}split:
  path: {P['split_path']}
  fixed_file: {str(P['split_fixed']).lower()}
  seed: {P.get('split_seed', 0)}
  test_size: {P.get('split_test_size', 1)}
oracle:
  k: {o['k']}
  target_recall: {o['target_recall']}
  recall_definition: tie_aware
  ef_grid: {{min: {o['ef_grid']['min']}, max: {o['ef_grid']['max']}, ratio: {o['ef_grid']['ratio']}}}
  search_threads: {o['threads']}
"""


def features_cfg(P, seed, name, st):
    f = P["features"]
    return f"""experiment_name: phase5_features_{name}_seed{seed}
{s0_block(P, seed)}{state_block(st)}split:
  path: {P['split_path']}
features: {{probe_k: {f['probe_k']}, probe_ef: {f['probe_ef']}, lid_k: {f['lid_k']}, lid_ef: {f['lid_ef']}, threads: {f['threads']}}}
"""


def main() -> int:
    plan_path = Path(sys.argv[1])
    P = yaml.safe_load(plan_path.read_text())
    root = Path(P["state_root"])
    root.mkdir(parents=True, exist_ok=True)
    D = P["data"]
    if D["kind"] == "synthetic":
        out = Path(D["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        P["base"], P["queries"], P["pool"] = (str(out / "base.fvecs"), str(out / "queries.fvecs"),
                                              str(out / "pool.fvecs"))
        if not Path(P["base"]).exists():
            rng = np.random.default_rng(D["seed"])
            means = rng.normal(0, D["mean_scale"], size=(D["components"], D["dim"]))

            def draw(n):
                c = rng.integers(0, D["components"], n)
                return (means[c] + rng.normal(size=(n, D["dim"]))).astype(np.float32)
            L.write_fvecs(P["base"], draw(D["n_base"]))
            L.write_fvecs(P["pool"], draw(D["n_pool"]))
            L.write_fvecs(P["queries"], draw(D["n_queries"]))
        P["split_path"] = str(out / "queries_tagging.csv")
        P["split_fixed"] = True
        with open(P["split_path"], "w") as f:
            f.write("query_id,split\n" + "".join(f"{i},test\n" for i in range(D["n_queries"])))
    else:
        P["base"], P["queries"], P["pool"] = D["base"], D["queries"], D["pool"]
        P["split_path"], P["split_fixed"] = D["split_path"], False
        P["split_seed"], P["split_test_size"] = D["split_seed"], D["split_test_size"]

    base = L.read_fvecs(P["base"])
    pool = L.read_fvecs(P["pool"])
    n0 = len(base)
    counts = [int(round(f * n0)) for f in P["counts_fraction"]]
    dcounts = [int(round(f * n0)) for f in P["delete_fraction"]]
    if counts[-1] > len(pool):
        raise SystemExit("STOP: insertion pool smaller than the largest count")
    S = P["seeds"]
    idp = L.id_order(len(pool), S["id_order"])
    oodp, anchor, _ = L.ood_order(pool, counts, idp, S["ood_anchor"])
    delp = L.fisher_yates(n0, S["deletion"])[: dcounts[-1]]
    orders = {"id": root / "order_id.csv", "ood": root / "order_ood.csv", "del": root / "order_del.csv"}
    L.write_order(orders["id"], idp[: counts[-1]])
    L.write_order(orders["ood"], oodp)
    L.write_order(orders["del"], delp)

    manifest = {"plan": str(plan_path), "plan_sha256": sha(plan_path), "n0": n0, "counts": counts,
                "delete_counts": dcounts, "anchor_pool_row": anchor,
                "data_sha256": {k: sha(P[k]) for k in ("base", "pool", "queries")},
                "order_sha256": {k: sha(v) for k, v in orders.items()}, "seeds": {}}
    for seed in P["index"]["seeds"]:
        sd = root / f"seed{seed}"
        m = {"states": {}}
        cfgdir = sd / "configs"
        o = run("ars_oracle", oracle_cfg(P, seed, "S0", None), cfgdir / "oracle_S0.yaml")
        f = run("ars_features", features_cfg(P, seed, "S0", None), cfgdir / "features_S0.yaml")
        s0_index = json.loads((Path(o) / "metadata.json").read_text())["index"]["cache_path"]
        m["states"]["S0"] = {"oracle_run": o, "features_run": f, "index": s0_index}
        st = {"name": f"S0viaState_seed{seed}", "index_path": s0_index}
        m["s0_via_state"] = {
            "oracle_run": run("ars_oracle", oracle_cfg(P, seed, "S0viaState", st), cfgdir / "oracle_S0viaState.yaml"),
            "features_run": run("ars_features", features_cfg(P, seed, "S0viaState", st), cfgdir / "features_S0viaState.yaml")}
        for traj, kind, cnts in (("id", "insert", counts), ("ood", "insert", counts), ("del", "delete", dcounts)):
            ecfg = f"""experiment_name: phase5_evolve_{traj}_seed{seed}
{s0_block(P, seed)}evolution:
  trajectory: {kind}
  name: {traj}
  pool: {P['pool']}
  order: {orders[traj]}
  counts: [{', '.join(map(str, cnts))}]
  state_dir: {sd / traj}
  verify_s0: {s0_index}
"""
            erun = run("ars_evolve", ecfg, cfgdir / f"evolve_{traj}.yaml")
            emeta = json.loads((Path(erun) / "metadata.json").read_text())
            for stt in emeta["states"]:
                name = f"{traj}_{stt['count']}"
                spec = {"name": f"{name}_seed{seed}", "index_path": stt["index"]}
                if kind == "insert":
                    spec.update(inserted_vectors=stt["inserted_vectors"], inserted_count=stt["count"])
                else:
                    spec.update(deleted_labels=stt["deleted_labels"])
                m["states"][name] = {
                    "trajectory": traj, "count": stt["count"], "evolve_run": erun,
                    "in_process_s0_byte_identical": emeta["in_process_s0_byte_identical"],
                    "spec": spec,
                    "oracle_run": run("ars_oracle", oracle_cfg(P, seed, name, spec), cfgdir / f"oracle_{name}.yaml"),
                    "features_run": run("ars_features", features_cfg(P, seed, name, spec), cfgdir / f"features_{name}.yaml")}
                print(f"[phase5] seed {seed} {name} done", flush=True)
        manifest["seeds"][str(seed)] = m
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(root / "run_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
