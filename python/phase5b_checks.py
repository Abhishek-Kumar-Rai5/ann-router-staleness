"""Phase 5'(b) real-data validation V1-V11, as pre-declared in docs/notes.md.
Read-only: a failure means stop and report, nothing gets repaired here.

Usage: python python/phase5b_checks.py configs/phase5/sift1m.yaml
"""

import datetime as dt
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_lib as L  # noqa: E402
from phase5_checks import FEATS, gt_of, numpy_gt  # noqa: E402


PHASE4B_ERA_FEATURE_RERUNS = {"42": "results/657980fb7c23_20261002T144828Z",
                              "43": "results/94945b3661cb_20261002T144831Z",
                              "44": "results/1c816578a946_20261002T144834Z"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def strip_first_col(p):
    return [ln.split(",", 1)[1] for ln in Path(p).read_text().splitlines()]


def hnsw_header(p):
    with open(p, "rb") as f:
        _, max_el, cur = struct.unpack("<QQQ", f.read(24))
    return max_el, cur


def main() -> int:
    P = yaml.safe_load(Path(sys.argv[1]).read_text())
    D = P["data"]
    root = Path(P["state_root"])
    M = json.loads((root / "run_manifest.json").read_text())
    base, Q, pool = L.read_fvecs(D["base"]), L.read_fvecs(D["queries"]), L.read_fvecs(D["pool"])
    n0, nq = len(base), len(Q)
    counts, dcounts = M["counts"], M["delete_counts"]
    seeds = [str(s) for s in P["index"]["seeds"]]
    R = {"checks": {}}

    def put(name, ok, **kw):
        R["checks"][name] = {"pass": bool(ok), **kw}
        print(f"{name}: {'PASS' if ok else 'FAIL'}", flush=True)

    cur = {"data_sha256": {k: sha(D[k]) for k in ("base", "pool", "queries")},
           "order_sha256": {k: sha(root / f"order_{k}.csv") for k in ("id", "ood", "del")},
           "plan_sha256": sha(sys.argv[1])}
    ok = all(cur[k] == M[k] for k in cur)
    ok &= P["seeds"] == {"id_order": 20261008, "deletion": 20261009, "ood_anchor": 20261010}
    put("V1_provenance", ok, recorded=cur, seeds=P["seeds"])

    o_id = pd.read_csv(root / "order_id.csv")["row"].to_numpy()
    o_ood = pd.read_csv(root / "order_ood.csv")["row"].to_numpy()
    o_del = pd.read_csv(root / "order_del.csv")["row"].to_numpy()
    idp = L.id_order(len(pool), P["seeds"]["id_order"])
    oodp, anchor, dist = L.ood_order(pool, counts, idp, P["seeds"]["ood_anchor"])
    by_dist = np.lexsort((np.arange(len(pool)), dist))
    v5 = {"id_order_is_fisher_yates_prefix": bool(np.array_equal(o_id, idp[: counts[-1]])),
          "ood_order_matches": bool(np.array_equal(o_ood, oodp)),
          "ood_prefixes_are_anchor_balls": all(set(o_ood[:c]) == set(by_dist[:c]) for c in counts),
          "del_order_is_fisher_yates_prefix": bool(np.array_equal(
              o_del, L.fisher_yates(n0, P["seeds"]["deletion"])[: dcounts[-1]])),
          "orders_distinct_rows": bool(len(set(o_id)) == len(o_id) and len(set(o_ood)) == len(o_ood)
                                       and len(set(o_del)) == len(o_del))}
    overlap_id_ood = {c: len(set(o_id[:c]) & set(o_ood[:c])) / c for c in counts}

    gt_ref = {}
    overlap = {}
    for seed in seeds:
        S = M["seeds"][seed]["states"]
        st_names = [n for n in S if n != "S0"]
        can, via = S["S0"], M["seeds"][seed]["s0_via_state"]
        v2 = {"curves": Path(can["oracle_run"], "oracle_curves.csv").read_bytes()
              == Path(via["oracle_run"], "oracle_curves.csv").read_bytes(),
              "labels": strip_first_col(Path(can["oracle_run"], "oracle_labels.csv"))
              == strip_first_col(Path(via["oracle_run"], "oracle_labels.csv")),
              "features": strip_first_col(Path(can["features_run"], "features.csv"))
              == strip_first_col(Path(via["features_run"], "features.csv"))}
        a = pd.read_csv(Path(can["features_run"], "features.csv"), dtype=str)
        b = pd.read_csv(Path(via["features_run"], "features.csv"), dtype=str)
        shared = [c for c in a.columns if c != "experiment_id"]
        v2["features_shared_columns_identical"] = bool(set(shared) <= set(b.columns)
                                                       and a[shared].equals(b[shared]))
        v2["extra_columns_in_state_path"] = [c for c in b.columns if c not in a.columns]
        ext = PHASE4B_ERA_FEATURE_RERUNS[seed]
        v2["features_identical_to_phase4b_era_rerun"] = (
            strip_first_col(Path(ext, "features.csv")) == strip_first_col(Path(via["features_run"], "features.csv")))
        v2["phase4b_era_rerun"] = ext
        v2["features_literal_pre_declared"] = v2["features"]
        approved = (v2["curves"] and v2["labels"] and v2["features_shared_columns_identical"]
                    and v2["features_identical_to_phase4b_era_rerun"])
        v2["rule"] = "approved interpretation 2026-10-04 (pre-existing schema evolution)"
        put(f"V2_S0_reproduction_seed{seed}", approved, **v2)

        v3 = {t: json.loads((Path(M["seeds"][seed]["evolve"][t]) / "metadata.json").read_text())
              ["in_process_s0_byte_identical"] for t in ("id", "ood")}
        put(f"V3_rebuilt_S0_identical_seed{seed}", all(v3.values()), **v3)

        v4, v8 = [], []
        for name in st_names:
            st, spec = S[name], S[name]["spec"]
            max_el, cur_el = hnsw_header(spec["index_path"])
            chk = subprocess.run(["./build/ars_check_state", str(root / f"seed{seed}" / "configs"
                                                                 / f"features_{name}.yaml")],
                                 capture_output=True, text=True)
            if chk.returncode != 0:
                raise SystemExit(f"STOP: ars_check_state failed on {name}: {chk.stderr[-500:]}")
            cs = json.loads(chk.stdout.strip().splitlines()[-1])
            if st["trajectory"] == "del":
                ndel = len(pd.read_csv(spec["deleted_labels"]))
                ok = cur_el == n0 and ndel == st["count"] and cs["index_deleted_count"] == st["count"]
                frac = ndel / n0
            else:
                ok = cur_el == n0 + st["count"] and cs["index_deleted_count"] == 0
                frac = spec["inserted_count"] / n0
            target = {"del": dcounts, "id": counts, "ood": counts}[st["trajectory"]]
            ok &= st["count"] in target
            v4.append({"state": name, "elements": cur_el, "capacity": max_el, "update_fraction": frac,
                       "deleted_in_index": cs["index_deleted_count"], "ok": bool(ok)})
            v8.append({"state": name, **cs, "ok": cs["deleted_returned"] == 0 and cs["out_of_range_returned"] == 0
                       and cs["labels_returned"] == 3 * nq * 10})
        put(f"V4_sizes_counts_seed{seed}", all(x["ok"] for x in v4), states=v4)
        put(f"V8_deleted_exclusion_seed{seed}", all(x["ok"] for x in v8), states=v8)

        v5f = []
        for traj, order in (("id", o_id), ("ood", o_ood)):
            ins = L.read_fvecs(S[f"{traj}_{counts[-1]}"]["spec"]["inserted_vectors"])
            v5f.append({"traj": traj, "ok": bool(ins.shape[0] == counts[-1]
                                                 and np.array_equal(ins, pool[order[: counts[-1]]]))})
            v5f[-1]["same_file_all_states"] = len({S[f"{traj}_{c}"]["spec"]["inserted_vectors"] for c in counts}) == 1
            v5f[-1]["ok"] &= v5f[-1]["same_file_all_states"]
        for c in dcounts:
            lab = pd.read_csv(S[f"del_{c}"]["spec"]["deleted_labels"])["label"].to_numpy()
            v5f.append({"traj": f"del_{c}", "ok": bool(np.array_equal(lab, o_del[:c]))})

        gt0_i, _ = gt_of(S["S0"]["oracle_run"])
        top0 = [set(r[:10]) for r in gt0_i]
        v7 = []
        for name in st_names:
            st, spec = S[name], S[name]["spec"]
            gi, gd = gt_of(st["oracle_run"])
            vec = base
            excl = np.zeros(n0, bool)
            if st["trajectory"] == "del":
                excl[pd.read_csv(spec["deleted_labels"])["label"].to_numpy()] = True
            else:
                vec = np.vstack([base, L.read_fvecs(spec["inserted_vectors"])[: spec["inserted_count"]]])
                excl = np.zeros(len(vec), bool)
            key = (st["trajectory"], st["count"])
            if key not in gt_ref:
                gt_ref[key] = numpy_gt(vec, excl, Q)
                print(f"  numpy GT {key} done", flush=True)
            ni, nd = gt_ref[key]
            bad = 0
            for q in np.nonzero((np.sort(gi[:, :10], 1) != np.sort(ni[:, :10], 1)).any(1))[0]:
                a, b = set(gi[q, :10]), set(ni[q, :10])
                if a == b:
                    continue
                kth = nd[q, 9]
                for x in a ^ b:
                    dx = float(((vec[x].astype(np.float64) - Q[q]) ** 2).sum())
                    if abs(dx - kth) > 1e-5 * max(kth, 1.0):
                        bad += 1
                        break
            prof = float(np.max(np.abs(gd.astype(np.float64) - nd) / np.maximum(nd, 1e-12)))
            del_in_gt = int(excl[gi].sum())
            v7.append({"state": name, "top10_rows_differing_beyond_float_ties": bad,
                       "max_rel_distance_error_top100": prof, "deleted_labels_in_gt": del_in_gt,
                       "ok": bad == 0 and prof < 1e-4 and del_in_gt == 0})
            overlap[(seed, name)] = np.array([len(top0[q] & set(gi[q, :10])) for q in range(nq)])
        put(f"V7_ground_truth_seed{seed}", all(x["ok"] for x in v7), states=v7)

        mono = {}
        for traj in ("id", "ood", "del"):
            names = sorted([n for n in st_names if S[n]["trajectory"] == traj], key=lambda n: S[n]["count"])
            seq = np.stack([np.full(nq, 10)] + [overlap[(seed, n)] for n in names])
            mono[traj] = {"states": names, "violations": int((np.diff(seq, axis=0) > 0).sum()),
                          "mean_overlap": [float(x.mean()) for x in seq]}
        ok5 = all(v5.values()) and all(x["ok"] for x in v5f) and all(m["violations"] == 0 for m in mono.values())
        put(f"V5_nestedness_seed{seed}", ok5, orders=v5, files=v5f, overlap_monotonicity=mono)

        v6 = []
        for traj in ("id", "ood", "del"):
            m1 = json.loads((Path(M["seeds"][seed]["evolve"][traj]) / "metadata.json").read_text())
            m2 = json.loads((Path(M["seeds"][seed]["evolve_repro"][traj]) / "metadata.json").read_text())
            for a, b in zip(m1["states"], m2["states"], strict=True):
                h1, h2 = sha(a["index"]), sha(b["index"])
                v6.append({"state": f"{traj}_{a['count']}", "sha256": h1, "repro_sha256": h2, "ok": h1 == h2})
                S[f"{traj}_{a['count']}"]["index_sha256"] = h1
        put(f"V6_index_reproducibility_seed{seed}", all(x["ok"] for x in v6), states=v6)

        v9 = [{"state": n, "ok": strip_first_col(Path(S[n]["features_run"], "features.csv"))
               == strip_first_col(Path(S[n]["features_repro"], "features.csv"))} for n in st_names]
        put(f"V9_feature_reproducibility_seed{seed}", all(x["ok"] for x in v9), states=v9)

        v11 = []
        for name in ["S0"] + st_names:
            st = S[name]
            fe = pd.read_csv(Path(st["features_run"]) / "features.csv")
            la = pd.read_csv(Path(st["oracle_run"]) / "oracle_labels.csv")
            ok = (fe["query_id"].tolist() == list(range(nq)) and la["query_id"].tolist() == list(range(nq))
                  and np.isfinite(fe[FEATS].to_numpy()).all())
            cfg_h = {}
            for kind in ("oracle_run", "features_run"):
                meta = json.loads((Path(st[kind]) / "metadata.json").read_text())
                ok &= meta.get("seeds", {}).get("hnsw_level_seed", int(seed)) == int(seed)
                cy = meta.get("config_yaml", "")
                ok &= f"seed: {seed}" in cy and "queries: data/sift/sift_query.fvecs" in cy
                cfg_h[kind] = hashlib.sha256(cy.encode()).hexdigest()[:16]
            v11.append({"state": name, "config_sha256_16": cfg_h, "ok": bool(ok)})
        put(f"V11_query_seed_config_identity_seed{seed}", all(x["ok"] for x in v11), states=v11)

    v10 = []
    for seed, name in P["oracle_repro"]:
        st = M["seeds"][str(seed)]["states"][name]
        a, b = st["oracle_run"], st["oracle_repro"]
        cur_ok = Path(a, "oracle_curves.csv").read_bytes() == Path(b, "oracle_curves.csv").read_bytes()
        lab_ok = strip_first_col(Path(a, "oracle_labels.csv")) == strip_first_col(Path(b, "oracle_labels.csv"))
        gi1, gd1 = gt_of(a)
        gi2, gd2 = gt_of(b)
        gt_ok = np.array_equal(gi1, gi2) and np.array_equal(gd1, gd2)
        v10.append({"seed": seed, "state": name, "curves_identical": cur_ok, "labels_identical": lab_ok,
                    "fresh_gt_identical": bool(gt_ok), "ok": cur_ok and lab_ok and gt_ok})
    put("V10_oracle_reproducibility", all(x["ok"] for x in v10), runs=v10)

    xs = []
    for name in [n for n in M["seeds"][seeds[0]]["states"] if n != "S0"]:
        g = [gt_of(M["seeds"][s]["states"][name]["oracle_run"]) for s in seeds]
        xs.append({"state": name, "ok": all(np.array_equal(g[0][0], x[0]) and np.array_equal(g[0][1], x[1])
                                            for x in g[1:])})
    put("V7b_ground_truth_identical_across_seeds", all(x["ok"] for x in xs), states=xs)

    R["descriptive"] = {"ood_id_overlap": overlap_id_ood, "anchor_pool_row": int(anchor)}
    R["ALL_PASS"] = all(v["pass"] for v in R["checks"].values())
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    od = Path(P["output_dir"]) / f"phase5b_validation_{ts}"
    od.mkdir(parents=True)
    (od / "report.json").write_text(json.dumps(R, indent=1, default=float) + "\n")
    np.savez_compressed(od / "overlap_top10.npz", **{f"{s}__{n}": v for (s, n), v in overlap.items()})
    (root / "run_manifest_with_hashes.json").write_text(json.dumps(M, indent=1) + "\n")
    print("ALL_PASS:", R["ALL_PASS"])
    print(od)
    return 0 if R["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
