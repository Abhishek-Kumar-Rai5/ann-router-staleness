"""Phase 5'(a) synthetic validation checks C1-C6, exactly as pre-declared in docs/notes.md.

Usage: python python/phase5_checks.py configs/phase5/synthetic.yaml
"""

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_lib as L  # noqa: E402

FEATS = ["knn_dist", "centroid_dist", "score_concentration", "lid"]


def gt_of(oracle_run):
    meta = json.loads((Path(oracle_run) / "metadata.json").read_text())
    p = meta["ground_truth"]["cache_path"]
    return L.read_ivecs(p + ".ivecs"), L.read_fvecs(p + ".fvecs")


def live_vectors(base, spec):
    vec = base
    if spec and spec.get("inserted_count", 0):
        vec = np.vstack([base, L.read_fvecs(spec["inserted_vectors"])[: spec["inserted_count"]]])
    excluded = np.zeros(len(vec), bool)
    if spec and spec.get("deleted_labels"):
        excluded[pd.read_csv(spec["deleted_labels"])["label"].to_numpy()] = True
    return vec, excluded


def numpy_gt(vec, excluded, Q, k=100):
    v = vec.astype(np.float64)
    vn = (v ** 2).sum(1)
    out_i = np.empty((len(Q), k), np.int64)
    out_d = np.empty((len(Q), k))
    for s in range(0, len(Q), 250):
        q = Q[s:s + 250].astype(np.float64)
        d = (q ** 2).sum(1)[:, None] + vn[None, :] - 2 * q @ v.T
        d[:, excluded] = np.inf
        idx = np.argpartition(d, k, axis=1)[:, :k]
        dd = np.take_along_axis(d, idx, 1)
        o = np.lexsort((idx, dd), axis=1)
        out_i[s:s + 250] = np.take_along_axis(idx, o, 1)
        out_d[s:s + 250] = np.take_along_axis(dd, o, 1)
    return out_i, np.maximum(out_d, 0)


def main() -> int:
    P = yaml.safe_load(Path(sys.argv[1]).read_text())
    root = Path(P["state_root"])
    M = json.loads((root / "run_manifest.json").read_text())
    D = P["data"]
    base = L.read_fvecs(str(Path(D["out_dir"]) / "base.fvecs"))
    Q = L.read_fvecs(str(Path(D["out_dir"]) / "queries.fvecs"))
    pool = L.read_fvecs(str(Path(D["out_dir"]) / "pool.fvecs"))
    n0 = len(base)
    rep = {"checks": {}, "descriptive": {}}

    for seed, m in M["seeds"].items():
        S = m["states"]
        gt0_i, _ = gt_of(S["S0"]["oracle_run"])
        top0 = [set(r[:10]) for r in gt0_i]
        f0 = pd.read_csv(Path(S["S0"]["features_run"]) / "features.csv").sort_values("query_id")

        c1 = []
        for name, st in S.items():
            if name == "S0":
                continue
            spec = st["spec"]
            if st["trajectory"] == "del":
                frac = len(pd.read_csv(spec["deleted_labels"])) / n0
            else:
                frac = spec["inserted_count"] / n0
            target = st["count"] / n0
            c1.append({"state": name, "fraction": frac, "ok": abs(frac - target) < 1e-12})
        rep["checks"][f"C1_seed{seed}"] = {"pass": all(x["ok"] for x in c1), "states": c1}

        overlap = {}
        c4 = []
        for name, st in S.items():
            if name == "S0":
                continue
            spec = st["spec"]
            gi, gd = gt_of(st["oracle_run"])
            vec, excl = live_vectors(base, spec)
            ni, nd = numpy_gt(vec, excl, Q)
            bad = 0
            for q in range(len(Q)):
                a, b = set(gi[q, :10]), set(ni[q, :10])
                if a != b:
                    kth = nd[q, 9]
                    for x in a ^ b:
                        dx = float(((vec[x].astype(np.float64) - Q[q]) ** 2).sum())
                        if abs(dx - kth) > 1e-5 * max(kth, 1.0):
                            bad += 1
                            break
            prof = float(np.max(np.abs(gd.astype(np.float64) - nd) / np.maximum(nd, 1e-12)))
            del_in_gt = int(excl[gi].sum()) if excl.any() else 0
            chk = subprocess.run(["./build/ars_check_state",
                                  str(root / f"seed{seed}" / "configs" / f"features_{name}.yaml")],
                                 capture_output=True, text=True)
            cs = json.loads(chk.stdout.strip().splitlines()[-1])
            c4.append({"state": name, "top10_rows_differing_beyond_float_ties": bad,
                       "max_rel_distance_error_top100": prof, "deleted_labels_in_gt": del_in_gt,
                       "search": cs,
                       "ok": bad == 0 and prof < 1e-4 and del_in_gt == 0
                       and cs["deleted_returned"] == 0 and cs["out_of_range_returned"] == 0})
            overlap[name] = np.array([len(top0[q] & set(gi[q, :10])) for q in range(len(Q))])
        rep["checks"][f"C4_seed{seed}"] = {"pass": all(x["ok"] for x in c4), "states": c4}
        c2 = {}
        for traj in ("id", "ood", "del"):
            names = sorted([n for n in S if S[n].get("trajectory") == traj], key=lambda n: S[n]["count"])
            seq = np.stack([np.full(len(Q), 10)] + [overlap[n] for n in names])
            viol = int((np.diff(seq, axis=0) > 0).sum())
            c2[traj] = {"states": names, "violations": viol,
                        "mean_overlap": [float(x.mean()) for x in seq]}
        rep["checks"][f"C2_seed{seed}"] = {"pass": all(v["violations"] == 0 for v in c2.values()), **c2}

        mu0 = base.astype(np.float64).mean(0)
        scale = np.sqrt(np.trace(np.cov(base.astype(np.float64), rowvar=False)))
        c3 = []
        for c in M["counts"]:
            dvals = {}
            for traj in ("id", "ood"):
                spec = S[f"{traj}_{c}"]["spec"]
                ins = L.read_fvecs(spec["inserted_vectors"])[:c].astype(np.float64)
                dvals[traj] = float(np.linalg.norm(ins.mean(0) - mu0) / scale)
            c3.append({"count": c, "D_id": dvals["id"], "D_ood": dvals["ood"], "ok": dvals["ood"] > dvals["id"]})
        rep["checks"][f"C3_seed{seed}"] = {"pass": all(x["ok"] for x in c3), "levels": c3}

        lo, so = S["S0"]["oracle_run"], m["s0_via_state"]["oracle_run"]
        lf, sf = S["S0"]["features_run"], m["s0_via_state"]["features_run"]
        same_curves = Path(lo, "oracle_curves.csv").read_bytes() == Path(so, "oracle_curves.csv").read_bytes()
        strip = lambda p: [ln.split(",", 1)[1] for ln in Path(p).read_text().splitlines()]  # noqa: E731
        same_labels = strip(Path(lo, "oracle_labels.csv")) == strip(Path(so, "oracle_labels.csv"))
        same_feats = strip(Path(lf, "features.csv")) == strip(Path(sf, "features.csv"))
        ok5 = same_curves and same_labels and same_feats
        nan_ok = True
        for name, st in S.items():
            fe = pd.read_csv(Path(st["features_run"]) / "features.csv")
            la = pd.read_csv(Path(st["oracle_run"]) / "oracle_labels.csv")
            nan_ok &= len(fe) == len(Q) and len(la) == len(Q) and np.isfinite(fe[FEATS].to_numpy()).all()
        rep["checks"][f"C5_seed{seed}"] = {"pass": ok5 and nan_ok, "curves_identical": same_curves,
                                           "labels_identical": same_labels, "features_identical": same_feats,
                                           "all_states_complete_no_nan": nan_ok}

        a = pool[M["anchor_pool_row"]].astype(np.float64)
        dq = np.sqrt(((Q.astype(np.float64) - a) ** 2).sum(1))
        rng = np.random.default_rng(20261002)
        c6 = []
        for c in M["counts"]:
            decay = 1 - overlap[f"ood_{c}"] / 10
            rho = stats.spearmanr(dq, decay).statistic
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(Q), len(Q))
                bs.append(stats.spearmanr(dq[i], decay[i]).statistic)
            lo_, hi_ = np.nanpercentile(bs, [2.5, 97.5])
            rid = stats.spearmanr(dq, 1 - overlap[f"id_{c}"] / 10).statistic
            c6.append({"count": c, "spearman_ood": float(rho), "ci95": [float(lo_), float(hi_)],
                       "spearman_id_descriptive": float(rid), "ok": bool(hi_ < 0)})
        rep["checks"][f"C6_seed{seed}"] = {"pass": all(x["ok"] for x in c6), "levels": c6}

        desc = {}
        for name, st in S.items():
            fe = pd.read_csv(Path(st["features_run"]) / "features.csv").sort_values("query_id")
            la = pd.read_csv(Path(st["oracle_run"]) / "oracle_labels.csv").sort_values("query_id")
            y = la["oracle_ef"].fillna(1e9).to_numpy()
            desc[name] = {
                "censored": int((la["reached"] == 0).sum()),
                "spearman_feature_vs_effort": {f: float(stats.spearmanr(fe[f], y).statistic) for f in FEATS},
                "mean_rel_change_knn_dist_vs_S0": float(((fe["knn_dist"].to_numpy() - f0["knn_dist"].to_numpy())
                                                         / f0["knn_dist"].to_numpy()).mean()),
                "mean_overlap_with_S0_top10": float(overlap[name].mean()) if name in overlap else 10.0}
        rep["descriptive"][f"seed{seed}"] = desc

    rep["ALL_PASS"] = all(v["pass"] for v in rep["checks"].values())
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    od = Path(P["output_dir"]) / f"phase5_synthetic_validation_{ts}"
    od.mkdir(parents=True)
    (od / "report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    for k, v in rep["checks"].items():
        print(k, "PASS" if v["pass"] else "FAIL")
    print("ALL_PASS:", rep["ALL_PASS"])
    print(od)
    return 0 if rep["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
