"""Phase 4b: train and FREEZE the routers — TRAINING SPLIT ONLY.

Reads only split == "train" rows (asserted). Never touches the old test split's
labels or curves, and the confirmation set does not exist yet. Everything is
driven by configs/phase4b/router_train.yaml.

Outputs: results/phase4b_train_<cfg-hash>_<ts>/
  routers/<seed>/<variant>/<contract>_<level>.json   frozen router specs
  models/<seed>/<variant>/<family>/cand_<j>.{joblib,json}
  folds/<seed>.csv                                   CV fold assignment
  diagnostics.json, diagnostics_rows.csv             TRAINING/CV diagnostics
  manifest.json (SHA-256 of every file), metadata.json, TRAINING_RECORD.md

Usage: python python/phase4b_train.py configs/phase4b/router_train.yaml
"""

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redesign_evidence_train as RE  # noqa: E402
import router4b_lib as rb  # noqa: E402
import router_lib as rl  # noqa: E402
import validate_audit_train as V  # noqa: E402

TOL = 1e-9


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def load_seed(seed_cfg):
    """Train rows only, for one index seed."""
    feats = pd.read_csv(Path(seed_cfg["features_run"]) / "features.csv")
    feats = feats[feats["split"] == "train"].sort_values("query_id").reset_index(drop=True)
    labels = pd.read_csv(Path(seed_cfg["oracle_run"]) / "oracle_labels.csv")
    labels = labels[labels["split"] == "train"].sort_values("query_id").reset_index(drop=True)
    curves = rl.load_curves(Path(seed_cfg["oracle_run"]) / "oracle_curves.csv", "train")
    assert set(feats["split"]) == {"train"} == set(labels["split"]) and len(feats) == 8000
    assert (feats["query_id"].to_numpy() == labels["query_id"].to_numpy()).all()
    return feats, labels, curves


def candidate_matrices(cands, probe_ef, probe_cost, curves, q, target):
    grid = [int(e) for e in curves.recall.columns]
    R = curves.recall.loc[q].to_numpy()
    D = curves.dc.loc[q].to_numpy().astype(float)
    assert np.array_equal(probe_cost, D[:, grid.index(probe_ef)]), "probe != ef search"
    ef_of = [probe_ef if c == "probe" else c for c in cands]
    rec = np.stack([R[:, grid.index(e)] for e in ef_of], 1)
    search = np.stack([np.zeros(len(q)) if c == "probe" else D[:, grid.index(c)]
                       for c in cands], 1)
    return {"recall": rec, "success": rec >= target - TOL, "search": search,
            "total": search + probe_cost[:, None], "probe": probe_cost}


def policy_metrics(M, choice, y, fallback, contract_oracle_fn):
    idx = np.arange(len(choice))
    s = M["success"][idx, choice]
    ach = y < M["success"].shape[1]
    cost = M["total"][idx, choice]
    srch = M["search"][idx, choice]
    yy = np.minimum(y, M["search"].shape[1] - 1)
    oracle_search = M["search"][idx, yy]
    over = (s & ach)
    avoid_fail = (~s) & ach
    out = {
        "success_rate": float(s.mean()), "mean_recall": float(M["recall"][idx, choice].mean()),
        "failure_rate": float(1 - s.mean()), "mean_total_cost": float(cost.mean()),
        "median_total_cost": float(np.median(cost)), "mean_search_cost": float(srch.mean()),
        "mean_probe_cost": float(M["probe"].mean()), "fallback_rate": float(fallback.mean()),
        "avoidable_failure_rate": float(avoid_fail.mean()),
        "unavoidable_failure_rate": float(((~s) & ~ach).mean()),
        "overspend_on_successes_mean": float((srch - oracle_search)[over].mean()) if over.any() else 0.0,
        "underspend_on_avoidable_failures_mean":
            float((oracle_search - srch)[avoid_fail].mean()) if avoid_fail.any() else 0.0,
        "choice_shares": np.bincount(choice, minlength=M["total"].shape[1]).tolist(),
    }
    co = contract_oracle_fn(out)
    out["contract_oracle_cost_at_realised_quality"] = co
    out["contract_regret"] = out["mean_total_cost"] - co if co is not None else None
    return out


def main() -> int:
    cfg_path = Path(sys.argv[1])
    raw = cfg_path.read_text()
    cfg = yaml.safe_load(raw)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    od = Path(cfg["output_dir"]) / f"phase4b_train_{hashlib.sha256(raw.encode()).hexdigest()[:12]}_{ts}"
    od.mkdir(parents=True)
    (od / "config.yaml").write_text(raw)
    cands = cfg["candidates"]
    m = len(cands)
    assert cands[-1] == cfg["fallback_candidate"]
    target = cfg["target_recall"]
    order = cfg["simplicity_order"]
    tie = cfg["tie_tolerance_relative"]
    fams = order
    diag = {"kind": "TRAINING/CV DIAGNOSTICS ONLY — not final results",
            "seeds": {}}
    rows = []

    for seed_s, scfg in cfg["seeds"].items():
        seed = int(seed_s)
        feats, labels, curves = load_seed(scfg)
        ometa = json.loads((Path(scfg["oracle_run"]) / "metadata.json").read_text())
        assert ometa["seeds"]["hnsw_level_seed"] == seed
        q = feats["query_id"].to_numpy()
        grid = [int(e) for e in curves.recall.columns]
        Rg = curves.recall.loc[q].to_numpy()
        Dg = curves.dc.loc[q].to_numpy().astype(float)
        Sg = Rg >= target - TOL
        sd = {"variants": {}, "b1": {}, "oracles": {}}

        # Variant matrices and labels.
        vdefs = {}
        p = cfg["variants"]["primary"]
        vdefs["primary"] = (p["features"], p["probe_ef"], p["probe_cost"])
        p = cfg["variants"]["lid_ablation"]
        vdefs["lid_ablation"] = (p["features"], p["probe_ef"], p["probe_cost"])
        p = cfg["variants"]["single_feature"]
        for f in p["candidates"]:
            vdefs[f"single_{f}"] = ([f], p["probe_ef"], p["probe_cost"])
        mats, ys = {}, {}
        for v, (fs, pef, pcol) in vdefs.items():
            mats[v] = candidate_matrices(cands, pef, feats[pcol].to_numpy(float),
                                         curves, q, target)
            ys[v] = rb.candidate_oracle_index(mats[v]["success"])

        # Folds: stratified on the PRIMARY candidate-oracle index (same folds
        # for every variant of this seed).
        skf = StratifiedKFold(cfg["cv"]["folds"], shuffle=cfg["cv"]["shuffle"],
                              random_state=cfg["cv"]["seed"])
        fold = np.empty(len(q), int)
        for k, (_, va) in enumerate(skf.split(np.zeros(len(q)), ys["primary"])):
            fold[va] = k
        (od / "folds").mkdir(exist_ok=True)
        pd.DataFrame({"query_id": q, "fold": fold, "y_primary": ys["primary"]}).to_csv(
            od / "folds" / f"{seed}.csv", index=False)
        y_counts = np.bincount(ys["primary"], minlength=m + 1).tolist()

        # Contract oracles (per variant, needed for regret).
        def oracle_fn(v, contract):
            M = mats[v]

            def fn(metrics):
                if contract == "per_query":
                    r = RE.contract_oracle(M["total"], M["success"], metrics["success_rate"])
                    return r["mean_cost"] if r["feasible"] else None
                j = V.lagrange(M["recall"], M["total"], metrics["mean_recall"])
                return None if j is None else float(M["total"][np.arange(len(q)), j].mean())
            return fn

        for v, (fs, pef, pcol) in vdefs.items():
            X = feats[fs].to_numpy(float)
            y = ys[v]
            M = mats[v]
            vd = {"features": fs, "probe_ef": pef, "probe_cost_column": pcol,
                  "y_counts": np.bincount(y, minlength=m + 1).tolist(), "families": {}}
            for fam in fams:
                P_oof = np.zeros((len(q), m))
                one_class_folds = 0
                for k in range(cfg["cv"]["folds"]):
                    tr, va = fold != k, fold == k
                    models, oc = rb.fit_candidate_models(fam, cfg, X[tr], y[tr], m)
                    one_class_folds += oc
                    P_oof[va] = rb.predict_candidates(models, X[va])
                final_models, oc_final = rb.fit_candidate_models(fam, cfg, X, y, m)
                P_tr = rb.predict_candidates(final_models, X)
                mdir = od / "models" / str(seed) / v / fam
                mdir.mkdir(parents=True)
                for j, mod in enumerate(final_models):
                    joblib.dump(mod, mdir / f"cand_{j}.joblib")
                    e = rb.export_binary(mod, fs)
                    (mdir / f"cand_{j}.json").write_text(json.dumps(e) + "\n")
                    assert np.allclose(rb.proba_from_export(e, X), rb.proba1(mod, X),
                                       atol=1e-12)
                fd = {"one_class_fits_cv": int(one_class_folds),
                      "one_class_fits_final": int(oc_final), "levels": {}}
                for contract, levels, qual in (
                        ("per_query", cfg["per_query_levels"], M["success"].astype(float)),
                        ("mean_recall", cfg["mean_recall_levels"], M["recall"])):
                    for lvl in levels:
                        tau, achieved = rb.calibrate_tau(P_oof, qual, lvl, cfg["tau_grid_step"])
                        if tau is None:
                            raise SystemExit(
                                f"STOP: no tau reaches {contract} {lvl} (seed {seed}, {v}, "
                                f"{fam}); best OOF {achieved:.4f}")
                        c_oof, fb_oof = rb.choose(P_oof, tau)
                        c_tr, fb_tr = rb.choose(P_tr, tau)
                        ofn = oracle_fn(v, contract)
                        oof = policy_metrics(M, c_oof, y, fb_oof, ofn)
                        trn = policy_metrics(M, c_tr, y, fb_tr, ofn)
                        quality = oof["success_rate"] if contract == "per_query" else oof["mean_recall"]
                        fd["levels"][f"{contract}_{lvl}"] = {
                            "tau": tau, "oof": oof, "train_in_sample": trn,
                            "feasible": quality >= lvl - 1e-12}
                        rows.append({"seed": seed, "variant": v, "family": fam,
                                     "contract": contract, "level": lvl, "tau": tau,
                                     "oof_cost": oof["mean_total_cost"],
                                     "oof_success": oof["success_rate"],
                                     "oof_mean_recall": oof["mean_recall"],
                                     "oof_fallback": oof["fallback_rate"],
                                     "train_cost": trn["mean_total_cost"],
                                     "train_success": trn["success_rate"],
                                     "one_class_cv": one_class_folds,
                                     "one_class_final": oc_final})
                vd["families"][fam] = fd
            sd["variants"][v] = vd

        # B1 (training-only two-ef mixture) and the three oracle references.
        for contract, levels, lvl_by_ef, qual_g in (
                ("per_query", cfg["per_query_levels"], Sg.mean(0), Sg.astype(float)),
                ("mean_recall", cfg["mean_recall_levels"], Rg.mean(0), Rg)):
            for lvl in levels:
                mix = rb.two_ef_mix(lvl_by_ef, grid, lvl)
                ef_assigned = rb.assign_mix(mix, q, cfg["b1"]["train_query_set_id"],
                                            cfg["b1"]["seed"])
                cols = [grid.index(e) for e in ef_assigned]
                idx = np.arange(len(q))
                j_lo, j_hi = grid.index(mix["ef_lo"]), grid.index(mix["ef_hi"])
                pure_j = int(np.argmax(lvl_by_ef >= lvl - 1e-12))
                sd["b1"][f"{contract}_{lvl}"] = {
                    **mix, "hash_seed": cfg["b1"]["seed"],
                    "expected_quality_train": float((1 - mix["w_hi"]) * lvl_by_ef[j_lo]
                                                    + mix["w_hi"] * lvl_by_ef[j_hi]),
                    "expected_cost_train": float((1 - mix["w_hi"]) * Dg[:, j_lo].mean()
                                                 + mix["w_hi"] * Dg[:, j_hi].mean()),
                    "realised_quality_train": float(qual_g[idx, cols].mean()),
                    "realised_success_train": float(Sg[idx, cols].mean()),
                    "realised_mean_recall_train": float(Rg[idx, cols].mean()),
                    "realised_cost_train": float(Dg[idx, cols].mean()),
                    "share_assigned_hi": float((ef_assigned == mix["ef_hi"]).mean()),
                    "secondary_pure_smallest_ef": {
                        "ef": grid[pure_j], "quality_train": float(lvl_by_ef[pure_j]),
                        "cost_train": float(Dg[:, pure_j].mean())}}
        oe = labels["oracle_ef"].to_numpy()
        assert labels["reached"].all(), "train split has no censored queries"
        Mp = mats["primary"]
        yp = ys["primary"]
        achp = yp < m
        sd["oracles"] = {
            "1_grid_min_ef_oracle": {
                "mean_search_cost": float(Dg[np.arange(len(q)), [grid.index(int(e)) for e in oe]].mean()),
                "success_rate": 1.0},
            "2_candidate_success_oracle_primary": {
                "achievable_rate": float(achp.mean()),
                "mean_total_cost_achievable": float(Mp["total"][np.where(achp)[0], yp[achp]].mean()),
                "label_counts": y_counts},
            "3_contract_oracle_primary": {
                **{f"per_query_{s}": RE.contract_oracle(Mp["total"], Mp["success"], s)
                   for s in cfg["per_query_levels"]},
                **{f"mean_recall_{r}": float(Mp["total"][np.arange(len(q)), V.lagrange(
                    Mp["recall"], Mp["total"], r)].mean()) for r in cfg["mean_recall_levels"]}},
        }
        diag["seeds"][str(seed)] = sd

    # ---- selection and freezing --------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(od / "diagnostics_rows.csv", index=False)
    frozen = []
    for (seed, contract, lvl), g in df.groupby(["seed", "contract", "level"]):
        for vgroup, vfilter in (("primary", g["variant"] == "primary"),
                                ("lid_ablation", g["variant"] == "lid_ablation"),
                                ("single_feature", g["variant"].str.startswith("single_"))):
            sub = g[vfilter]
            qcol = "oof_success" if contract == "per_query" else "oof_mean_recall"
            cand_rows = [{"name": r.family, "variant": r.variant, "cost": r.oof_cost,
                          "feasible": getattr(r, qcol) >= lvl - 1e-12}
                         for r in sub.itertuples()]
            feas = [r for r in cand_rows if r["feasible"]]
            if not feas:
                raise SystemExit(f"STOP: no feasible family for {seed} {contract} {lvl} {vgroup}")
            best = min(r["cost"] for r in feas)
            near = [r for r in feas if r["cost"] <= best * (1 + tie) + 1e-12]
            pick = sorted(near, key=lambda r: (order.index(r["name"]), r["cost"]))[0]
            v, fam = pick["variant"], pick["name"]
            sd = diag["seeds"][str(seed)]
            lv = sd["variants"][v]["families"][fam]["levels"][f"{contract}_{lvl}"]
            vdef = sd["variants"][v]
            spec = {
                "status": "FROZEN", "seed": int(seed), "variant_group": vgroup,
                "variant": v, "contract": contract, "level": float(lvl),
                "family": fam, "hyperparameters": cfg["families"][fam],
                "random_state": cfg["families"]["random_state"],
                "features": vdef["features"], "probe_ef": vdef["probe_ef"],
                "probe_cost_column": vdef["probe_cost_column"],
                "candidates": cands, "fallback_candidate": cfg["fallback_candidate"],
                "decision_rule": "cheapest candidate with cummax P(y<=c|x) >= tau; "
                                 "else fallback (highest candidate)",
                "probability_calibration": cfg["probability_calibration"],
                "tau": lv["tau"], "tau_rule": f"smallest tau on {cfg['tau_grid_step']} grid "
                                             "with OOF quality >= level",
                "cv": cfg["cv"], "model_dir": f"models/{seed}/{v}/{fam}",
                "one_class_fits_cv": sd["variants"][v]["families"][fam]["one_class_fits_cv"],
                "one_class_fits_final": sd["variants"][v]["families"][fam]["one_class_fits_final"],
                "b1": sd["b1"][f"{contract}_{lvl}"],
                "selection_table": cand_rows,
            }
            rdir = od / "routers" / str(seed) / vgroup
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / f"{contract}_{lvl}.json").write_text(rl.to_json(spec) + "\n")
            frozen.append({"seed": int(seed), "group": vgroup, "contract": contract,
                           "level": float(lvl), "variant": v, "family": fam,
                           "tau": lv["tau"], "oof": lv["oof"], "train": lv["train_in_sample"],
                           "b1": sd["b1"][f"{contract}_{lvl}"]})
    diag["frozen_routers"] = frozen
    (od / "diagnostics.json").write_text(rl.to_json(diag) + "\n")

    meta = {
        "phase": "4b_router_training", "status": "routers FROZEN; no evaluation performed",
        "config_path": str(cfg_path), "config_yaml": raw,
        "git": {"commit": sh("git", "rev-parse", "--verify", "-q", "HEAD") or "none",
                "dirty": bool(sh("git", "status", "--porcelain"))},
        "random_seeds": {"index_seeds": [42, 43, 44], "query_split": 20261001,
                         "cv_and_models": cfg["cv"]["seed"], "b1_hash": cfg["b1"]["seed"]},
        "hnsw": {"M": 16, "ef_construction": 200, "build_threads": 1,
                 "probe": {"k": 10, "ef": 10}, "lid_probe": {"k": 20, "ef": 20}},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "hnswlib_commit": "3f3429661187e4c24a490a0f148fc6bc89042b3d"},
        "hardware": {"cpu": sh("bash", "-c", "lscpu | grep 'Model name' | sed 's/.*: *//'"),
                     "cores": sh("nproc"),
                     "mem": sh("bash", "-c", "free -g | awk '/Mem/{print $2\" GiB\"}'")},
        "timing_methodology": "No latency measured. Cost = distance computations "
                              "(CountingL2Space), deterministic.",
        "data_access": {"splits_read": ["train"], "old_test_split_read": False,
                        "confirmation_set_exists": False},
    }
    (od / "metadata.json").write_text(rl.to_json(meta) + "\n")
    manifest = {str(p.relative_to(od)): rl.sha256_file(p)
                for p in sorted(od.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (od / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(od)
    return 0


if __name__ == "__main__":
    sys.exit(main())
