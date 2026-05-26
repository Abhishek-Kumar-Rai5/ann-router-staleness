"""Phase 4 router training — TRAINING SPLIT ONLY.

Implements the pre-registered rule (docs/notes.md, Phase 4 PRE-REGISTRATION):
5-fold stratified CV over the candidates, feasibility constraint on the
out-of-fold failure rate, matched-recall cost ratio as the score, ties to the
simplest model. The winner is refit on all training queries and frozen.

Test-split rows are never loaded: every input is filtered to split == "train"
at read time (router_lib.load_split_frame / load_curves).

Outputs in results/<experiment_id>/:
  cv_candidates.csv        per-candidate CV scores (primary + single-feature)
  oof_predictions.csv      out-of-fold tier predictions, one row per train query
  model_<role>.joblib      frozen fitted routers: primary, single_feature,
                           lid_ablation (+ SHA-256 in metadata.json)
  model_<role>.json        sklearn-free export (thresholds / coefficients)
  model_primary_rules.txt  human-readable rules if the primary is a tree
  metadata.json

Usage: python python/train_router.py configs/phase4_router_sift1m.yaml
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
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_lib as rl  # noqa: E402


def git_info():
    root = Path(__file__).resolve().parent.parent
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a],  # noqa: E731
                                    capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "--verify", "-q", "HEAD") or "none",
            "dirty": bool(run("status", "--porcelain"))}


def cv_evaluate(name, features, train, curves, tier_ef, probe_col, cfg):
    """Out-of-fold predictions and the pre-registered score for one candidate."""
    seed = cfg["router"]["seed"]
    X = train[features].to_numpy()
    y = train["tier_label"].to_numpy()
    oof = np.empty(len(train), dtype=object)
    skf = StratifiedKFold(cfg["router"]["cv_folds"], shuffle=True,
                          random_state=seed)
    for tr_idx, va_idx in skf.split(X, y):
        model = rl.make_candidate(name, seed)   # fresh, fit on 4/5 only
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict(X[va_idx])
    target = cfg["target_recall"]
    out = rl.route(oof, train["query_id"], train[probe_col], curves, tier_ef,
                   target)
    curve = rl.fixed_curve(curves, train["query_id"], target)
    mean_recall = out["recall"].mean()
    matched_ef = rl.cheapest_fixed_ef(curve, mean_recall)
    matched_fail = (float(curve.loc[curve["ef"] == matched_ef, "failure_rate"]
                          .iloc[0]) if matched_ef is not None else np.nan)
    fixed_cost = rl.interp_cost_at_recall(curve, mean_recall)
    return oof, {
        "name": name, "features": "+".join(features),
        "oof_accuracy": float((oof == y).mean()),
        "oof_mean_recall": float(mean_recall),
        "oof_failure_rate": float(out["failure"].mean()),
        "oof_mean_total_dc": float(out["total_dc"].mean()),
        "matched_fixed_ef": matched_ef,
        "matched_fixed_failure_rate": matched_fail,
        "fixed_cost_at_recall": fixed_cost,
        "score": float(fixed_cost / out["total_dc"].mean()),
        "feasible": bool(out["failure"].mean() <= matched_fail + rl.TOL),
        "oof_conservative_rate": float((rl.tier_error(oof, y) > 0).mean()),
        "oof_aggressive_rate": float((rl.tier_error(oof, y) < 0).mean()),
    }


def save_model(out_dir, role, model, features):
    path = out_dir / f"model_{role}.joblib"
    joblib.dump(model, path)
    export = rl.export_model(model, features)
    (out_dir / f"model_{role}.json").write_text(json.dumps(export, indent=1) + "\n")
    return {"path": path.name, "sha256": rl.sha256_file(path),
            "features": list(features), "export": f"model_{role}.json"}


def main() -> int:
    cfg_path = Path(sys.argv[1])
    raw = cfg_path.read_text()
    cfg = yaml.safe_load(raw)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exp_id = hashlib.sha256(raw.encode()).hexdigest()[:12] + "_" + ts
    out_dir = Path(cfg["output_dir"]) / exp_id
    out_dir.mkdir(parents=True)
    (out_dir / "config.yaml").write_text(raw)

    inp = cfg["inputs"]
    tier_ef = json.loads(Path(inp["tiers"]).read_text())["fixed_ef_baselines"]
    train = rl.load_split_frame(Path(inp["features_run"]) / "features.csv",
                                inp["strata"], "train")
    curves = rl.load_curves(Path(inp["oracle_run"]) / "oracle_curves.csv", "train")
    assert len(train) == 8000 and set(train["split"]) == {"train"}
    assert set(curves.recall.index) == set(train["query_id"])
    feats = cfg["router"]["features"]

    # 1. Primary candidates.
    rows, oofs = [], {}
    for name in cfg["router"]["candidates"]:
        oof, r = cv_evaluate(name, feats, train, curves, tier_ef,
                             "probe_distance_computations", cfg)
        r["role"] = "primary_candidate"
        rows.append(r)
        oofs[name] = oof
    prim = pd.DataFrame(rows)
    chosen = rl.select_candidate(prim, cfg["router"]["tie_tolerance"])

    # 2. Single-feature threshold router: feature chosen by the same CV score.
    srows = []
    for f in cfg["router"]["single_feature_candidates"]:
        _, r = cv_evaluate("SINGLE", [f], train, curves, tier_ef,
                           "probe_distance_computations", cfg)
        r["role"] = "single_feature_candidate"
        srows.append(r)
    single = pd.DataFrame(srows)
    single_feat = single.sort_values("score", ascending=False)["features"].iloc[0]

    # 3. LID ablation: chosen configuration + lid; pays both probes.
    train["probe_plus_lid_dc"] = (train["probe_distance_computations"]
                                  + train["lid_probe_distance_computations"])
    lid_feats = feats + [cfg["router"]["ablation_feature"]]
    _, lr_ = cv_evaluate(chosen, lid_feats, train, curves, tier_ef,
                         "probe_plus_lid_dc", cfg)
    lr_["role"] = "lid_ablation_cv"

    table = pd.concat([prim, single, pd.DataFrame([lr_])], ignore_index=True)
    table.to_csv(out_dir / "cv_candidates.csv", index=False)
    pd.DataFrame({"query_id": train["query_id"], "tier_label": train["tier_label"],
                  **{f"oof_{k}": v for k, v in oofs.items()}}).to_csv(
        out_dir / "oof_predictions.csv", index=False)

    # 4. Refit the winners on ALL training queries and freeze them.
    seed = cfg["router"]["seed"]
    y = train["tier_label"].to_numpy()
    models = {}
    primary = rl.make_candidate(chosen, seed).fit(train[feats].to_numpy(), y)
    models["primary"] = save_model(out_dir, "primary", primary, feats)
    sf = single_feat.split("+")
    single_m = rl.make_candidate("SINGLE", seed).fit(train[sf].to_numpy(), y)
    models["single_feature"] = save_model(out_dir, "single_feature", single_m, sf)
    lid_m = rl.make_candidate(chosen, seed).fit(train[lid_feats].to_numpy(), y)
    models["lid_ablation"] = save_model(out_dir, "lid_ablation", lid_m, lid_feats)
    models["primary"]["candidate"] = chosen
    models["single_feature"]["candidate"] = "SINGLE"
    models["lid_ablation"]["candidate"] = chosen
    models["lid_ablation"]["probe_cost"] = "core probe + LID probe"
    # Self-check: the sklearn-free export reproduces the fitted model.
    for role, m in (("primary", primary), ("single_feature", single_m),
                    ("lid_ablation", lid_m)):
        fs = models[role]["features"]
        exp = json.loads((out_dir / models[role]["export"]).read_text())
        assert (rl.predict_from_export(exp, train[fs].to_numpy())
                == m.predict(train[fs].to_numpy())).all(), role
    if chosen.startswith("DT"):
        exp = json.loads((out_dir / "model_primary.json").read_text())
        (out_dir / "model_primary_rules.txt").write_text(
            "\n".join(rl.tree_rules(exp)) + "\n")

    meta = {
        "experiment_id": exp_id, "phase": "4_router_training",
        "config_path": str(cfg_path), "config_yaml": raw, "git": git_info(),
        "seeds": {"router": seed, "cv": seed},
        "versions": {"python": platform.python_version(),
                     "sklearn": sklearn.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__},
        "train_rows": len(train), "test_rows_loaded": 0,
        "train_query_ids_sha256": hashlib.sha256(
            train["query_id"].to_numpy().tobytes()).hexdigest(),
        "input_sha256": {"tiers": rl.sha256_file(inp["tiers"]),
                         "strata": rl.sha256_file(inp["strata"]),
                         "features": rl.sha256_file(
                             Path(inp["features_run"]) / "features.csv")},
        "selection": {"chosen_primary": chosen,
                      "chosen_single_feature": single_feat,
                      "rule": "feasible (OOF failure <= matched fixed ef); max "
                              "matched-recall cost ratio; ties within "
                              f"{cfg['router']['tie_tolerance']} -> simplest "
                              f"{rl.SIMPLICITY}"},
        "models": models,
    }
    (out_dir / "metadata.json").write_text(rl.to_json(meta) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.drop(columns=["features"]).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"chosen primary: {chosen}; single feature: {single_feat}")
    print(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
