"""Phase 4b FINAL EVALUATION of the frozen routers (no fitting, no selection).

Modes (configs/phase4b/evaluate.yaml):
  dry_run_train          code check on TRAINING rows (in-sample; never a result)
  confirmation           PRIMARY: the sift_learn confirmation set (used once)
  old_test_second_look   SECONDARY / DISCLOSED SECOND LOOK: the old test split

Everything frozen comes from the training run (router specs, models, tau, B1)
and is hash-verified against its manifest before use. Statistical plan
(docs/phase4_bias_audit.md §10): query-clustered bootstrap (resample query
ids, all seeds of a query together), Wilcoxon on per-query seed-averaged
paired cost differences, exact McNemar per seed, one gate per S*, Phase 4 PASS
iff all three S* pass.

Usage: python python/phase4b_evaluate.py configs/phase4b/evaluate.yaml <mode>
"""

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase4b_train as T  # noqa: E402
import redesign_evidence_train as RE  # noqa: E402
import router4b_lib as rb  # noqa: E402
import router_lib as rl  # noqa: E402
import validate_audit_train as V  # noqa: E402

TOL = 1e-9


def verify_manifest(run: Path):
    man = json.loads((run / "manifest.json").read_text())
    bad = [k for k, h in man.items()
           if hashlib.sha256((run / k).read_bytes()).hexdigest() != h]
    if bad:
        raise SystemExit(f"STOP: frozen artefacts changed: {bad[:5]}")
    return len(man)


def descent_dc(index_path, qvecs):
    """Upper-layer descent distance count per query (independent replica)."""
    idx = V.HnswFile(index_path)
    vec = idx.vec.astype(np.float64)
    out = np.empty(len(qvecs))
    for t, qv in enumerate(qvecs):
        qd = qv.astype(np.float64)
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
                for cand, dd in zip(nb, d):
                    if dd < best:
                        best, cur, changed = float(dd), int(cand), True
        out[t] = c
    return out


def b2_expected_cost(level_by_ef, dc_by_ef, grid, target):
    """Hindsight fixed policy: cheapest two-ef mix on THIS query set reaching
    `target` (linear in the quality measure); inf if unreachable."""
    mix = rb.two_ef_mix(level_by_ef, grid, target)
    if mix is None:
        return float("inf"), None
    lo, hi = grid.index(mix["ef_lo"]), grid.index(mix["ef_hi"])
    return float((1 - mix["w_hi"]) * dc_by_ef[lo] + mix["w_hi"] * dc_by_ef[hi]), mix


def main() -> int:
    cfg_path, mode = Path(sys.argv[1]), sys.argv[2]
    raw = cfg_path.read_text()
    cfg = yaml.safe_load(raw)
    es = cfg["eval_sets"][mode]
    train_run = Path(cfg["train_run"])
    n_hashed = verify_manifest(train_run)
    tcfg = yaml.safe_load((train_run / "config.yaml").read_text())
    target = tcfg["target_recall"]
    cands = tcfg["candidates"]
    st = cfg["statistics"]
    gate = cfg["gate"]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "_dryrun_" if mode == "dry_run_train" else ""
    od = Path(cfg["output_dir"]) / f"{prefix}phase4b_eval_{mode}_{ts}"
    od.mkdir(parents=True)
    (od / "config.yaml").write_text(raw)
    label = {"dry_run_train": "DRY RUN ON TRAINING ROWS — in-sample code check, NOT a result",
             "confirmation": "PRIMARY CONFIRMATION-SET EVALUATION",
             "old_test_second_look": "SECONDARY / DISCLOSED SECOND-LOOK ANALYSIS"}[mode]

    seeds = sorted(int(s) for s in es["seeds"])
    per_seed = {}
    rows = []
    for seed in seeds:
        sc = es["seeds"][seed]
        feats = pd.read_csv(Path(sc["features_run"]) / "features.csv")
        feats = feats[feats["split"] == es["split_value"]].sort_values("query_id").reset_index(drop=True)
        labels = pd.read_csv(Path(sc["oracle_run"]) / "oracle_labels.csv")
        labels = labels[labels["split"] == es["split_value"]].sort_values("query_id").reset_index(drop=True)
        curves = rl.load_curves(Path(sc["oracle_run"]) / "oracle_curves.csv", es["split_value"])
        q = feats["query_id"].to_numpy()
        assert (labels["query_id"].to_numpy() == q).all() and set(curves.recall.index) == set(q)
        ometa = json.loads((Path(sc["oracle_run"]) / "metadata.json").read_text())
        fmeta = json.loads((Path(sc["features_run"]) / "metadata.json").read_text())
        assert ometa["seeds"]["hnsw_level_seed"] == seed == fmeta["seeds"]["hnsw_level_seed"]
        assert ometa["index"]["cache_path"] == fmeta["index"]["cache_path"]
        grid = [int(e) for e in curves.recall.columns]
        Rg = curves.recall.loc[q].to_numpy()
        Dg = curves.dc.loc[q].to_numpy().astype(float)
        Sg = Rg >= target - TOL
        M = {"primary": T.candidate_matrices(cands, 10, feats["probe_distance_computations"].to_numpy(float),
                                             curves, q, target),
             "lid": T.candidate_matrices(cands, 20, feats["lid_probe_distance_computations"].to_numpy(float),
                                         curves, q, target)}
        y = rb.candidate_oracle_index(M["primary"]["success"])
        # Grid (per-query minimum-ef) oracle and difficulty groups (frozen 18/48).
        reached = labels["reached"].to_numpy() == 1
        oef = labels["oracle_ef"].to_numpy()
        grid_or_dc = np.full(len(q), np.nan)
        grid_or_dc[reached] = Dg[np.where(reached)[0], [grid.index(int(e)) for e in oef[reached]]]
        diff = np.where(~reached | (np.nan_to_num(oef, nan=1e9) > cfg["difficulty"]["medium_max"]), "hard",
                        np.where(oef <= cfg["difficulty"]["easy_max"], "easy", "medium"))
        # Shared-descent sensitivity input (independent replica, label-free).
        qcfg = yaml.safe_load((Path(sc["oracle_run"]) / "config.yaml").read_text())
        qv = np.fromfile(qcfg["dataset"]["queries"], dtype=np.float32).reshape(-1, 129)[:, 1:][q]
        desc = descent_dc(ometa["index"]["cache_path"], qv)
        ps = {"q": q, "grid": grid, "Rg": Rg, "Dg": Dg, "Sg": Sg, "M": M, "y": y,
              "grid_or_dc": grid_or_dc, "reached": reached, "diff": diff, "descent": desc,
              "policies": {}}
        # Frozen routers.
        for spec_path in sorted((train_run / "routers" / str(seed)).glob("*/*.json")):
            spec = json.loads(spec_path.read_text())
            mdir = train_run / spec["model_dir"]
            models = [joblib.load(mdir / f"cand_{j}.joblib") for j in range(len(cands))]
            X = feats[spec["features"]].to_numpy(float)
            P = rb.predict_candidates(models, X)
            P_exp = np.maximum.accumulate(np.stack(
                [rb.proba_from_export(json.loads((mdir / f"cand_{j}.json").read_text()), X)
                 for j in range(len(cands))], 1), axis=1)
            assert np.allclose(P, P_exp, atol=1e-12), "export mismatch"
            choice, fb = rb.choose(P, spec["tau"])
            Mv = M["lid"] if spec["variant"] == "lid_ablation" else M["primary"]
            idx = np.arange(len(q))
            key = (spec["variant_group"], spec["contract"], float(spec["level"]))
            pol = {"cost": Mv["total"][idx, choice], "search": Mv["search"][idx, choice],
                   "probe": Mv["probe"], "success": Mv["success"][idx, choice],
                   "recall": Mv["recall"][idx, choice], "choice": choice, "fallback": fb,
                   "ef": np.array([10 if cands[c] == "probe" else cands[c] for c in choice]) if spec["variant"] != "lid_ablation"
                   else np.array([20 if cands[c] == "probe" else cands[c] for c in choice]),
                   "spec": spec}
            # Sensitivities: shared descent (routed search re-uses the probe's
            # descent) and probe-free.
            routed = np.array([cands[c] != "probe" for c in choice])
            pol["cost_shared_descent"] = pol["cost"] - np.where(routed, desc, 0.0)
            pol["cost_probe_free"] = pol["search"]
            # B1 (frozen two-ef mix, deterministic hash) and secondary pure B1.
            b1 = spec["b1"]
            ef_b1 = rb.assign_mix(b1, q, es["query_set_id"], b1["hash_seed"])
            cols = np.array([grid.index(e) for e in ef_b1])
            pol["b1"] = {"cost": Dg[idx, cols], "success": Sg[idx, cols], "recall": Rg[idx, cols],
                         "ef": ef_b1}
            jp = grid.index(b1["secondary_pure_smallest_ef"]["ef"])
            pol["b1_pure"] = {"cost": Dg[:, jp], "success": Sg[:, jp], "recall": Rg[:, jp]}
            ps["policies"][key] = pol
            for i in range(len(q)):
                rows.append((seed, int(q[i]), *key, spec["variant"], int(pol["ef"][i]),
                             float(pol["cost"][i]), float(pol["search"][i]), float(pol["probe"][i]),
                             bool(pol["success"][i]), float(pol["recall"][i]), bool(fb[i]),
                             int(ef_b1[i]), float(pol["b1"]["cost"][i]), bool(pol["b1"]["success"][i]),
                             float(pol["b1"]["recall"][i]), diff[i], float(desc[i])))
        per_seed[seed] = ps
    pd.DataFrame(rows, columns=["seed", "query_id", "group", "contract", "level", "variant",
                                "router_ef", "router_cost", "router_search", "router_probe",
                                "router_success", "router_recall", "router_fallback",
                                "b1_ef", "b1_cost", "b1_success", "b1_recall", "difficulty",
                                "descent_dc"]).to_csv(od / "per_query_rows.csv", index=False)

    # ---------------- analysis per (group, contract, level) -----------------
    n = len(per_seed[seeds[0]]["q"])
    assert all(len(per_seed[s]["q"]) == n and (per_seed[s]["q"] == per_seed[seeds[0]]["q"]).all() for s in seeds)
    rng = np.random.default_rng(st["bootstrap_seed"])
    boot_idx = rng.integers(0, n, size=(st["bootstrap_resamples"], n))
    report = {"label": label, "mode": mode, "train_run": str(train_run),
              "frozen_files_hash_verified": n_hashed, "n_queries": n, "seeds": seeds,
              "query_set_id": es["query_set_id"], "results": {}, "oracles": {}}

    for s in seeds:
        ps = per_seed[s]
        Mp = ps["M"]["primary"]
        ach = ps["y"] < len(cands)
        report["oracles"][str(s)] = {
            "1_grid_min_ef_oracle": {
                "reached": int(ps["reached"].sum()), "censored": int((~ps["reached"]).sum()),
                "mean_search_cost_reached": float(np.nanmean(ps["grid_or_dc"]))},
            "2_candidate_success_oracle": {
                "achievable": int(ach.sum()),
                "mean_total_cost_achievable": float(Mp["total"][np.where(ach)[0], ps["y"][ach]].mean())},
            "3_contract_oracle": {
                **{f"per_query_{lv}": RE.contract_oracle(Mp["total"], Mp["success"], lv)
                   for lv in tcfg["per_query_levels"]},
                **{f"mean_recall_{lv}": float(Mp["total"][np.arange(n), V.lagrange(
                    Mp["recall"], Mp["total"], lv)].mean()) for lv in tcfg["mean_recall_levels"]}},
            "difficulty_counts": {g: int((ps["diff"] == g).sum()) for g in ("easy", "medium", "hard")},
        }

    W_boot = np.stack([np.bincount(bi, minlength=n) for bi in boot_idx]).astype(float)
    boot_curves = {}
    for s in seeds:
        for contract_, Qm in (("per_query", per_seed[s]["Sg"].astype(float)),
                              ("mean_recall", per_seed[s]["Rg"])):
            boot_curves[(s, contract_)] = (W_boot @ Qm / n, W_boot @ per_seed[s]["Dg"] / n)

    keys = sorted(per_seed[seeds[0]]["policies"].keys())
    for key in keys:
        grp, contract, lvl = key
        qkey = "success" if contract == "per_query" else "recall"
        A = {f: np.stack([per_seed[s]["policies"][key][f] for s in seeds]).astype(float)
             for f in ("cost", "success", "recall", "cost_shared_descent", "cost_probe_free", "search", "probe")}
        B = {f: np.stack([per_seed[s]["policies"][key]["b1"][f] for s in seeds]).astype(float)
             for f in ("cost", "success", "recall")}
        Bp = {f: np.stack([per_seed[s]["policies"][key]["b1_pure"][f] for s in seeds]).astype(float)
              for f in ("cost", "success", "recall")}
        # Per-seed curves for B2 (quality per grid ef, cost per grid ef).
        Qg = [per_seed[s]["Sg"].astype(float) if contract == "per_query" else per_seed[s]["Rg"] for s in seeds]
        Dgs = [per_seed[s]["Dg"] for s in seeds]
        grid = per_seed[seeds[0]]["grid"]

        def b2_pooled_point():
            costs = []
            for k_ in range(len(seeds)):
                c, _ = b2_expected_cost(Qg[k_].mean(0), Dgs[k_].mean(0), grid,
                                        float(A[qkey][k_].mean()))
                costs.append(c)
            return float(np.mean(costs)), costs

        b2_cost, b2_seed = b2_pooled_point()
        pt = {"router_cost": float(A["cost"].mean()), "b1_cost": float(B["cost"].mean()),
              "abs_saving_vs_b1": float(B["cost"].mean() - A["cost"].mean()),
              "saving_vs_b1": float(1 - A["cost"].mean() / B["cost"].mean()),
              "router_success": float(A["success"].mean()), "b1_success": float(B["success"].mean()),
              "success_diff": float(A["success"].mean() - B["success"].mean()),
              "router_mean_recall": float(A["recall"].mean()), "b1_mean_recall": float(B["recall"].mean()),
              "mean_recall_diff": float(A["recall"].mean() - B["recall"].mean()),
              "router_failure_rate": float(1 - A["success"].mean()),
              "b1_failure_rate": float(1 - B["success"].mean()),
              "router_median_cost": float(np.median(A["cost"])), "b1_median_cost": float(np.median(B["cost"])),
              "b2_cost": b2_cost, "saving_vs_b2": float(1 - A["cost"].mean() / b2_cost),
              "b1_pure": {"cost": float(Bp["cost"].mean()), "success": float(Bp["success"].mean()),
                          "mean_recall": float(Bp["recall"].mean()),
                          "saving_router_vs": float(1 - A["cost"].mean() / Bp["cost"].mean())},
              "sensitivity_shared_descent_saving_vs_b1": float(1 - A["cost_shared_descent"].mean() / B["cost"].mean()),
              "sensitivity_probe_free_saving_vs_b1": float(1 - A["cost_probe_free"].mean() / B["cost"].mean()),
              "router_fallback_rate": float(np.mean([per_seed[s]["policies"][key]["fallback"].mean() for s in seeds]))}
        # Query-clustered bootstrap (resample query ids; every seed of a query
        # moves together). W[b, i] = multiplicity of query i in resample b.
        W = W_boot
        S_ = len(seeds)
        rc = (W @ A["cost"].T).sum(1) / (S_ * n)
        bc = (W @ B["cost"].T).sum(1) / (S_ * n)
        bs = {"saving_vs_b1": 1 - rc / bc, "abs_saving_vs_b1": bc - rc,
              "success_diff": (W @ (A["success"] - B["success"]).T).sum(1) / (S_ * n),
              "mean_recall_diff": (W @ (A["recall"] - B["recall"]).T).sum(1) / (S_ * n)}
        rq = (W @ A[qkey].T) / n                     # (B, seeds) router quality
        b2c = np.zeros(len(W))
        for k_, s in enumerate(seeds):
            QW, DW = boot_curves[(s, contract)]
            b2c += np.array([b2_expected_cost(QW[b], DW[b], grid, float(rq[b, k_]))[0]
                             for b in range(len(W))])
        bs["saving_vs_b2"] = 1 - rc / (b2c / S_)
        ci = {k: [float(x) for x in np.percentile(v, [2.5, 97.5])] for k, v in bs.items()}
        # Wilcoxon on per-query seed-averaged paired differences.
        wil = rl.wilcoxon_paired(A["cost"].mean(0), B["cost"].mean(0))
        # Per seed + exact McNemar.
        per = {}
        for k_, s in enumerate(seeds):
            b_ = int(((A["success"][k_] == 1) & (B["success"][k_] == 0)).sum())
            c_ = int(((A["success"][k_] == 0) & (B["success"][k_] == 1)).sum())
            mc = stats.binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ > 0 else 1.0
            per[str(s)] = {"router_cost": float(A["cost"][k_].mean()), "b1_cost": float(B["cost"][k_].mean()),
                           "saving_vs_b1": float(1 - A["cost"][k_].mean() / B["cost"][k_].mean()),
                           "router_success": float(A["success"][k_].mean()),
                           "b1_success": float(B["success"][k_].mean()),
                           "success_diff": float(A["success"][k_].mean() - B["success"][k_].mean()),
                           "router_mean_recall": float(A["recall"][k_].mean()),
                           "b1_mean_recall": float(B["recall"][k_].mean()),
                           "b2_cost": b2_seed[k_], "mcnemar_router_only_success": b_,
                           "mcnemar_b1_only_success": c_, "mcnemar_exact_p": float(mc),
                           "b1_mix": {k: per_seed[s]["policies"][key]["spec"]["b1"][k] for k in ("ef_lo", "ef_hi", "w_hi")},
                           "tau": per_seed[s]["policies"][key]["spec"]["tau"],
                           "family": per_seed[s]["policies"][key]["spec"]["family"],
                           "variant": per_seed[s]["policies"][key]["spec"]["variant"]}
        # Regret decomposition (per seed, then pooled mean).
        reg = []
        for s in seeds:
            ps = per_seed[s]
            pol = ps["policies"][key]
            Mv = ps["M"]["lid"] if pol["spec"]["variant"] == "lid_ablation" else ps["M"]["primary"]
            yv = rb.candidate_oracle_index(Mv["success"])

            def ofn(metrics, Mv=Mv, contract=contract):
                if contract == "per_query":
                    r = RE.contract_oracle(Mv["total"], Mv["success"], metrics["success_rate"])
                    return r["mean_cost"] if r["feasible"] else None
                j = V.lagrange(Mv["recall"], Mv["total"], metrics["mean_recall"])
                return None if j is None else float(Mv["total"][np.arange(n), j].mean())
            m = T.policy_metrics(Mv, pol["choice"], yv, pol["fallback"], ofn)
            m["legacy_regret_total_minus_grid_oracle_dc"] = float(np.nanmean(pol["cost"] - ps["grid_or_dc"]))
            reg.append(m)
        regret = {k: float(np.mean([r[k] for r in reg])) for k in
                  ("avoidable_failure_rate", "unavoidable_failure_rate", "overspend_on_successes_mean",
                   "underspend_on_avoidable_failures_mean", "mean_probe_cost", "contract_regret",
                   "contract_oracle_cost_at_realised_quality", "legacy_regret_total_minus_grid_oracle_dc")}
        # Difficulty groups (descriptive).
        dg = {}
        for g in ("easy", "medium", "hard"):
            msk = np.stack([per_seed[s]["diff"] == g for s in seeds])
            dg[g] = {"n_query_seed": int(msk.sum()), "router_cost": float(A["cost"][msk].mean()),
                     "b1_cost": float(B["cost"][msk].mean()), "router_success": float(A["success"][msk].mean()),
                     "b1_success": float(B["success"][msk].mean())}
        res = {"point": pt, "ci95": ci, "wilcoxon_seed_averaged": wil, "per_seed": per,
               "regret": regret, "difficulty": dg}
        # Frozen gate (per-query family gates; mean family reported identically).
        qdiff_ci = ci["success_diff"] if contract == "per_query" else ci["mean_recall_diff"]
        qdiff_seed = [per[str(s)]["success_diff" if contract == "per_query" else "router_mean_recall"]
                      - (0 if contract == "per_query" else per[str(s)]["b1_mean_recall"]) for s in seeds]
        crit = {
            "1_saving_ge_10pct_ci_lb_gt_0_wilcoxon_p_lt_0.01":
                pt["saving_vs_b1"] >= gate["min_saving"] and ci["saving_vs_b1"][0] > 0
                and wil["p_value"] < gate["alpha"],
            "2_quality_noninferior_ci_lb_ge_-0.01": qdiff_ci[0] >= gate["quality_margin"],
            "3_saving_vs_b2_ci_lb_gt_0": ci["saving_vs_b2"][0] > 0,
            "4_every_seed_saving_gt_0_and_quality_diff_ge_-0.01":
                all(per[str(s)]["saving_vs_b1"] > 0 for s in seeds)
                and all(d >= gate["quality_margin"] for d in qdiff_seed),
        }
        calib = {str(s): {"router_quality": per[str(s)]["router_success" if contract == "per_query" else "router_mean_recall"],
                          "b1_quality": per[str(s)]["b1_success" if contract == "per_query" else "b1_mean_recall"]}
                 for s in seeds}
        shift_flag = all(v["router_quality"] < lvl - gate["calibration_flag_tolerance"]
                         and v["b1_quality"] < lvl - gate["calibration_flag_tolerance"] for v in calib.values())
        res["gate"] = {"criteria": crit, "pass": all(crit.values()),
                       "gating": grp == "primary" and contract == "per_query",
                       "calibration_report": calib, "query_set_shift_flag": shift_flag}
        if contract == "mean_recall" and abs(lvl - 0.95) < 1e-12:
            res["note"] = ("Frozen candidate-set limitation (ex-ante feasibility): with the frozen "
                           "ladder, even perfect information saves only ~0.7-1.5% vs B1 at R=0.95.")
        report["results"][f"{grp}|{contract}|{lvl}"] = res

    prim = [report["results"][f"primary|per_query|{lv}"]["gate"]["pass"] for lv in tcfg["per_query_levels"]]
    report["phase4_gate"] = {"per_S_star": dict(zip(map(str, tcfg["per_query_levels"]), prim)),
                             "PASS": all(prim), "rule": "PASS iff gate passes at all three S*"}
    report["generated_utc"] = ts
    report["git"] = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"],
                                   capture_output=True, text=True).stdout.strip() or "none"
    (od / "eval_report.json").write_text(rl.to_json(report) + "\n")
    man = {str(p.relative_to(od)): rl.sha256_file(p) for p in sorted(od.rglob("*")) if p.is_file()}
    (od / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(label)
    for lv in tcfg["per_query_levels"]:
        r = report["results"][f"primary|per_query|{lv}"]
        print(f"S*={lv}: router {r['point']['router_cost']:.0f} vs B1 {r['point']['b1_cost']:.0f} "
              f"saving {r['point']['saving_vs_b1']:+.1%} CI {r['ci95']['saving_vs_b1']} "
              f"succ {r['point']['router_success']:.4f}/{r['point']['b1_success']:.4f} "
              f"-> gate {'PASS' if r['gate']['pass'] else 'FAIL'}")
    print("PHASE 4 GATE:", "PASS" if report["phase4_gate"]["PASS"] else "FAIL")
    print(od)
    return 0


if __name__ == "__main__":
    sys.exit(main())
