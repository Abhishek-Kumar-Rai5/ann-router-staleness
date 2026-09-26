"""One-time test-split evaluation of the frozen Phase 4 router. Nothing here feeds
back into the router: no fitting, no thresholds, no selection.

Usage: python python/evaluate_router.py configs/phase4_router_sift1m.yaml <train_run>
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_lib as rl  # noqa: E402

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def summarize(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    out = {"n": int(len(df))}
    for col, key in (("recall", "mean_recall"), ("total_dc", "mean_total_dc")):
        out[key] = float(df[col].mean())
        out[key + "_ci95"] = rl.bootstrap_ci(df[col], n_boot, seed)
    out["mean_recall_id"] = float(df["recall_id"].mean())
    out["failure_rate"] = float(df["failure"].mean())
    out["failure_rate_ci95"] = rl.bootstrap_ci(df["failure"].astype(float),
                                               n_boot, seed)
    out["median_total_dc"] = float(df["total_dc"].median())
    out["mean_search_dc"] = float(df["search_dc"].mean())
    out["mean_probe_dc"] = float(df["probe_dc"].mean())
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int) -> dict:
    a = a.set_index("query_id").sort_index()
    b = b.set_index("query_id").loc[a.index]
    d_cost = a["total_dc"] - b["total_dc"]
    d_rec = a["recall"] - b["recall"]
    d_fail = a["failure"].astype(float) - b["failure"].astype(float)
    return {
        "mean_cost_diff": float(d_cost.mean()),
        "mean_cost_diff_ci95": rl.bootstrap_ci(d_cost, n_boot, seed),
        "relative_cost_saving": float(-d_cost.mean() / b["total_dc"].mean()),
        "cost_wilcoxon": rl.wilcoxon_paired(a["total_dc"], b["total_dc"]),
        "mean_recall_diff": float(d_rec.mean()),
        "mean_recall_diff_ci95": rl.bootstrap_ci(d_rec, n_boot, seed),
        "recall_wilcoxon": rl.wilcoxon_paired(a["recall"], b["recall"]),
        "failure_rate_diff": float(d_fail.mean()),
        "failure_rate_diff_ci95": rl.bootstrap_ci(d_fail, n_boot, seed),
        "queries_router_cheaper": int((d_cost < 0).sum()),
        "queries_router_costlier": int((d_cost > 0).sum()),
        "router_dominates": bool(a["recall"].mean() >= b["recall"].mean()
                                 and a["total_dc"].mean() <= b["total_dc"].mean()),
        "router_dominated": bool(a["recall"].mean() <= b["recall"].mean()
                                 and a["total_dc"].mean() >= b["total_dc"].mean()),
    }


def main() -> int:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    run = Path(sys.argv[2])
    out = run / "test_eval"
    out.mkdir(exist_ok=False)  # evaluate once, never overwrite
    meta = json.loads((run / "metadata.json").read_text())
    inp, target = cfg["inputs"], cfg["target_recall"]
    nb, seed = (cfg["evaluation"]["bootstrap_resamples"],
                cfg["evaluation"]["bootstrap_seed"])
    tier_ef = json.loads(Path(inp["tiers"]).read_text())["fixed_ef_baselines"]

    models = {}
    for role, m in meta["models"].items():
        path = run / m["path"]
        assert rl.sha256_file(path) == m["sha256"], f"{role} model changed"
        models[role] = (joblib.load(path), m)

    test = rl.load_split_frame(Path(inp["features_run"]) / "features.csv",
                               inp["strata"], "test")
    curves = rl.load_curves(Path(inp["oracle_run"]) / "oracle_curves.csv", "test")
    assert len(test) == 2000 and set(test["split"]) == {"test"}
    qids = test["query_id"].to_numpy()
    test["probe_plus_lid_dc"] = (test["probe_distance_computations"]
                                 + test["lid_probe_distance_computations"])

    preds, leaves = {}, None
    for role, (model, m) in models.items():
        X = test[m["features"]].to_numpy()
        p = model.predict(X)
        exp = json.loads((run / m["export"]).read_text())
        assert (rl.predict_from_export(exp, X) == p).all(), role
        preds[role] = p
        if role == "primary" and hasattr(model, "apply"):
            leaves = model.apply(X)

    pol = {}
    pol["router"] = rl.route(preds["primary"], qids,
                             test["probe_distance_computations"], curves,
                             tier_ef, target)
    pol["router_single_feature"] = rl.route(
        preds["single_feature"], qids, test["probe_distance_computations"],
        curves, tier_ef, target)
    pol["router_lid_ablation"] = rl.route(
        preds["lid_ablation"], qids, test["probe_plus_lid_dc"], curves,
        tier_ef, target)
    for name, ef in tier_ef.items():
        pol[f"fixed_{ef}"] = rl.fixed(qids, ef, curves, target)
    curve = rl.fixed_curve(curves, qids, target)
    matched_ef = rl.cheapest_fixed_ef(curve, pol["router"]["recall"].mean())
    pol[f"fixed_matched_{matched_ef}"] = rl.fixed(qids, matched_ef, curves, target)
    max_ef = int(curves.recall.columns.max())
    o_ef = np.where(test["reached"] == 1, test["oracle_ef"].fillna(max_ef),
                    max_ef).astype(int)
    rec, rec_id, dc = curves.lookup(qids, o_ef)
    pol["oracle"] = pd.DataFrame({"query_id": qids, "tier": "", "ef": o_ef,
                                  "recall": rec, "recall_id": rec_id,
                                  "search_dc": dc, "probe_dc": 0.0,
                                  "total_dc": dc.astype(float),
                                  "failure": rec < target - rl.TOL})

    ctx = test[["query_id", "difficulty", "censored", "tier_label", "oracle_ef"]]
    rows = pd.concat([p.assign(policy=k) for k, p in pol.items()])
    rows = rows.merge(ctx, on="query_id")
    rows.to_csv(out / "test_rows.csv", index=False)
    for k in ("router", "router_single_feature", "router_lid_ablation", "oracle"):
        pol[k][["query_id", "ef"]].to_csv(out / f"policy_{k}.csv", index=False)

    reg = rl.regret(pol["router"], test["oracle_ef"], test["reached"] == 1, curves)
    terr = rl.tier_error(preds["primary"], test["tier_label"])
    pq = test[["query_id", "difficulty", "censored", "tier_label", "oracle_ef",
               *rl.CORE_FEATURES, "lid"]].copy()
    pq["pred_primary"] = preds["primary"]
    pq["pred_single_feature"] = preds["single_feature"]
    pq["pred_lid_ablation"] = preds["lid_ablation"]
    pq["tier_error"] = terr
    pq["error_kind"] = np.select([terr > 0, terr < 0], ["conservative", "aggressive"],
                                 "correct")
    pq["router_recall"] = pol["router"]["recall"].to_numpy()
    pq["router_failure"] = pol["router"]["failure"].to_numpy()
    pq["router_total_dc"] = pol["router"]["total_dc"].to_numpy()
    pq["regret_total_dc"] = reg["regret_total_dc"].to_numpy()
    pq["regret_search_dc"] = reg["regret_search_dc"].to_numpy()
    if leaves is not None:
        pq["leaf"] = leaves
    pq.to_csv(out / "test_predictions.csv", index=False)

    rep = {"train_run": run.name, "matched_fixed_ef": matched_ef,
           "tiers": tier_ef, "n_test": int(len(test)),
           "censored_query_ids": test.loc[test["censored"], "query_id"].tolist()}

    unc = test.loc[~test["censored"], "query_id"]
    rep["summary"] = {k: summarize(p, nb, seed) for k, p in pol.items()}
    rep["summary_uncensored"] = {
        k: summarize(p[p["query_id"].isin(unc)], nb, seed) for k, p in pol.items()}
    rep["paired_router_vs"] = {k: paired(pol["router"], p, nb, seed)
                               for k, p in pol.items() if k != "router"}
    for k in ("router", "router_single_feature", "router_lid_ablation"):
        r = pol[k]
        cost_fixed = rl.interp_cost_at_recall(curve, r["recall"].mean())
        rep.setdefault("matched_recall", {})[k] = {
            "router_mean_recall": float(r["recall"].mean()),
            "router_mean_total_dc": float(r["total_dc"].mean()),
            "router_mean_search_dc_only": float(r["search_dc"].mean()),
            "cheapest_fixed_ef_at_that_recall": rl.cheapest_fixed_ef(
                curve, r["recall"].mean()),
            "fixed_cost_interpolated": cost_fixed,
            "fixed_over_router_cost": float(cost_fixed / r["total_dc"].mean()),
            "fixed_over_router_search_only": float(cost_fixed / r["search_dc"].mean()),
        }
    ok = ~test["censored"].to_numpy()
    rt, rs = reg["regret_total_dc"].to_numpy()[ok], reg["regret_search_dc"].to_numpy()[ok]
    fail = pol["router"]["failure"].to_numpy()[ok]
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    rep["regret"] = {
        "n": int(ok.sum()),
        "total_dc_mean": float(rt.mean()),
        "total_dc_mean_ci95": rl.bootstrap_ci(rt, nb, seed),
        "total_dc_quantiles": dict(zip(map(str, q), np.quantile(rt, q).tolist())),
        "search_dc_mean": float(rs.mean()),
        "search_dc_quantiles": dict(zip(map(str, q), np.quantile(rs, q).tolist())),
        "met_target": {"n": int((~fail).sum()),
                       "total_dc_mean": float(rt[~fail].mean()),
                       "search_dc_mean": float(rs[~fail].mean())},
        "failed_target_aggressive": {
            "n": int(fail.sum()),
            "search_dc_mean": float(rs[fail].mean()),
            "note": "effort saved by under-routing, paid for in recall"},
    }
    conf = pd.crosstab(test["tier_label"], preds["primary"],
                       rownames=["true"], colnames=["pred"]).reindex(
        index=list(rl.TIERS), columns=list(rl.TIERS), fill_value=0)
    rep["confusion_true_x_pred"] = conf.to_dict(orient="index")
    rep["tier_errors"] = {
        "accuracy": float((terr == 0).mean()),
        "conservative_rate": float((terr > 0).mean()),
        "aggressive_rate": float((terr < 0).mean()),
        "by_magnitude": pd.Series(terr).value_counts().sort_index().to_dict(),
        "failure_rate_when_aggressive": float(
            pq.loc[pq["error_kind"] == "aggressive", "router_failure"].mean()),
        "failure_rate_when_correct": float(
            pq.loc[pq["error_kind"] == "correct", "router_failure"].mean()),
        "failure_rate_when_conservative": float(
            pq.loc[pq["error_kind"] == "conservative", "router_failure"].mean()),
    }
    rep["by_difficulty"] = {}
    for g in ("easy", "medium", "hard"):
        ids = test.loc[test["difficulty"] == g, "query_id"]
        sub = {k: p[p["query_id"].isin(ids)] for k, p in pol.items()}
        gm = pq[pq["difficulty"] == g]
        rep["by_difficulty"][g] = {
            "n": int(len(ids)),
            "policies": {k: summarize(s, nb, seed) for k, s in sub.items()},
            "router_vs_fixed_48": paired(sub["router"], sub["fixed_48"], nb, seed),
            "router_vs_matched": paired(sub["router"],
                                        sub[f"fixed_matched_{matched_ef}"], nb, seed),
            "router_tier_errors": gm["error_kind"].value_counts().to_dict(),
            "router_pred_counts": gm["pred_primary"].value_counts().to_dict(),
            "regret_total_dc_mean_uncensored": float(
                gm.loc[~gm["censored"], "regret_total_dc"].mean()),
        }
    if leaves is not None:
        rep["by_leaf"] = (pq.groupby("leaf").agg(
            n=("query_id", "size"), predicted=("pred_primary", "first"),
            true_low=("tier_label", lambda s: float((s == "low").mean())),
            true_med=("tier_label", lambda s: float((s == "med").mean())),
            true_high=("tier_label", lambda s: float((s == "high").mean())),
            failure_rate=("router_failure", "mean"),
            mean_total_dc=("router_total_dc", "mean"))
            .reset_index().to_dict(orient="records"))
    rep["feature_medians_by_error_kind"] = (
        pq.groupby("error_kind")[[*rl.CORE_FEATURES, "lid", "oracle_ef"]]
        .median().to_dict(orient="index"))
    rep["error_kind_by_true_tier"] = pd.crosstab(
        pq["tier_label"], pq["error_kind"]).to_dict(orient="index")
    rep["censored_queries"] = rows[rows["censored"]][
        ["query_id", "policy", "ef", "recall", "recall_id", "total_dc",
         "failure"]].to_dict(orient="records")
    rep["censored_query_features"] = pq[pq["censored"]][
        ["query_id", *rl.CORE_FEATURES, "lid", "pred_primary"]].to_dict(
        orient="records")
    pm = rep["paired_router_vs"][f"fixed_matched_{matched_ef}"]
    crit = {
        "i_cost_lower_ci_below_0_and_p_lt_0.01":
            pm["mean_cost_diff_ci95"][1] < 0 and pm["cost_wilcoxon"]["p_value"] < 0.01
            and pm["mean_cost_diff"] < 0,
        "ii_relative_saving_ge_10pct": pm["relative_cost_saving"] >= 0.10,
        "iii_failure_not_worse_ci_upper_le_0.01":
            pm["failure_rate_diff_ci95"][1] <= 0.01,
    }
    rep["checkpoint"] = {"criteria": crit, "met": all(crit.values()),
                         "against": f"fixed ef {matched_ef} (cheapest grid ef "
                                    "with test mean recall >= router's)"}
    (out / "eval_report.json").write_text(rl.to_json(rep) + "\n")
    plot(out, pol, curve, rep, tier_ef, matched_ef)
    print(rl.to_json({"summary": {k: {kk: v[kk] for kk in
                                      ("mean_recall", "failure_rate",
                                       "mean_total_dc", "median_total_dc")}
                                  for k, v in rep["summary"].items()},
                      "matched_recall": rep["matched_recall"],
                      "checkpoint": rep["checkpoint"]}))
    print(out)
    return 0


def plot(out, pol, curve, rep, tier_ef, matched_ef):
    plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(curve["mean_dc"], curve["mean_recall"], color=C_BLUE, lw=2,
            label="fixed ef (test, full grid)")
    marks = [("router", C_ORANGE, "router (DT, incl. probe)", (-10, -16)),
             ("router_single_feature", C_AQUA, "single-feature router", (-60, 10)),
             ("router_lid_ablation", C_MAGENTA, "LID ablation router", (8, -14)),
             ("oracle", C_YELLOW, "oracle", (8, -12))]
    for k, c, lab, off in marks:
        s = pol[k]
        ax.plot(s["total_dc"].mean(), s["recall"].mean(), "o", ms=9, color=c,
                mec=SURFACE, mew=2, label=lab, zorder=5)
        ax.annotate(lab, (s["total_dc"].mean(), s["recall"].mean()),
                    textcoords="offset points", xytext=off, fontsize=8,
                    color=INK)
    for ef in tier_ef.values():
        s = pol[f"fixed_{ef}"]
        ax.plot(s["total_dc"].mean(), s["recall"].mean(), "s", ms=8,
                color=C_BLUE, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"fixed ef={ef}", (s["total_dc"].mean(), s["recall"].mean()),
                    textcoords="offset points", xytext=(8, 4), fontsize=8,
                    color=INK)
    ax.set_xscale("log")
    ax.set_xlim(left=300)
    ax.set_ylim(0.78, 1.005)
    ax.set_xlabel("mean distance computations per query (router: incl. probe)")
    ax.set_ylabel("mean tie-aware recall@10")
    ax.set_title("S0 test split (2,000 queries): router vs fixed efSearch",
                 color=INK, fontsize=11)
    ax.grid(alpha=0.25, color=INK2, lw=0.6)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "router_vs_fixed.png", dpi=140)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    groups = ["easy", "medium", "hard"]
    show = [("router", C_ORANGE), ("fixed_48", C_BLUE),
            (f"fixed_matched_{matched_ef}", C_AQUA), ("fixed_327", C_YELLOW)]
    w = 0.2
    for j, (k, c) in enumerate(show):
        fr = [rep["by_difficulty"][g]["policies"][k]["failure_rate"] for g in groups]
        dcm = [rep["by_difficulty"][g]["policies"][k]["mean_total_dc"] for g in groups]
        x = np.arange(3) + (j - 1.5) * w
        axes[0].bar(x, fr, w * 0.9, color=c, label=k)
        axes[1].bar(x, dcm, w * 0.9, color=c, label=k)
    for a, t in zip(axes, ["failure rate (recall < 0.95)",
                           "mean distance computations (router incl. probe)"]):
        a.set_xticks(range(3), groups)
        a.set_title(t, fontsize=10, color=INK)
        a.grid(axis="y", alpha=0.25, color=INK2, lw=0.6)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Test split by frozen difficulty group (train thresholds 18 / 48)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "router_by_difficulty.png", dpi=140)


def replot(cfg, run: Path) -> int:
    out = run / "test_eval"
    rows = pd.read_csv(out / "test_rows.csv")
    rep = json.loads((out / "eval_report.json").read_text())
    pol = {k: g for k, g in rows.groupby("policy")}
    curves = rl.load_curves(Path(cfg["inputs"]["oracle_run"]) / "oracle_curves.csv",
                            "test")
    curve = rl.fixed_curve(curves, sorted(rows["query_id"].unique()),
                           cfg["target_recall"])
    plot(out, pol, curve, rep, rep["tiers"], rep["matched_fixed_ef"])
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[3] == "--replot":
        sys.exit(replot(yaml.safe_load(Path(sys.argv[1]).read_text()),
                        Path(sys.argv[2])))
    sys.exit(main())
