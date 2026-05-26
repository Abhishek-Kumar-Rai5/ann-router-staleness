"""Phase 5'(b) analysis: feature -> oracle-effort signal stability (design_doc.md
A1.6/A1.7; computations pre-declared in docs/notes.md before any real-data state
existed). Run only after phase5b_checks.py reports ALL_PASS.

No router is fitted, loaded or evaluated. Phase 4b artefacts are not read.

Usage: python python/phase5b_analysis.py configs/phase5/sift1m.yaml <validation_dir>
"""

import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_lib as L  # noqa: E402

FEATS = ["knn_dist", "centroid_dist", "score_concentration", "lid"]
CORE = FEATS[:3]
X = 0.05
B = 2000
BOOT_SEED = 20261002
T1, T2 = 18, 48  # frozen tertile thresholds (derived/sift1m_s0_effort_tiers.json)
TRAJ_STATES = {"id": [10000, 20000, 40000, 80000], "ood": [10000, 20000, 40000, 80000],
               "del": [20000, 80000]}
EIGHT_PCT_OOD = ("At 8% insertion magnitude, the available pool forces the regional-OOD construction "
                 "to overlap approximately 91% with the matched ID set, so this level does not "
                 "provide a clean ID-vs-OOD comparison.")
LIMITATIONS = [
    "\"OOD\" is a narrow construction: regionally concentrated real SIFT vectors (a nearest-neighbour "
    "ball around one seeded anchor in the same sift_learn pool that supplies the ID inserts), not an "
    "independently sourced or semantically different distribution. Its concentration weakens with "
    "magnitude (distance ratio 0.67 at 1% to 0.97 at 8%, where OOD ~ ID). Findings about \"OOD\" apply "
    "to this construction only.",
    "Magnitude cap: inserts are capped at 8% of |S0| by the size of the in-distribution pool, and "
    "deletion at the stated levels. Any stability finding is \"stable within 1-8% insertion churn (and "
    "2-8% lazy deletion)\", not a general claim about index evolution.",
    "Deletion is lazy only (markDelete): no physical removal or graph repair.",
    "One dataset (SIFT1M) at this stage.",
    "Signal stability is necessary, not sufficient: a stable feature-effort correlation does not imply "
    "that a router would remain robust or would save cost. Nothing here is a claim about router "
    "cost savings.",
    EIGHT_PCT_OOD,
]


def classify(lo, hi):
    if lo >= -X:
        return "STABLE"
    if hi < -X:
        return "DEGRADED"
    return "INCONCLUSIVE"


def ranks2d(a):
    return stats.rankdata(a, axis=-1)


def spearman_rows(fr, yr):
    """Row-wise Pearson of already-ranked arrays (B, n)."""
    fc = fr - fr.mean(-1, keepdims=True)
    yc = yr - yr.mean(-1, keepdims=True)
    return (fc * yc).sum(-1) / np.sqrt((fc ** 2).sum(-1) * (yc ** 2).sum(-1))


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    P = yaml.safe_load(Path(sys.argv[1]).read_text())
    vdir = Path(sys.argv[2])
    V = json.loads((vdir / "report.json").read_text())
    if not V["ALL_PASS"]:
        raise SystemExit("STOP: validation did not pass; no analysis")
    ovl = np.load(vdir / "overlap_top10.npz")
    root = Path(P["state_root"])
    M = json.loads((root / "run_manifest.json").read_text())
    seeds = [str(s) for s in P["index"]["seeds"]]
    nq = 10000
    states = ["S0"] + [f"{t}_{c}" for t, cs in TRAJ_STATES.items() for c in cs]
    n0 = M["n0"]

    # ---- load row-level data -------------------------------------------------------
    rows = []
    data = {}
    for s in seeds:
        for name in states:
            st = M["seeds"][s]["states"][name]
            fe = pd.read_csv(Path(st["features_run"]) / "features.csv").sort_values("query_id")
            la = pd.read_csv(Path(st["oracle_run"]) / "oracle_labels.csv").sort_values("query_id")
            assert fe["query_id"].tolist() == list(range(nq)) == la["query_id"].tolist()
            y = np.where(la["reached"].to_numpy() == 1, la["oracle_ef"].to_numpy(dtype=float), np.inf)
            d = {f: fe[f].to_numpy() for f in FEATS}
            d["y"] = y
            d["censored"] = ~np.isfinite(y)
            d["overlap"] = ovl[f"{s}__{name}"] if name != "S0" else np.full(nq, 10)
            d["dist_comp"] = la["oracle_distance_computations"].to_numpy()
            data[(s, name)] = d
            traj, cnt = (name.split("_") + [0])[:2] if name != "S0" else ("S0", 0)
            df = pd.DataFrame({"seed": int(s), "state": name, "trajectory": traj, "count": int(cnt),
                               "update_fraction": int(cnt) / n0, "query_id": np.arange(nq),
                               "split": fe["split"].to_numpy(), **{f: d[f] for f in FEATS},
                               "oracle_ef": la["oracle_ef"].to_numpy(), "reached": la["reached"].to_numpy(),
                               "oracle_distance_computations": d["dist_comp"],
                               "overlap_top10_with_S0": d["overlap"]})
            rows.append(df)
    long = pd.concat(rows, ignore_index=True)
    s0 = long[long.state == "S0"][["seed", "query_id", *FEATS, "oracle_ef", "reached"]]
    long = long.merge(s0.rename(columns={c: c + "_S0" for c in [*FEATS, "oracle_ef", "reached"]}),
                      on=["seed", "query_id"])
    long["local_density_drift"] = (long["knn_dist"] - long["knn_dist_S0"]) / long["knn_dist_S0"]
    both = (long.reached == 1) & (long.reached_S0 == 1)
    long["log2_effort_ratio"] = np.where(both, np.log2(long.oracle_ef / long.oracle_ef_S0), np.nan)
    e0 = np.where(long.reached_S0 == 1, long.oracle_ef_S0, np.inf)
    long["difficulty_S0"] = np.select([e0 <= T1, e0 <= T2], ["easy", "medium"], "hard")

    # ---- point estimates and bootstrap ---------------------------------------------------
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, nq, size=(B, nq))
    rho, rho_bs = {}, {}
    for (s, name), d in data.items():
        yr = stats.rankdata(d["y"])
        for f in FEATS:
            rho[(s, name, f)] = float(stats.spearmanr(d[f], d["y"]).statistic)
            assert abs(rho[(s, name, f)] - float(np.corrcoef(stats.rankdata(d[f]), yr)[0, 1])) < 1e-12
        out = {f: np.empty(B) for f in FEATS}
        for b0 in range(0, B, 250):
            ii = idx[b0:b0 + 250]
            yrr = ranks2d(d["y"][ii])
            for f in FEATS:
                out[f][b0:b0 + 250] = spearman_rows(ranks2d(d[f][ii]), yrr)
        for f in FEATS:
            rho_bs[(s, name, f)] = out[f]
        print(f"bootstrap {s} {name}", flush=True)

    evolved = [n for n in states if n != "S0"]
    delta, delta_bs = {}, {}
    for s in seeds:
        for n in evolved:
            for f in FEATS:
                delta[(s, n, f)] = abs(rho[(s, n, f)]) - abs(rho[(s, "S0", f)])
                delta_bs[(s, n, f)] = np.abs(rho_bs[(s, n, f)]) - np.abs(rho_bs[(s, "S0", f)])

    cells = []
    for n in evolved:
        traj, cnt = n.split("_")
        for f in FEATS:
            pooled = float(np.mean([delta[(s, n, f)] for s in seeds]))
            pbs = np.mean([delta_bs[(s, n, f)] for s in seeds], axis=0)
            lo, hi = ci(pbs)
            rho0 = float(np.mean([abs(rho[(s, "S0", f)]) for s in seeds]))
            rhos = float(np.mean([abs(rho[(s, n, f)]) for s in seeds]))
            per_seed = {}
            for s in seeds:
                sl, sh = ci(delta_bs[(s, n, f)])
                per_seed[s] = {"rho_S0": rho[(s, "S0", f)], "rho_s": rho[(s, n, f)], "delta": delta[(s, n, f)],
                               "ci95": [sl, sh], "class": classify(sl, sh)}
            cells.append({"feature": f, "core": f in CORE, "state": n, "trajectory": traj, "count": int(cnt),
                          "magnitude_pct": 100 * int(cnt) / n0,
                          "abs_rho_S0_mean": rho0, "abs_rho_s_mean": rhos,
                          "delta_pooled": pooled, "abs_delta_pooled": abs(pooled),
                          "rel_delta_pooled": pooled / rho0, "ci95": [lo, hi], "class": classify(lo, hi),
                          "per_seed": per_seed,
                          "ood_8pct_descriptive_only": n == "ood_80000"})
    cdf = pd.DataFrame([{k: v for k, v in c.items() if k != "per_seed"} for c in cells])

    # ---- H1' -----------------------------------------------------------------------------
    h1 = {}
    for f in FEATS:
        cl = [c["class"] for c in cells if c["feature"] == f]
        verdict = "refuted" if "DEGRADED" in cl else ("supported" if all(x == "STABLE" for x in cl)
                                                       else "inconclusive")
        seed_deg = [{"state": c["state"], "seed": s, **v} for c in cells if c["feature"] == f
                    for s, v in c["per_seed"].items() if v["class"] == "DEGRADED"]
        seed_inc = [{"state": c["state"], "seed": s, "delta": v["delta"], "ci95": v["ci95"]}
                    for c in cells if c["feature"] == f for s, v in c["per_seed"].items()
                    if v["class"] == "INCONCLUSIVE"]
        h1[f] = {"role": "core" if f in CORE else "secondary (LID ablation)", "verdict": verdict,
                 "classes": dict(zip([c["state"] for c in cells if c["feature"] == f], cl)),
                 "min_pooled_ci_lower": min(c["ci95"][0] for c in cells if c["feature"] == f),
                 "seed_specific_DEGRADED": seed_deg, "seed_specific_INCONCLUSIVE": seed_inc}
    core_v = [h1[f]["verdict"] for f in CORE]
    h1_overall = ("supported" if all(v == "supported" for v in core_v)
                  else "refuted" if "refuted" in core_v else "inconclusive")

    # ---- H2' -----------------------------------------------------------------------------
    h2 = []
    for c in TRAJ_STATES["id"]:
        for f in FEATS:
            dd = float(np.mean([delta[(s, f"ood_{c}", f)] - delta[(s, f"id_{c}", f)] for s in seeds]))
            bs = np.mean([delta_bs[(s, f"ood_{c}", f)] - delta_bs[(s, f"id_{c}", f)] for s in seeds], axis=0)
            lo, hi = ci(bs)
            inferential = c < 80000
            h2.append({"magnitude_pct": 100 * c / n0, "feature": f, "delta_ood_minus_delta_id": dd, "ci95": [lo, hi],
                       "ci_excludes_0": bool(lo > 0 or hi < 0),
                       "status": "inferential (H2')" if inferential else "DESCRIPTIVE ONLY - ≈ ID by construction",
                       "note": "" if inferential else EIGHT_PCT_OOD})

    # ---- H3' -----------------------------------------------------------------------------
    h3 = []
    pairs = [("centroid_dist", "knn_dist"), ("centroid_dist", "score_concentration"),
             ("knn_dist", "score_concentration")]
    for traj, cs in TRAJ_STATES.items():
        n = f"{traj}_{cs[-1]}"
        for a, b in pairs:
            dd = float(np.mean([delta[(s, n, a)] - delta[(s, n, b)] for s in seeds]))
            bs = np.mean([delta_bs[(s, n, a)] - delta_bs[(s, n, b)] for s in seeds], axis=0)
            lo, hi = ci(bs)
            h3.append({"state": n, "contrast": f"delta({a}) - delta({b})", "estimate": dd, "ci95": [lo, hi],
                       "ci_excludes_0": bool(lo > 0 or hi < 0),
                       "note": EIGHT_PCT_OOD if n == "ood_80000" else ""})

    # ---- trend ---------------------------------------------------------------------------
    trend = []
    for traj, cs in TRAJ_STATES.items():
        for f in FEATS:
            dv = [cdf[(cdf.state == f"{traj}_{c}") & (cdf.feature == f)]["delta_pooled"].item() for c in cs]
            r = stats.spearmanr(np.log(cs), dv).statistic if len(cs) > 2 else np.nan
            trend.append({"trajectory": traj, "feature": f, "n_points": len(cs),
                          "spearman_logmag_delta": None if np.isnan(r) else float(r), "deltas": dv})

    # ---- descriptive per state -------------------------------------------------------
    base = L.read_fvecs(P["data"]["base"]).astype(np.float64)
    mu0 = base.mean(0)
    scale = np.sqrt(np.trace(np.cov(base, rowvar=False)))
    drift_D = {}
    for traj in ("id", "ood"):
        ins = L.read_fvecs(M["seeds"][seeds[0]]["states"][f"{traj}_80000"]["spec"]["inserted_vectors"]).astype(
            np.float64)
        for c in TRAJ_STATES[traj]:
            drift_D[f"{traj}_{c}"] = float(np.linalg.norm(ins[:c].mean(0) - mu0) / scale)
    for c in TRAJ_STATES["del"]:
        lab = pd.read_csv(M["seeds"][seeds[0]]["states"][f"del_{c}"]["spec"]["deleted_labels"])["label"].to_numpy()
        keep = np.ones(len(base), bool)
        keep[lab] = False
        drift_D[f"del_{c}"] = float(np.linalg.norm(base[keep].mean(0) - mu0) / scale)
    del base

    per_state = []
    for (s, name), d in data.items():
        g = long[(long.seed == int(s)) & (long.state == name)]
        fin = d["y"][np.isfinite(d["y"])]
        rec = {"seed": int(s), "state": name, "censored": int(d["censored"].sum()),
               "effort_mean_uncensored": float(fin.mean()), "effort_median": float(np.median(fin)),
               "effort_p90": float(np.percentile(fin, 90)), "effort_p99": float(np.percentile(fin, 99)),
               "mean_log2_effort_uncensored": float(np.log2(fin).mean()),
               "mean_oracle_distance_computations": float(d["dist_comp"].mean()),
               "mean_overlap_top10_with_S0": float(d["overlap"].mean()),
               "frac_queries_overlap_lt_10": float((d["overlap"] < 10).mean()),
               "distributional_drift_D": drift_D.get(name, 0.0)}
        for f in FEATS:
            rec[f"{f}_mean"] = float(d[f].mean())
            rec[f"{f}_median"] = float(np.median(d[f]))
            rec[f"{f}_p10"] = float(np.percentile(d[f], 10))
            rec[f"{f}_p90"] = float(np.percentile(d[f], 90))
        if name != "S0":
            for f in FEATS:
                a, b0 = g[f].to_numpy(), g[f + "_S0"].to_numpy()
                rec[f"{f}_median_abs_drift"] = float(np.median(np.abs(a - b0)))
                rec[f"{f}_median_rel_drift"] = float(np.median((a - b0) / np.abs(b0)))
                rec[f"{f}_frac_changed"] = float((a != b0).mean())
                rec[f"{f}_spearman_with_S0"] = float(stats.spearmanr(a, b0).statistic)
            lr = g["log2_effort_ratio"].dropna()
            rec.update({"effort_log2_ratio_mean": float(lr.mean()), "effort_log2_ratio_median": float(lr.median()),
                        "effort_log2_ratio_p10": float(lr.quantile(0.1)), "effort_log2_ratio_p90": float(lr.quantile(0.9)),
                        "effort_frac_increased": float((lr > 0).mean()), "effort_frac_decreased": float((lr < 0).mean()),
                        "effort_frac_unchanged": float((lr == 0).mean()),
                        "effort_spearman_with_S0": float(stats.spearmanr(np.where(d["censored"], np.inf, d["y"]),
                                                                         data[(s, "S0")]["y"]).statistic),
                        "local_density_drift_mean": float(g["local_density_drift"].mean()),
                        "local_density_drift_median": float(g["local_density_drift"].median())})
        per_state.append(rec)
    ps = pd.DataFrame(per_state)

    # ---- difficulty strata (descriptive) --------------------------------------------
    strata = []
    for (s, name), d in data.items():
        g = long[(long.seed == int(s)) & (long.state == name)]
        for lev in ("easy", "medium", "hard"):
            m = (g["difficulty_S0"] == lev).to_numpy()
            rec = {"seed": int(s), "state": name, "stratum": lev, "n": int(m.sum())}
            for f in FEATS:
                rec[f"rho_{f}"] = float(stats.spearmanr(d[f][m], d["y"][m]).statistic)
                rec[f"rho_{f}_S0"] = float(stats.spearmanr(data[(s, "S0")][f][m], data[(s, "S0")]["y"][m]).statistic)
                rec[f"delta_{f}"] = abs(rec[f"rho_{f}"]) - abs(rec[f"rho_{f}_S0"])
            lr = g["log2_effort_ratio"].to_numpy()[m]
            lr = lr[np.isfinite(lr)]
            rec.update({"effort_log2_ratio_mean": float(lr.mean()) if name != "S0" else 0.0,
                        "effort_frac_increased": float((lr > 0).mean()) if name != "S0" else 0.0,
                        "censored": int(d["censored"][m].sum()),
                        "mean_overlap": float(d["overlap"][m].mean())})
            strata.append(rec)
    sdf = pd.DataFrame(strata)
    sp = sdf[sdf.state != "S0"].groupby(["state", "stratum"])[
        [f"delta_{f}" for f in FEATS] + ["effort_log2_ratio_mean", "effort_frac_increased", "mean_overlap"]].mean()

    # ---- write --------------------------------------------------------------------------
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    od = Path(P["output_dir"]) / f"phase5b_analysis_{ts}"
    od.mkdir(parents=True)
    long.to_csv(od / "rows_query_state_seed.csv.gz", index=False)
    cdf.to_csv(od / "delta_cells_pooled.csv", index=False)
    pd.DataFrame([{"feature": c["feature"], "state": c["state"], "seed": s, **v}
                  for c in cells for s, v in c["per_seed"].items()]).to_csv(od / "delta_cells_per_seed.csv", index=False)
    pd.DataFrame([{"seed": int(s), "state": n, "feature": f, "rho": r} for (s, n, f), r in rho.items()]).to_csv(
        od / "rho_by_seed_state.csv", index=False)
    ps.to_csv(od / "per_state_descriptive.csv", index=False)
    sdf.to_csv(od / "strata_by_seed_state.csv", index=False)
    sp.to_csv(od / "strata_pooled.csv")
    rep = {"validation_report": str(vdir / "report.json"), "manifest": str(root / "run_manifest.json"),
           "X": X, "bootstrap": {"B": B, "seed": BOOT_SEED, "unit": "query id (shared across seeds/states)"},
           "H1_overall_core": h1_overall, "H1": h1, "H2": h2, "H3": h3, "trend": trend,
           "distributional_drift_D": drift_D, "cells": cells, "limitations_A1_10": LIMITATIONS,
           "eight_pct_ood_wording": EIGHT_PCT_OOD}
    (od / "analysis.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")

    # figure: pooled delta vs magnitude, per feature and trajectory
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    col = {"id": "#2a6fdb", "ood": "#d9822b", "del": "#3a9d5d"}
    for ax, f in zip(axes, FEATS, strict=True):
        for traj, cs in TRAJ_STATES.items():
            sub = cdf[(cdf.feature == f) & (cdf.trajectory == traj)].sort_values("count")
            x = sub["magnitude_pct"].to_numpy() * (1.03 if traj == "ood" else 0.97 if traj == "del" else 1)
            lo = np.array([c[0] for c in sub["ci95"]])
            hi = np.array([c[1] for c in sub["ci95"]])
            ax.errorbar(x, sub["delta_pooled"], yerr=[sub["delta_pooled"] - lo, hi - sub["delta_pooled"]],
                        marker="o", ms=5, lw=1.5, capsize=3, color=col[traj], label=traj)
            if traj == "ood":
                ax.scatter([x[-1]], [sub["delta_pooled"].iloc[-1]], s=110, facecolors="none",
                           edgecolors=col["ood"], lw=1.2)
        ax.axhline(0, color="#888", lw=0.8)
        ax.axhline(-X, color="#b22", lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1%", "2%", "4%", "8%"])
        ax.set_title(f + (" (secondary)" if f == "lid" else ""))
        ax.set_xlabel("update magnitude (% of |S0|)")
    axes[0].set_ylabel("pooled Δ = |ρ_s| − |ρ_0|  (95% CI)")
    axes[0].legend(title="trajectory", fontsize=8)
    fig.suptitle("Phase 5′: feature→oracle-effort Spearman change vs S0 (dashed: −X = −0.05; "
                 "circled OOD 8% = ≈ ID by construction, descriptive)", fontsize=10)
    fig.tight_layout()
    fig.savefig(od / "delta_vs_magnitude.png", dpi=130)

    print("H1' overall (core):", h1_overall)
    for f in FEATS:
        print(f, h1[f]["verdict"], "min CI lb", round(h1[f]["min_pooled_ci_lower"], 4),
              "seed-DEGRADED:", len(h1[f]["seed_specific_DEGRADED"]))
    print(od)
    return 0


if __name__ == "__main__":
    sys.exit(main())
