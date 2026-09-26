"""Experiment E analysis. It refuses to run on missing or incomplete inputs, and it
never tunes, selects or drops anything.

Usage: python python/exp_e_analysis.py configs/exp_e/experiment_e.yaml <results/exp_e_rows_dir>
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_e_lib as E  # noqa: E402

POLICIES = ("B1", "DARTH")
ROW_COLS = ["policy", "seed", "state", "query_id", "cost", "recall_tie_aware", "recall_id"]
DARTH_COLS = ["stopped_early", "last_predicted", "predictor_calls", "predictor_inferences",
              "predictor_seconds", "wall_seconds"]
PRED_NAMES = ["update_fraction", "overlap_decay", "density_drift", "distributional_drift"]


class AnalysisInputError(RuntimeError):
    pass


def state_magnitude(state, n0):
    return int(state.split("_")[1]) / n0


def state_trajectory(state):
    return state.split("_")[0]


def validate(rows, preds_q, preds_state, cfg):
    def fail(msg):
        raise AnalysisInputError(msg)

    for c in ROW_COLS:
        if c not in rows.columns:
            fail(f"missing column {c}")
    d = rows[rows.policy == "DARTH"]
    for c in DARTH_COLS:
        if c not in rows.columns or d[c].isna().any():
            fail(f"missing DARTH column {c}")
    seeds, states = list(cfg["seeds"]), ["S0", *cfg["insertion_states"], *cfg["deletion_states"]]
    n = cfg["n_queries"]
    qids = None
    for pol in ("B1", "REF", "DARTH"):
        for s in seeds:
            for st in states:
                g = rows[(rows.policy == pol) & (rows.seed == s) & (rows.state == st)]
                if len(g) != n:
                    fail(f"{pol} seed {s} {st}: {len(g)} rows, expected {n}")
                if g.query_id.duplicated().any():
                    fail(f"{pol} seed {s} {st}: duplicate query ids")
                q = np.sort(g.query_id.to_numpy())
                if qids is None:
                    qids = q
                elif not np.array_equal(q, qids):
                    fail(f"{pol} seed {s} {st}: query set differs")
                if g[["cost", "recall_tie_aware", "recall_id"]].isna().any().any():
                    fail(f"{pol} seed {s} {st}: missing values")
                if not (g.recall_tie_aware.between(0, 1).all() and g.recall_id.between(0, 1).all()):
                    fail(f"{pol} seed {s} {st}: recall outside [0, 1]")
                if (g.cost <= 0).any():
                    fail(f"{pol} seed {s} {st}: non-positive cost")
    extra = set(zip(rows.policy, rows.seed, rows.state)) - {(p, s, st) for p in ("B1", "REF", "DARTH")
                                                            for s in seeds for st in states}
    if extra:
        fail(f"unexpected cells present: {sorted(extra)[:5]}")
    for s in seeds:
        b = rows[(rows.policy == "B1") & (rows.seed == s) & (rows.state == "S0")].sort_values("query_id")
        r = rows[(rows.policy == "REF") & (rows.seed == s) & (rows.state == "S0")].sort_values("query_id")
        if not (np.array_equal(b.cost.to_numpy(), r.cost.to_numpy())
                and np.array_equal(b.recall_tie_aware.to_numpy(), r.recall_tie_aware.to_numpy())):
            fail(f"seed {s}: REF(S0) differs from frozen B1(S0)")
    for s in seeds:
        for st in states[1:]:
            g = preds_q[(preds_q.seed == s) & (preds_q.state == st)]
            if not np.array_equal(np.sort(g.query_id.to_numpy()), qids) or g[["overlap_decay", "density_drift"]].isna().any().any():
                fail(f"H-E3 predictors incomplete for seed {s} {st}")
    for st in states[1:]:
        if st not in preds_state.index or preds_state.loc[st, ["update_fraction", "distributional_drift"]].isna().any():
            fail(f"state-level predictors missing for {st}")
    return qids


def _vec(rows, pol, seed, state, col, qids):
    g = rows[(rows.policy == pol) & (rows.seed == seed) & (rows.state == state)].set_index("query_id")
    return g.loc[qids, col].to_numpy(dtype=float)


def _ci(a):
    a = np.asarray(a, float)
    if not np.isfinite(a).all():
        raise AnalysisInputError("non-finite bootstrap replicate in a decision quantity")
    return E.ci95(a)


def _ci_desc(a):
    a = np.asarray(a, float)
    f = a[np.isfinite(a)]
    return {"ci95": E.ci95(f) if len(f) else [float("nan"), float("nan")],
            "nonfinite_replicates": int((~np.isfinite(a)).sum())}


def _spearman_point(x, y):
    return float(E.spearman_rows(np.asarray(x)[None, :], np.asarray(y)[None, :])[0])


def analyze(rows, preds_q, preds_state, cfg):
    qids = validate(rows, preds_q, preds_state, cfg)
    seeds = list(cfg["seeds"])
    ins, dels = list(cfg["insertion_states"]), list(cfg["deletion_states"])
    evolved = ins + dels
    n0 = cfg["n0"]
    idx = E.boot_index(len(qids))
    V = lambda pol, s, st, col="cost": _vec(rows, pol, s, st, col, qids)  # noqa: E731
    out = {"design": "docs/exp_e_design_freeze_v2.md", "n_queries": int(len(qids)),
           "bootstrap": {"B": E.B, "seed": E.BOOT_SEED, "unit": "query id (shared matrix)", "interval": "95% percentile"},
           "margins": {"H-E1": E.M1, "H-E2": E.M2}}

    mean_c, rep_c = {}, {}
    for pol in ("B1", "REF", "DARTH"):
        for s in seeds:
            for st in ["S0", *evolved]:
                c = V(pol, s, st)
                mean_c[(pol, s, st)] = float(c.mean())
                rep_c[(pol, s, st)] = E.rep_mean(c, idx)
    recall = {}
    for pol in ("B1", "REF", "DARTH"):
        for st in ["S0", *evolved]:
            per = {s: V(pol, s, st, "recall_tie_aware") for s in seeds}
            recall[f"{pol}|{st}"] = {
                "per_seed": {s: float(per[s].mean()) for s in seeds},
                "seed_mean": float(np.mean([per[s].mean() for s in seeds])),
                "seed_mean_ci95": E.ci95(np.mean([E.rep_mean(per[s], idx) for s in seeds], axis=0))}
    out["mean_recall_tie_aware"] = recall

    he1, d_point, d_rep = {}, {}, {}
    for pol in POLICIES:
        cells = {}
        for st in evolved:
            per_seed, reps = {}, []
            for s in seeds:
                a = (mean_c[(pol, s, st)], mean_c[("REF", s, st)], mean_c[(pol, s, "S0")], mean_c[("REF", s, "S0")])
                ra = (rep_c[(pol, s, st)], rep_c[("REF", s, st)], rep_c[(pol, s, "S0")], rep_c[("REF", s, "S0")])
                dp, dr = float(E.d_stat(*a)), E.d_stat(*ra)
                rp, rr = float(E.r_stat(a[0], a[1])), E.r_stat(ra[0], ra[1])
                r0p, r0r = float(E.r_stat(a[2], a[3])), E.r_stat(ra[2], ra[3])
                cp, cr = V(pol, s, st), V("REF", s, st)
                rel = (cp - cr) / cr
                d_point[(pol, s, st)], d_rep[(pol, s, st)] = dp, dr
                ci = _ci(dr)
                per_seed[s] = {"D": dp, "D_ci95": ci, "class": E.classify_d(ci),
                               "R_descriptive": rp, "R_ci95": E.ci95(rr),
                               "R_S0_descriptive": r0p,
                               "deltaR_additive_descriptive": rp - r0p, "deltaR_ci95": E.ci95(rr - r0r),
                               "per_query_relative_descriptive": float(rel.mean()),
                               "per_query_relative_ci95": E.ci95(E.rep_mean(rel, idx))}
                reps.append(dr)
            pooled = np.mean(reps, axis=0)
            point = float(np.mean([per_seed[s]["D"] for s in seeds]))
            ci = _ci(pooled)
            cells[st] = {"family": "insertion" if st in ins else "deletion",
                         "D_bar": point, "D_bar_ci95": ci, "class": E.classify_d(ci),
                         "seed_specific_stale": [s for s in seeds if per_seed[s]["class"].startswith("STALE")],
                         "per_seed": per_seed,
                         "note_8pct_ood": E.EIGHT_PCT_OOD if st == "ood_80000" else ""}
            d_rep[(pol, "pooled", st)] = pooled
        he1[pol] = {"cells": cells,
                    "family_verdict": {"insertion": E.family_verdict([cells[st]["class"] for st in ins]),
                                       "deletion": E.family_verdict([cells[st]["class"] for st in dels])}}
    out["H-E1"] = he1

    he2, e_point, e_rep = {}, {}, {}
    passv = {(pol, s, st): E.is_pass(V(pol, s, st, "recall_tie_aware"))
             for pol in ("B1", "REF", "DARTH") for s in seeds for st in ["S0", *evolved]}
    for pol in POLICIES:
        cells = {}
        for st in evolved:
            per_seed, reps = {}, []
            for s in seeds:
                k_p, k_ref = passv[(pol, s, "S0")], passv[("B1", s, "S0")]
                fail_p, fail_ref = ~passv[(pol, s, st)], ~passv[("REF", s, st)]
                nfp, nfr = E.nf_rate(k_p, fail_p), E.nf_rate(k_ref, fail_ref)
                rep = E.nf_rate(k_p, fail_p, idx) - E.nf_rate(k_ref, fail_ref, idx)
                ci = _ci(rep)
                e_point[(pol, s, st)], e_rep[(pol, s, st)] = nfp - nfr, rep
                per_seed[s] = {"E": nfp - nfr, "E_ci95": ci, "class": E.classify_e(ci),
                               "NF_frozen": nfp, "NF_reference": nfr,
                               "cohort_size_frozen": int(k_p.sum()), "cohort_size_reference": int(k_ref.sum()),
                               "transitions_frozen": E.transitions(k_p, passv[(pol, s, st)]),
                               "transitions_reference": E.transitions(passv[("REF", s, "S0")], passv[("REF", s, st)])}
                reps.append(rep)
            pooled = np.mean(reps, axis=0)
            ci = _ci(pooled)
            cells[st] = {"family": "insertion" if st in ins else "deletion",
                         "E_bar": float(np.mean([per_seed[s]["E"] for s in seeds])), "E_bar_ci95": ci,
                         "class": E.classify_e(ci),
                         "seed_specific_stale": [s for s in seeds if per_seed[s]["class"].startswith("STALE")],
                         "per_seed": per_seed,
                         "note_8pct_ood": E.EIGHT_PCT_OOD if st == "ood_80000" else ""}
            e_rep[(pol, "pooled", st)] = pooled
        he2[pol] = {"cells": cells,
                    "family_verdict": {"insertion": E.family_verdict([cells[st]["class"] for st in ins]),
                                       "deletion": E.family_verdict([cells[st]["class"] for st in dels])}}
    he2["caveat"] = ("DARTH's S0-pass cohort differs in composition from the reference cohort (K_REF = K_B1); "
                     "E(DARTH) includes a cohort-mix component (reported, not corrected).")
    out["H-E2"] = he2

    def predictor_cells(states_):
        point, rep = {}, {}
        for name in PRED_NAMES:
            pv, rv = [], []
            for s in seeds:
                for st in states_:
                    if name in ("update_fraction", "distributional_drift"):
                        v = float(preds_state.loc[st, name])
                        pv.append(v)
                        rv.append(np.full(E.B, v))
                    else:
                        g = preds_q[(preds_q.seed == s) & (preds_q.state == st)].set_index("query_id").loc[qids, name].to_numpy(float)
                        pv.append(float(g.mean()))
                        rv.append(E.rep_mean(g, idx))
            point[name], rep[name] = np.array(pv), np.stack(rv, axis=1)
        return point, rep

    def outcome_cells(pol, which, states_):
        P_, R_ = (d_point, d_rep) if which == "D" else (e_point, e_rep)
        return (np.array([P_[(pol, s, st)] for s in seeds for st in states_]),
                np.stack([R_[(pol, s, st)] for s in seeds for st in states_], axis=1))

    ip, irp = predictor_cells(ins)
    tests = []
    for pol in POLICIES:
        for which in ("D", "E"):
            yp, yr = outcome_cells(pol, which, ins)
            for name in PRED_NAMES:
                rho = _spearman_point(ip[name], yp)
                reps = E.spearman_rows(irp[name], yr)
                if not np.isfinite(rho) or not np.isfinite(reps).all():
                    raise AnalysisInputError(f"H-E3 Spearman undefined (constant input) for {pol} {which} {name}")
                tests.append({"policy": pol, "outcome": which, "predictor": name, "n_cells": int(len(yp)),
                              "rho": rho, "ci95": E.ci95(reps), "p_boot": E.boot_p_two_sided(reps)})
    adj, rej = E.holm([t["p_boot"] for t in tests])
    for t, a, r in zip(tests, adj, rej, strict=True):
        t["p_holm"], t["associated"] = float(a), bool(r)
    sec = []
    for pol in POLICIES:
        for which in ("D", "E"):
            yp, yr = outcome_cells(pol, which, ins)
            ref_r = E.spearman_rows(irp["update_fraction"], yr)
            ref_p = _spearman_point(ip["update_fraction"], yp)
            for name in PRED_NAMES[1:]:
                r_ = E.spearman_rows(irp[name], yr)
                sec.append({"policy": pol, "outcome": which, "predictor": name,
                            "abs_rho_minus_abs_rho_update_fraction": abs(_spearman_point(ip[name], yp)) - abs(ref_p),
                            "ci95": E.ci95(np.abs(r_) - np.abs(ref_r))})
    dp_, drp = predictor_cells(dels)
    deletion = []
    for pol in POLICIES:
        for which in ("D", "E"):
            yp, yr = outcome_cells(pol, which, dels)
            for name in PRED_NAMES:
                r_ = E.spearman_rows(drp[name], yr)
                f = r_[np.isfinite(r_)]
                deletion.append({"policy": pol, "outcome": which, "predictor": name, "n_cells": int(len(yp)),
                                 "rho": _spearman_point(dp_[name], yp),
                                 "ci95": E.ci95(f) if len(f) else [float("nan"), float("nan")],
                                 "nonfinite_replicates": int((~np.isfinite(r_)).sum())})
    perq = []
    for pol in POLICIES:
        cd, nf, od, dd = [], [], [], []
        for s in seeds:
            k = passv[(pol, s, "S0")]
            for st in ins:
                g = preds_q[(preds_q.seed == s) & (preds_q.state == st)].set_index("query_id").loc[qids]
                cd.append(V(pol, s, st) - V("REF", s, st))
                nf.append(np.where(k, ~passv[(pol, s, st)], np.nan))
                od.append(g.overlap_decay.to_numpy(float))
                dd.append(g.density_drift.to_numpy(float))
        cd, nf, od, dd = map(np.concatenate, (cd, nf, od, dd))
        m = np.isfinite(nf)
        for name, x in (("overlap_decay", od), ("density_drift", dd)):
            perq.append({"policy": pol, "predictor": name,
                         "rho_vs_cost_diff": _spearman_point(x, cd),
                         "rho_vs_newly_failing_in_cohort": _spearman_point(x[m], nf[m])})
    out["H-E3"] = {"primary_insertion_tests": tests, "multiplicity": "Holm-Bonferroni, 16 tests, alpha 0.05",
                   "secondary_abs_rho_vs_update_fraction": sec, "deletion_descriptive": deletion,
                   "secondary_per_query_insertion": perq}

    he4 = {}
    for real_col, label in (("recall_id", "primary_id"), ("recall_tie_aware", "secondary_tie_aware")):
        cells = {}
        base = {}
        for st in ["S0", *evolved]:
            per_seed, rb_, rm_ = {}, [], []
            for s in seeds:
                stp = V("DARTH", s, st, "stopped_early") == 1
                pred, real = V("DARTH", s, st, "last_predicted"), V("DARTH", s, st, real_col)
                pt = E.h_e4(stp, pred, real)
                rp = E.h_e4(stp, pred, real, idx)
                if st == "S0":
                    base[s] = (pt, rp)
                entry = {**pt, "bias_ci": _ci_desc(rp["bias"]), "mace_ci": _ci_desc(rp["mace"])}
                if st != "S0":
                    entry["bias_change_from_S0"] = pt["bias"] - base[s][0]["bias"]
                    entry["bias_change_ci"] = _ci_desc(rp["bias"] - base[s][1]["bias"])
                    entry["mace_change_from_S0"] = pt["mace"] - base[s][0]["mace"]
                    entry["mace_change_ci"] = _ci_desc(rp["mace"] - base[s][1]["mace"])
                per_seed[s] = entry
                rb_.append(rp["bias"] - (base[s][1]["bias"] if st != "S0" else 0))
                rm_.append(rp["mace"] - (base[s][1]["mace"] if st != "S0" else 0))
            cells[st] = {"per_seed": per_seed,
                         "stopped_fraction_mean": float(np.mean([per_seed[s]["stopped_fraction"] for s in seeds])),
                         "bias_mean": float(np.mean([per_seed[s]["bias"] for s in seeds])),
                         "mace_mean": float(np.mean([per_seed[s]["mace"] for s in seeds])),
                         "small_n_any_seed": any(per_seed[s]["small_n"] for s in seeds)}
            if st != "S0":
                cells[st]["bias_change_mean"] = float(np.mean([per_seed[s]["bias_change_from_S0"] for s in seeds]))
                cells[st]["bias_change_mean_ci"] = _ci_desc(np.mean(rb_, axis=0))
                cells[st]["mace_change_mean"] = float(np.mean([per_seed[s]["mace_change_from_S0"] for s in seeds]))
                cells[st]["mace_change_mean_ci"] = _ci_desc(np.mean(rm_, axis=0))
        he4[label] = cells
    he4["disclosures"] = ["conditional on early-stopped queries (selection at predicted >= 0.95); not probabilistic calibration",
                          "realised recall is discrete (tenths): MACE has an irreducible floor",
                          "cap-run queries excluded (their last_predicted is a stale check value)",
                          "no margin or decision rule (descriptive)"]
    he4["overhead_descriptive"] = {
        f"{s}|{st}": {c: float(V("DARTH", s, st, c).mean()) for c in ("predictor_calls", "predictor_seconds", "wall_seconds")}
        for s in seeds for st in ["S0", *evolved]}
    out["H-E4"] = he4

    trends = {}
    for pol in POLICIES:
        for which, P_, R_ in (("D", d_point, d_rep), ("E", e_point, e_rep)):
            df = pd.DataFrame([{"seed": s, "L": float(np.log2(state_magnitude(st, n0) / 0.01)),
                                "T": int(state_trajectory(st) == "ood"), "y": P_[(pol, s, st)]}
                               for s in seeds for st in ins])
            model = E.mixedlm_trend(df, "y")
            pooled = {st: np.mean([R_[(pol, s, st)] for s in seeds], axis=0) for st in evolved}
            pooled_pt = {st: float(np.mean([P_[(pol, s, st)] for s in seeds])) for st in evolved}
            robust = {}
            for traj in ("id", "ood"):
                sts = [st for st in ins if state_trajectory(st) == traj]
                if len(sts) >= 2:
                    L = [np.log2(state_magnitude(st, n0) / 0.01) for st in sts]
                    robust[traj] = {"slope": float(E.ols_slope(L, [[pooled_pt[st] for st in sts]])[0]),
                                    "ci95": E.ci95(E.ols_slope(L, np.stack([pooled[st] for st in sts], axis=1)))}
            dele = {}
            if len(dels) == 2:
                a, b = sorted(dels, key=lambda st: state_magnitude(st, n0))
                dele = {"difference": f"{b} - {a}", "point": pooled_pt[b] - pooled_pt[a],
                        "ci95": E.ci95(pooled[b] - pooled[a])}
            trends[f"{pol}|{which}"] = {"insertion_mixedlm": model, "robustness_ols_slope_pooled": robust,
                                       "deletion_difference_descriptive": dele}
    out["trends_secondary"] = trends
    out["note_8pct_ood"] = E.EIGHT_PCT_OOD
    return out


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_real(plan, rows_dir):
    import yaml
    P = yaml.safe_load(Path(plan).read_text())
    meta = json.loads((Path(rows_dir) / "metadata.json").read_text())
    for s, c in P["s0"].items():
        if _sha(c["darth_policy"]) != c["darth_policy_sha256"]:
            raise AnalysisInputError(f"policy hash mismatch seed {s}")
        pol = json.loads(Path(c["darth_policy"]).read_text())
        if _sha(pol["model_path"]) != pol["model_sha256"]:
            raise AnalysisInputError(f"model hash mismatch seed {s}")
    rows = pd.read_csv(Path(rows_dir) / "rows.csv.gz")
    split = pd.read_csv(P["split_path"])
    test = set(split.loc[split.split == "test", "query_id"])
    pq = pd.read_csv(Path(P["phase5_analysis"]) / "rows_query_state_seed.csv.gz",
                     usecols=["seed", "state", "query_id", "update_fraction", "overlap_top10_with_S0",
                              "local_density_drift"])
    pq = pq[pq.query_id.isin(test)].assign(overlap_decay=lambda d: 1 - d.overlap_top10_with_S0 / 10,
                                           density_drift=lambda d: d.local_density_drift)
    drift = json.loads((Path(P["phase5_analysis"]) / "analysis.json").read_text())["distributional_drift_D"]
    ps = (pq.groupby("state")["update_fraction"].first().to_frame()
          .assign(distributional_drift=lambda d: d.index.map(drift)))
    cfg = {"seeds": P["seeds"], "insertion_states": P["insertion_states"],
           "deletion_states": P["deletion_states"], "n0": P["n0"], "n_queries": 2000}
    return rows, pq, ps, cfg, meta, P


def main() -> int:
    plan, rows_dir = sys.argv[1], sys.argv[2]
    rows, pq, ps, cfg, meta, P = load_real(plan, rows_dir)
    res = analyze(rows, pq, ps, cfg)
    res["provenance"] = {"rows_dir": rows_dir, "rows_metadata": meta, "plan_sha256": _sha(plan),
                         "analysis_script_sha256": _sha(__file__), "lib_sha256": _sha(Path(__file__).with_name("exp_e_lib.py"))}
    od = Path(P["output_dir"]) / f"exp_e_analysis_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    od.mkdir(parents=True)
    (od / "analysis.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(od)
    return 0


if __name__ == "__main__":
    sys.exit(main())
