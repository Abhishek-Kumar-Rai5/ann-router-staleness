"""Experiment E (docs/exp_e_design_freeze_v2.md): pure functions, no I/O side effects.

Section numbers refer to the design freeze. The reference construction reuses the
frozen Phase-4b code (router4b_lib.two_ef_mix / assign_mix, router_lib.load_curves)
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router4b_lib as rb  # noqa: E402

QUERY_SET_ID = "sift_query"
B1_HASH_SEED = 20261006
TARGET = 0.95
M1, M2 = 0.10, 0.05          # H-E1 / H-E2 margins (frozen)
B, BOOT_SEED = 2000, 20261002
PASS_TOL = 1e-9
SMALL_N = 30                 # H-E4 small-n flag
EIGHT_PCT_OOD = ("At 8% insertion magnitude, the available pool forces the regional-OOD construction "
                 "to overlap approximately 91% with the matched ID set, so this level does not provide "
                 "a clean ID-vs-OOD comparison.")


# ---- §3 / §5: policies -----------------------------------------------------------------
def ref_mix(train_curves):
    """§5: Phase-4b mean-recall two-ef mix on mean training tie-aware recall."""
    grid = [int(e) for e in train_curves.recall.columns]
    level = train_curves.recall.to_numpy().mean(0)
    return rb.two_ef_mix(level, grid, TARGET)


def assign(mix, query_ids):
    return rb.assign_mix(mix, [int(q) for q in query_ids], QUERY_SET_ID, B1_HASH_SEED)


def same_mix(a, b):
    return (a is not None and b is not None and a["ef_lo"] == b["ef_lo"] and a["ef_hi"] == b["ef_hi"]
            and abs(a["w_hi"] - b["w_hi"]) <= 1e-12)


def is_pass(recall_tie_aware):
    return np.asarray(recall_tie_aware) >= 1 - PASS_TOL


# ---- §10: bootstrap ---------------------------------------------------------------------------
def boot_index(n):
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(B, n))


def ci95(a):
    lo, hi = np.percentile(np.asarray(a, float), [2.5, 97.5])
    return [float(lo), float(hi)]


def rep_mean(x, idx):
    """Mean of x over each bootstrap replicate (rows of idx)."""
    return np.asarray(x, float)[idx].mean(1)


# ---- §6: H-E1 -------------------------------------------------------------------------------
def d_stat(cp, cr, cp0, cr0):
    """§6.2: D = [cp/cr] / [cp0/cr0] - 1 (all arguments are mean costs, scalars or arrays)."""
    return (np.asarray(cp) / np.asarray(cr)) / (np.asarray(cp0) / np.asarray(cr0)) - 1


def r_stat(cp, cr):
    return (np.asarray(cp) - np.asarray(cr)) / np.asarray(cr)


def classify_d(ci, m=M1):
    lo, hi = ci
    if lo >= -m and hi <= m:
        return "STABLE"
    if lo > m:
        return "STALE-COSTLIER"
    if hi < -m:
        return "STALE-CHEAPER"
    return "INCONCLUSIVE"


def classify_e(ci, m=M2):
    lo, hi = ci
    if lo >= -m and hi <= m:
        return "STABLE"
    if lo > m:
        return "STALE-WORSE"
    if hi < -m:
        return "STALE-BETTER"
    return "INCONCLUSIVE"


def family_verdict(classes):
    """§6.4 / §7.3 intersection rule (unchanged)."""
    if all(c == "STABLE" for c in classes):
        return "STABLE-FAMILY"
    if any(c.startswith("STALE") for c in classes):
        return "STALE-FAMILY"
    return "INCONCLUSIVE-FAMILY"


# ---- §7: H-E2 -------------------------------------------------------------------------------
def nf_rate(cohort, fail_s, idx=None):
    """§7.2: share of the S0-pass cohort failing at s (denominator = cohort size)."""
    cohort = np.asarray(cohort, bool)
    hit = cohort & np.asarray(fail_s, bool)
    if idx is None:
        return float(hit.sum() / cohort.sum())
    return hit[idx].sum(1) / cohort[idx].sum(1)


def transitions(pass0, pass_s):
    p0, ps = np.asarray(pass0, bool), np.asarray(pass_s, bool)
    return {"stable_pass": int((p0 & ps).sum()), "newly_failing": int((p0 & ~ps).sum()),
            "stable_fail": int((~p0 & ~ps).sum()), "improved": int((~p0 & ps).sum())}


# ---- §8: H-E3 ---------------------------------------------------------------------------------
def spearman_rows(x, y):
    """Row-wise Spearman between two (R, n) arrays (average ranks); nan if a row is constant."""
    rx = stats.rankdata(np.atleast_2d(x), axis=1)
    ry = stats.rankdata(np.atleast_2d(y), axis=1)
    rx = rx - rx.mean(1, keepdims=True)
    ry = ry - ry.mean(1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rx * ry).sum(1) / den, np.nan)


def boot_p_two_sided(reps):
    r = np.asarray(reps, float)
    r = r[np.isfinite(r)]
    p = 2 * min(np.mean(r <= 0), np.mean(r >= 0))
    return float(min(1.0, max(p, 1 / B)))


def holm(pvals, alpha=0.05):
    """Holm–Bonferroni: returns adjusted p-values (same order) and reject flags."""
    p = np.asarray(pvals, float)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj, adj < alpha


# ---- §9: H-E4 ---------------------------------------------------------------------------------
def h_e4(stopped, pred, real, idx=None):
    s = np.asarray(stopped, bool)
    err = np.asarray(pred, float) - np.asarray(real, float)
    if idx is None:
        n = int(s.sum())
        return {"n_stopped": n, "stopped_fraction": float(s.mean()),
                "bias": float(err[s].mean()) if n else float("nan"),
                "mace": float(np.abs(err[s]).mean()) if n else float("nan"),
                "small_n": n < SMALL_N}
    sw = s[idx]
    n = sw.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"bias": (err[idx] * sw).sum(1) / n, "mace": (np.abs(err)[idx] * sw).sum(1) / n}


# ---- §11: mixed-effects ------------------------------------------------------------------------
def mixedlm_trend(df, outcome):
    """y ~ L + T + L:T, groups = seed, random intercept, REML. Returns a dict (never raises)."""
    import warnings

    import statsmodels.formula.api as smf
    out = {"formula": f"{outcome} ~ L + T + L:T", "groups": "seed", "re_formula": "1", "reml": True}
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit = smf.mixedlm(f"{outcome} ~ L + T + L:T", df, groups=df["seed"], re_formula="1").fit(reml=True)
        ci = fit.conf_int(alpha=0.05)
        out.update({
            "converged": bool(fit.converged),
            "warnings": sorted({str(x.message)[:200] for x in w}),
            "coef": {k: float(v) for k, v in fit.fe_params.items()},
            "ci95_wald": {k: [float(ci.loc[k, 0]), float(ci.loc[k, 1])] for k in fit.fe_params.index},
            "group_var": float(fit.cov_re.iloc[0, 0]),
            "id_slope_L": float(fit.fe_params["L"]),
            "id_slope_ci95": [float(ci.loc["L", 0]), float(ci.loc["L", 1])],
            "changes_with_magnitude": bool(ci.loc["L", 0] > 0 or ci.loc["L", 1] < 0),
            "descriptive_only_terms": ["T", "L:T"],
        })
    except Exception as e:  # §11: report failure, never substitute another model
        out.update({"fit_failed": True, "error": repr(e)[:500]})
    return out


def ols_slope(x, y):
    """Least-squares slope of y on x; y may be (R, n) for replicates."""
    x = np.asarray(x, float)
    y = np.atleast_2d(np.asarray(y, float))
    xc = x - x.mean()
    return (y - y.mean(1, keepdims=True)) @ xc / (xc ** 2).sum()
