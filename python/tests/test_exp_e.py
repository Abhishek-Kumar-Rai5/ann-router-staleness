"""Experiment E tests. They only use synthetic data."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp_e_analysis as A  # noqa: E402
import exp_e_lib as E  # noqa: E402


def test_d_zero_when_ratio_unchanged_and_signs():
    assert E.d_stat(1110, 1000, 1110, 1000) == pytest.approx(0)
    assert E.d_stat(1221, 1000, 1110, 1000) == pytest.approx(0.10)
    assert E.d_stat(999, 1000, 1110, 1000) == pytest.approx(-0.10)
    assert E.d_stat(1300, 1000, 1000, 1000) > 0
    assert E.d_stat(700, 1000, 1000, 1000) < 0


def test_d_equals_r_when_s0_ratio_is_one_and_anchors_s0():
    assert E.d_stat(1050, 1000, 900, 900) == pytest.approx(E.r_stat(1050, 1000))
    assert E.r_stat(1110, 1000) == pytest.approx(0.11)
    assert E.d_stat(1110, 1000, 1110, 1000) == pytest.approx(0)


def test_bootstrap_recomputes_state_and_s0_terms():
    rng = np.random.default_rng(1)
    cp, cr, cp0, cr0 = (rng.uniform(500, 1500, 50) for _ in range(4))
    idx = np.random.default_rng(2).integers(0, 50, size=(20, 50))
    got = E.d_stat(E.rep_mean(cp, idx), E.rep_mean(cr, idx), E.rep_mean(cp0, idx), E.rep_mean(cr0, idx))
    for b in range(20):
        i = idx[b]
        exp = (cp[i].mean() / cr[i].mean()) / (cp0[i].mean() / cr0[i].mean()) - 1
        assert got[b] == pytest.approx(exp)


@pytest.mark.parametrize("ci,cls", [
    ([-0.10, 0.10], "STABLE"), ([-0.05, 0.09], "STABLE"),
    ([0.1000001, 0.3], "STALE-COSTLIER"), ([-0.3, -0.1000001], "STALE-CHEAPER"),
    ([0.10, 0.2], "INCONCLUSIVE"), ([-0.2, -0.10], "INCONCLUSIVE"),
    ([0.05, 0.11], "INCONCLUSIVE"), ([-0.11, 0.11], "INCONCLUSIVE")])
def test_h_e1_classification_at_margin(ci, cls):
    assert E.classify_d(ci) == cls


@pytest.mark.parametrize("ci,cls", [
    ([-0.05, 0.05], "STABLE"), ([0.0500001, 0.1], "STALE-WORSE"), ([-0.2, -0.0500001], "STALE-BETTER"),
    ([0.05, 0.08], "INCONCLUSIVE"), ([0.04, 0.06], "INCONCLUSIVE")])
def test_h_e2_classification_at_margin(ci, cls):
    assert E.classify_e(ci) == cls


def test_family_verdict():
    assert E.family_verdict(["STABLE", "STABLE"]) == "STABLE-FAMILY"
    assert E.family_verdict(["STABLE", "INCONCLUSIVE"]) == "INCONCLUSIVE-FAMILY"
    assert E.family_verdict(["INCONCLUSIVE", "STALE-CHEAPER"]) == "STALE-FAMILY"
    assert E.family_verdict(["STALE-WORSE"]) == "STALE-FAMILY"


def test_transitions_and_rates():
    p0 = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    ps = np.array([1, 1, 0, 0, 0, 1, 0, 1], bool)
    assert E.transitions(p0, ps) == {"stable_pass": 2, "newly_failing": 2, "stable_fail": 2, "improved": 2}
    assert E.nf_rate(p0, ~ps) == pytest.approx(0.5)
    idx = np.array([[0, 0, 2, 4], [1, 2, 3, 3]])
    np.testing.assert_allclose(E.nf_rate(p0, ~ps, idx), [1 / 3, 3 / 4])


def test_is_pass_contract():
    np.testing.assert_array_equal(E.is_pass([1.0, 0.9999999999, 0.9, 1 - 1e-6]), [True, True, False, False])


def test_spearman_rows_matches_scipy_including_ties():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 4, size=(5, 12)).astype(float)
    y = rng.normal(size=(5, 12))
    for r in range(5):
        assert E.spearman_rows(x[r:r + 1], y[r:r + 1])[0] == pytest.approx(stats.spearmanr(x[r], y[r]).statistic)
    assert np.isnan(E.spearman_rows(np.ones((1, 6)), y[:1, :6])[0])


def test_boot_p_and_holm():
    assert E.boot_p_two_sided(np.full(2000, 0.3)) == pytest.approx(1 / 2000)
    assert E.boot_p_two_sided(np.r_[np.full(1000, -1.0), np.full(1000, 1.0)]) == pytest.approx(1.0)
    assert E.boot_p_two_sided(np.r_[np.full(100, -1.0), np.full(1900, 1.0)]) == pytest.approx(0.1)
    from statsmodels.stats.multitest import multipletests
    p = [0.01, 0.04, 0.03, 0.005, 0.2, 0.0005]
    adj, rej = E.holm(p)
    r2, a2, _, _ = multipletests(p, alpha=0.05, method="holm")
    np.testing.assert_allclose(adj, a2)
    np.testing.assert_array_equal(rej, r2)


def test_h_e4_bias_mace_and_stopped_filter():
    stopped = np.array([1, 1, 1, 0, 0], bool)
    pred = np.array([0.96, 0.97, 0.95, 0.5, 0.6])
    real = np.array([1.0, 0.9, 1.0, 0.0, 0.0])
    r = E.h_e4(stopped, pred, real)
    assert r["n_stopped"] == 3 and r["stopped_fraction"] == pytest.approx(0.6)
    assert r["bias"] == pytest.approx(np.mean([-0.04, 0.07, -0.05]))
    assert r["mace"] == pytest.approx(np.mean([0.04, 0.07, 0.05]))
    assert r["small_n"] is True
    z = E.h_e4(np.zeros(5, bool), pred, real)
    assert z["n_stopped"] == 0 and np.isnan(z["bias"]) and z["small_n"] is True
    big = E.h_e4(np.ones(40, bool), np.full(40, 0.96), np.ones(40))
    assert big["small_n"] is False


def _trend_df(seed_shift=0.0):
    rows = []
    for s, u in zip((42, 43, 44), (0.01, -0.02, 0.015), strict=True):
        for L in (0, 1, 2, 3):
            for T in (0, 1):
                rows.append({"seed": s, "L": float(L), "T": T,
                             "y": 0.02 * L + 0.01 * T + u + seed_shift + 0.001 * ((L + T) % 2)})
    return pd.DataFrame(rows)


def test_mixedlm_trend_recovers_slope_and_is_deterministic():
    a = E.mixedlm_trend(_trend_df(), "y")
    b = E.mixedlm_trend(_trend_df(), "y")
    assert "fit_failed" not in a
    assert a["id_slope_L"] == pytest.approx(0.02, abs=2e-3)
    assert a["changes_with_magnitude"] is True
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_mixedlm_trend_reports_failure_without_raising():
    bad = _trend_df().assign(y=np.nan)
    r = E.mixedlm_trend(bad, "y")
    assert r.get("fit_failed") is True and "error" in r


def test_ols_slope():
    assert E.ols_slope([0, 1, 2, 3], [[1, 3, 5, 7]])[0] == pytest.approx(2.0)
    reps = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], float)
    np.testing.assert_allclose(E.ols_slope([0, 1, 2, 3], reps), [1.0, -1.0])


SEEDS = (42, 43, 44)
INS = ["id_10000", "id_20000", "ood_10000", "ood_20000"]
DELS = ["del_20000", "del_80000"]
N = 200
B1_F = {"id_10000": 1.00, "id_20000": 1.02, "ood_10000": 1.30, "ood_20000": 1.10,
        "del_20000": 1.00, "del_80000": 0.70}
DARTH_F = {"id_10000": 1.00, "id_20000": 1.01, "ood_10000": 0.99, "ood_20000": 1.02,
           "del_20000": 1.00, "del_80000": 1.01}
B1_NEWLY_FAIL = {"id_10000": 0, "id_20000": 2, "ood_10000": 40, "ood_20000": 4, "del_20000": 0, "del_80000": 1}
DARTH_NEWLY_FAIL = {"id_10000": 10, "id_20000": 15, "ood_10000": 8, "ood_20000": 20, "del_20000": 5, "del_80000": 12}


def make_fixture():
    qids = np.arange(1000, 1000 + N)
    # Counts are large enough that no bootstrap resample makes every cell identical.
    base = np.repeat(np.linspace(600, 1600, N // 2), 2)
    eps = np.tile([0.05, -0.05], N // 2)
    b1_pass0 = np.arange(N) < 140
    darth_pass0 = np.arange(N) < 150
    rows = []

    def add(pol, s, st, cost, passv, extra=None):
        d = {"policy": pol, "seed": s, "state": st, "query_id": qids, "cost": cost,
             "recall_tie_aware": np.where(passv, 1.0, 0.9), "recall_id": np.where(passv, 1.0, 0.9)}
        rows.append(pd.DataFrame({**d, **(extra or {})}))

    for s in SEEDS:
        for st in ["S0", *INS, *DELS]:
            g = 1.0 if st == "S0" else 1.0 + 0.01 * (INS + DELS).index(st)
            ref = base * g
            ref_pass = b1_pass0.copy()
            if st != "S0":
                ref_pass[:1] = False
            b1 = ref if st == "S0" else ref * B1_F[st] * (1 + eps)
            b1_pass = b1_pass0.copy()
            if st != "S0":
                b1_pass[:B1_NEWLY_FAIL[st]] = False
            dr = ref * 1.11 * (1 + eps) * (1.0 if st == "S0" else DARTH_F[st])
            d_pass = darth_pass0.copy()
            if st != "S0":
                d_pass[:DARTH_NEWLY_FAIL[st]] = False
            stopped = (np.arange(N) % 10) != 0
            add("REF", s, st, ref, ref_pass)
            add("B1", s, st, b1, b1_pass)
            add("DARTH", s, st, dr, d_pass, {
                "stopped_early": stopped.astype(int), "last_predicted": np.where(stopped, 0.96, 0.9),
                "predictor_calls": np.full(N, 8), "predictor_inferences": np.full(N, 8),
                "predictor_seconds": np.full(N, 1e-4), "wall_seconds": np.full(N, 5e-4)})
    rows = pd.concat(rows, ignore_index=True)
    pq = []
    for s in SEEDS:
        for k, st in enumerate(INS + DELS):
            pq.append(pd.DataFrame({"seed": s, "state": st, "query_id": qids,
                                    "overlap_decay": 0.02 * (k + 1) + 0.001 * (np.arange(N) % 7),
                                    "density_drift": 0.001 * (k % 3) + 0.0001 * (np.arange(N) % 5)}))
    pq = pd.concat(pq, ignore_index=True)
    ps = pd.DataFrame({"update_fraction": [int(st.split("_")[1]) / 1e6 for st in INS + DELS],
                       "distributional_drift": [0.01, 0.009, 0.66, 0.63, 0.0001, 0.0002]}, index=INS + DELS)
    cfg = {"seeds": list(SEEDS), "insertion_states": INS, "deletion_states": DELS, "n0": 1_000_000, "n_queries": N}
    return rows, pq, ps, cfg


@pytest.fixture(scope="module")
def result():
    return A.analyze(*make_fixture())


def test_end_to_end_h_e1(result):
    b1 = result["H-E1"]["B1"]["cells"]
    assert b1["id_10000"]["D_bar"] == pytest.approx(0, abs=1e-12) and b1["id_10000"]["class"] == "STABLE"
    assert b1["id_20000"]["class"] == "STABLE"
    assert b1["ood_10000"]["D_bar"] == pytest.approx(0.30) and b1["ood_10000"]["class"] == "STALE-COSTLIER"
    assert b1["ood_20000"]["D_bar"] == pytest.approx(0.10) and b1["ood_20000"]["class"] == "INCONCLUSIVE"
    assert b1["del_80000"]["D_bar"] == pytest.approx(-0.30) and b1["del_80000"]["class"] == "STALE-CHEAPER"
    assert result["H-E1"]["B1"]["family_verdict"] == {"insertion": "STALE-FAMILY", "deletion": "STALE-FAMILY"}
    ps = b1["ood_10000"]["per_seed"][42]
    assert ps["D"] == pytest.approx(ps["R_descriptive"])
    d = result["H-E1"]["DARTH"]["cells"]
    assert d["id_10000"]["per_seed"][42]["R_descriptive"] == pytest.approx(0.11)
    assert d["id_10000"]["D_bar"] == pytest.approx(0, abs=1e-12)
    assert all(d[st]["class"] == "STABLE" for st in INS + DELS)
    assert result["H-E1"]["DARTH"]["family_verdict"] == {"insertion": "STABLE-FAMILY", "deletion": "STABLE-FAMILY"}
    assert "mean_recall_tie_aware" in result and "B1|ood_10000" in result["mean_recall_tie_aware"]


def test_end_to_end_h_e2(result):
    c = result["H-E2"]["B1"]["cells"]
    ps = c["ood_10000"]["per_seed"][42]
    assert ps["cohort_size_frozen"] == 140 and ps["cohort_size_reference"] == 140
    assert ps["transitions_frozen"] == {"stable_pass": 100, "newly_failing": 40, "stable_fail": 60, "improved": 0}
    assert ps["NF_frozen"] == pytest.approx(40 / 140) and ps["NF_reference"] == pytest.approx(1 / 140)
    assert c["ood_10000"]["E_bar"] == pytest.approx(39 / 140) and c["ood_10000"]["class"] == "STALE-WORSE"
    assert c["id_10000"]["E_bar"] == pytest.approx(-1 / 140)
    dc = result["H-E2"]["DARTH"]["cells"]["ood_20000"]["per_seed"][43]
    assert dc["cohort_size_frozen"] == 150 and dc["E"] == pytest.approx(20 / 150 - 1 / 140)


def test_end_to_end_h_e3_h_e4_trends(result):
    t = result["H-E3"]["primary_insertion_tests"]
    assert len(t) == 16 and {x["n_cells"] for x in t} == {12}
    assert all(0 < x["p_holm"] <= 1 and x["p_holm"] >= x["p_boot"] for x in t)
    assert len(result["H-E3"]["deletion_descriptive"]) == 16
    h4 = result["H-E4"]["primary_id"]
    s0 = h4["S0"]["per_seed"][42]
    assert s0["stopped_fraction"] == pytest.approx(0.9) and s0["n_stopped"] == 180
    exp_bias = (135 * (0.96 - 1.0) + 45 * (0.96 - 0.9)) / 180
    assert s0["bias"] == pytest.approx(exp_bias)
    assert s0["mace"] == pytest.approx((135 * 0.04 + 45 * 0.06) / 180)
    assert "bias_change_from_S0" in h4["ood_10000"]["per_seed"][42]
    tr = result["trends_secondary"]
    assert set(tr) == {"B1|D", "B1|E", "DARTH|D", "DARTH|E"}
    assert "robustness_ols_slope_pooled" in tr["B1|D"]
    assert tr["B1|D"]["deletion_difference_descriptive"]["point"] == pytest.approx(-0.30)


def test_end_to_end_deterministic():
    a = A.analyze(*make_fixture())
    b = A.analyze(*make_fixture())
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def test_validation_hard_fails():
    rows, pq, ps, cfg = make_fixture()
    drop = rows[~((rows.policy == "DARTH") & (rows.seed == 43) & (rows.state == "del_80000"))]
    with pytest.raises(A.AnalysisInputError, match="DARTH seed 43 del_80000"):
        A.analyze(drop, pq, ps, cfg)
    dup = rows.copy()
    dup.loc[dup.index[1], "query_id"] = dup.loc[dup.index[0], "query_id"]
    with pytest.raises(A.AnalysisInputError, match="duplicate"):
        A.analyze(dup, pq, ps, cfg)
    bad_ref = rows.copy()
    m = (bad_ref.policy == "REF") & (bad_ref.seed == 42) & (bad_ref.state == "S0")
    bad_ref.loc[m, "cost"] = bad_ref.loc[m, "cost"] * 1.01
    with pytest.raises(A.AnalysisInputError, match="REF\\(S0\\) differs"):
        A.analyze(bad_ref, pq, ps, cfg)
    with pytest.raises(A.AnalysisInputError, match="predictors"):
        A.analyze(rows, pq[pq.state != "ood_20000"], ps, cfg)
    with pytest.raises(A.AnalysisInputError, match="missing column"):
        A.analyze(rows.drop(columns=["recall_id"]), pq, ps, cfg)
    with pytest.raises(A.AnalysisInputError, match="expected"):
        A.analyze(rows, pq, ps, {**cfg, "n_queries": 2000})


def test_degenerate_h_e3_input_stops():
    rows, pq, ps, cfg = make_fixture()
    pq = pq.assign(overlap_decay=0.5)
    ps = ps.assign(update_fraction=0.01, distributional_drift=0.1)
    with pytest.raises(A.AnalysisInputError, match="undefined"):
        A.analyze(rows, pq, ps, cfg)
