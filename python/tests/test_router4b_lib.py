import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import router4b_lib as rb  # noqa: E402

CFG = {"families": {
    "random_state": 20261002,
    "LR": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000},
    "DT2": {"max_depth": 2, "min_samples_leaf": 100, "criterion": "gini"},
    "DT3": {"max_depth": 3, "min_samples_leaf": 100, "criterion": "gini"},
    "DT4": {"max_depth": 4, "min_samples_leaf": 100, "criterion": "gini"}}}


def test_candidate_oracle_index_stable_reach_and_beyond_last():
    s = np.array([[1, 1, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [0, 0, 0],
                  [0, 1, 0]],
                 dtype=bool)
    assert rb.candidate_oracle_index(s).tolist() == [0, 1, 2, 3, 3]


def test_choose_cheapest_and_fallback_to_highest():
    P = np.array([[0.2, 0.6, 0.9],
                  [0.95, 0.96, 0.99],
                  [0.1, 0.2, 0.3]])
    c, fb = rb.choose(P, 0.5)
    assert c.tolist() == [1, 0, 2] and fb.tolist() == [False, False, True]


def test_calibrate_tau_smallest_meeting_target():
    P = np.array([[0.2, 0.6, 1.0], [0.9, 0.95, 1.0], [0.1, 0.3, 1.0],
                  [0.5, 0.8, 1.0]])
    succ = np.array([[0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1]])
    tau, q = rb.calibrate_tau(P, succ, 0.75)
    c, _ = rb.choose(P, tau)
    assert succ[np.arange(4), c].mean() >= 0.75 and q >= 0.75
    c_prev, _ = rb.choose(P, tau - 0.001)
    assert succ[np.arange(4), c_prev].mean() < 0.75
    assert rb.calibrate_tau(P, succ * 0, 0.5) == (None, 0.0)


def test_one_class_target_gives_constant_model():
    X = np.random.default_rng(0).normal(size=(50, 3))
    m, oc = rb.fit_binary("LR", CFG, X, np.ones(50, int))
    assert oc and rb.proba1(m, X).tolist() == [1.0] * 50
    m, oc = rb.fit_binary("DT2", CFG, X, np.zeros(50, int))
    assert oc and rb.proba1(m, X).max() == 0.0


def test_predict_candidates_is_monotone():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 3))
    y = np.digitize(X[:, 0] + rng.normal(scale=0.5, size=600), [-1, 0, 1])
    models, oc = rb.fit_candidate_models("DT3", CFG, X, y, 4)
    P = rb.predict_candidates(models, X)
    assert (np.diff(P, axis=1) >= 0).all() and oc == 1


@pytest.mark.parametrize("name", ["LR", "DT2", "DT3", "DT4"])
def test_export_reproduces_probabilities(name):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(800, 3))
    t = (X[:, 0] + 0.3 * X[:, 1] + rng.normal(scale=0.5, size=800) > 0).astype(int)
    m, _ = rb.fit_binary(name, CFG, X, t)
    e = rb.export_binary(m, ["a", "b", "c"])
    Xt = rng.normal(size=(300, 3))
    assert np.allclose(rb.proba_from_export(e, Xt), rb.proba1(m, Xt), atol=1e-12)
    assert rb.proba_from_export(rb.export_binary(rb.ConstantModel(1.0), []),
                                Xt).tolist() == [1.0] * 300


def test_select_family_tie_goes_to_simplest():
    order = ["DT2", "DT3", "LR", "DT4"]
    rows = [{"name": "DT2", "cost": 999, "feasible": True},
            {"name": "LR", "cost": 1000, "feasible": True},
            {"name": "DT4", "cost": 990, "feasible": True}]
    assert rb.select_family(rows, 0.01, order) == "DT2"
    rows[0]["cost"] = 1009
    assert rb.select_family(rows, 0.01, order) == "DT4"
    rows[2]["feasible"] = False
    assert rb.select_family(rows, 0.01, order) == "DT2"
    assert rb.select_family([{"name": "LR", "cost": 1, "feasible": False}],
                            0.01, order) is None


def test_splitmix64_reference_vector_and_mix_determinism():
    assert rb.splitmix64(0) == 0xE220A8397B1DCDAF
    assert rb.fnv1a64("") == 1469598103934665603
    u = [rb.mix_uniform("sift_query", r, 20261006) for r in range(20000)]
    assert u == [rb.mix_uniform("sift_query", r, 20261006) for r in range(20000)]
    assert 0 <= min(u) and max(u) < 1 and abs(np.mean(u) - 0.5) < 0.01


def test_two_ef_mix_hits_target_in_expectation():
    level = np.array([0.80, 0.90, 0.96, 0.99])
    mix = rb.two_ef_mix(level, [10, 20, 40, 80], 0.95)
    assert (mix["ef_lo"], mix["ef_hi"]) == (20, 40)
    assert 0.90 + mix["w_hi"] * 0.06 == pytest.approx(0.95)
    assert rb.two_ef_mix(level, [10, 20, 40, 80], 0.90)["w_hi"] == 1.0
    assert rb.two_ef_mix(level, [10, 20, 40, 80], 0.999) is None
    rows = np.arange(20000)
    a = rb.assign_mix(mix, rows, "sift_query", 20261006)
    assert abs((a == 40).mean() - mix["w_hi"]) < 0.02


def test_tree_export_matches_sklearn_float32_split_semantics():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 1)) * 100 + 200
    t = (X[:, 0] > 200).astype(int)
    m, _ = rb.fit_binary("DT2", CFG, X, t)
    e = rb.export_binary(m, ["a"])
    thr = e["threshold"][0]
    probes = []
    x = np.float64(thr)
    for _ in range(200):
        x = np.nextafter(x, np.inf)
        if np.float64(np.float32(x)) <= thr:
            probes.append(x)
    assert probes, "no float32/float64 gap value found"
    Xp = np.array(probes)[:, None]
    assert np.array_equal(rb.proba_from_export(e, Xp), rb.proba1(m, Xp))
