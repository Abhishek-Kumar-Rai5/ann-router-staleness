import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import router_lib as rl  # noqa: E402

TIER_EF = {"low": 18, "med": 48, "high": 327}


def toy_curves():
    efs = [10, 18, 48, 327]
    rec = pd.DataFrame([[0.5, 0.9, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [0.2, 0.4, 0.8, 0.9]],
                       index=[0, 1, 2], columns=efs)
    dc = pd.DataFrame([[100, 200, 400, 2000]] * 3, index=[0, 1, 2], columns=efs)
    return rl.Curves(recall=rec, recall_id=rec.copy(), dc=dc)


def test_route_looks_up_tier_ef_and_adds_probe_cost():
    c = toy_curves()
    out = rl.route(["low", "med", "high"], [0, 1, 2], [50, 60, 70], c,
                   TIER_EF, 0.95)
    assert out["ef"].tolist() == [18, 48, 327]
    assert out["recall"].tolist() == [0.9, 1.0, 0.9]
    assert out["search_dc"].tolist() == [200, 400, 2000]
    assert out["total_dc"].tolist() == [250, 460, 2070]
    assert out["failure"].tolist() == [True, False, True]


def test_fixed_baseline_pays_no_probe():
    out = rl.fixed([0, 1], 48, toy_curves(), 0.95)
    assert out["probe_dc"].tolist() == [0.0, 0.0]
    assert out["total_dc"].tolist() == out["search_dc"].tolist()


def test_lookup_rejects_unknown_query_or_ef():
    with pytest.raises(KeyError):
        toy_curves().lookup([5], [18])
    with pytest.raises(KeyError):
        toy_curves().lookup([0], [19])


def test_fixed_curve_matching_and_interpolation():
    c = toy_curves()
    curve = rl.fixed_curve(c, [0, 1, 2], 0.95)
    assert rl.cheapest_fixed_ef(curve, 0.9) == 48
    assert rl.cheapest_fixed_ef(curve, 0.95) == 327
    assert rl.cheapest_fixed_ef(curve, 0.99) is None
    assert rl.interp_cost_at_recall(curve, 0.99) == float("inf")
    r48 = curve.loc[curve["ef"] == 48, "mean_recall"].iloc[0]
    assert rl.interp_cost_at_recall(curve, r48) == pytest.approx(400)
    mid = (r48 + curve.loc[curve["ef"] == 327, "mean_recall"].iloc[0]) / 2
    assert rl.interp_cost_at_recall(curve, mid) == pytest.approx(1200)
    assert rl.interp_cost_at_recall(curve, 0.1) == pytest.approx(100)
    assert curve["failure_rate"].tolist() == pytest.approx([2 / 3, 2 / 3, 1 / 3, 1 / 3])


def test_regret_signs_and_censoring():
    c = toy_curves()
    routed = rl.route(["high", "low", "high"], [0, 1, 2], [50, 50, 50], c,
                      TIER_EF, 0.95)
    r = rl.regret(routed, [48, 10, np.nan], [True, True, False], c)
    assert r["regret_search_dc"].iloc[0] == 2000 - 400
    assert r["regret_total_dc"].iloc[0] == 2050 - 400
    assert r["regret_search_dc"].iloc[1] == 200 - 100
    assert np.isnan(r["regret_total_dc"].iloc[2])


def test_tier_error_direction():
    e = rl.tier_error(["high", "low", "med", "low"], ["low", "high", "med", "med"])
    assert e.tolist() == [2, -2, 0, -1]


def test_select_candidate_constraint_score_and_tie_break():
    s = pd.DataFrame({"name": ["DT2", "DT3", "LR", "DT4"],
                      "score": [1.50, 1.51, 1.70, 1.71],
                      "feasible": [True, True, True, True]})
    assert rl.select_candidate(s, 0.02) == "LR"
    s.loc[s["name"] == "LR", "feasible"] = False
    assert rl.select_candidate(s, 0.02) == "DT4"
    assert rl.select_candidate(s.assign(score=1.0), 0.02) == "DT2"
    with pytest.raises(RuntimeError):
        rl.select_candidate(s.assign(feasible=False), 0.02)


def test_load_split_frame_never_returns_other_split(tmp_path):
    f = pd.DataFrame({"query_id": [0, 1, 2, 3],
                      "split": ["train", "test", "train", "test"],
                      "knn_dist": [1.0, 2, 3, 4]})
    s = pd.DataFrame({"query_id": [0, 1, 2, 3],
                      "split": ["train", "test", "train", "test"],
                      "tier_label": ["low", "med", "high", "low"],
                      "difficulty": ["easy"] * 4, "censored": [False] * 4,
                      "oracle_ef": [10, 20, 30, 40], "reached": [1] * 4})
    f.to_csv(tmp_path / "f.csv", index=False)
    s.to_csv(tmp_path / "s.csv", index=False)
    tr = rl.load_split_frame(tmp_path / "f.csv", tmp_path / "s.csv", "train")
    assert tr["query_id"].tolist() == [0, 2]
    assert set(tr["split"]) == {"train"}


@pytest.mark.parametrize("name", ["DT2", "DT3", "DT4", "LR", "SINGLE"])
def test_export_reproduces_sklearn_predictions(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = np.array(rl.TIERS)[np.digitize(X[:, 0] + 0.5 * X[:, 1], [-0.5, 0.5])]
    feats = ["a", "b", "c"]
    m = rl.make_candidate(name, 1)
    Xf = X[:, :1] if name == "SINGLE" else X
    m.fit(Xf, y)
    exp = rl.export_model(m, feats[:Xf.shape[1]])
    Xt = rng.normal(size=(300, Xf.shape[1]))
    assert (rl.predict_from_export(exp, Xt) == m.predict(Xt)).all()


def test_tree_rules_render():
    X = np.array([[0.0], [1.0], [2.0], [3.0]] * 50)
    y = np.array(["low", "low", "high", "high"] * 50)
    m = rl.make_candidate("SINGLE", 0).fit(X, y)
    text = "\n".join(rl.tree_rules(rl.export_model(m, ["knn_dist"])))
    assert "if knn_dist <=" in text and "-> low" in text and "-> high" in text


def test_wilcoxon_rank_biserial_known_cases():
    a = np.arange(1, 21, dtype=float)
    r = rl.wilcoxon_paired(a + 1, a)
    assert r["rank_biserial"] == pytest.approx(1.0)
    assert r["p_value"] < 1e-4
    r = rl.wilcoxon_paired(a, a)
    assert r["n_nonzero"] == 0 and r["p_value"] == 1.0


def test_bootstrap_ci_deterministic_and_covers_mean():
    v = np.random.default_rng(1).normal(5, 1, 500)
    ci1 = rl.bootstrap_ci(v, 500, 7)
    ci2 = rl.bootstrap_ci(v, 500, 7)
    assert ci1 == ci2
    assert ci1[0] < v.mean() < ci1[1]
