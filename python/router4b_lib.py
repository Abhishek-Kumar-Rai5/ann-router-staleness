"""Phase 4b router logic (frozen contract: docs/phase4_redesign.md as amended by
docs/phase4_bias_audit.md §10 and the user's Phase 4b gap resolutions).

Pure functions on NumPy arrays so every rule is unit-tested
(python/tests/test_router4b_lib.py).

Candidates are ordered by cost: index 0 = "probe result", then the frozen
ladder 20 .. 2661. Label y_i = candidate-oracle index (stable reach within
candidates); y_i = M ("beyond last") if no candidate satisfies the target.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MASK64 = (1 << 64) - 1


# ------------------------------------------------------------- labels -------

def candidate_oracle_index(success: np.ndarray) -> np.ndarray:
    """success: (n, M) bool. Smallest j such that success holds at j and at
    every larger candidate; M if the last candidate fails."""
    n, m = success.shape
    y = np.full(n, m)
    ok = np.ones(n, bool)
    for j in range(m - 1, -1, -1):
        ok &= success[:, j]
        y = np.where(ok, j, y)
    return y


# ------------------------------------------------------------- models -------

class ConstantModel:
    """Used when a training target has one class (approved gap resolution 2):
    predicts the observed class probability (0.0 or 1.0)."""

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba1(self, X):
        return np.full(len(X), self.p)


def make_family(name: str, cfg: dict):
    fam = cfg["families"][name]
    rs = cfg["families"]["random_state"]
    if name == "LR":
        return Pipeline([("scale", StandardScaler()),
                         ("lr", LogisticRegression(C=fam["C"], solver=fam["solver"],
                                                   max_iter=fam["max_iter"]))])
    return DecisionTreeClassifier(max_depth=fam["max_depth"],
                                  min_samples_leaf=fam["min_samples_leaf"],
                                  criterion=fam["criterion"], random_state=rs)


def fit_binary(name, cfg, X, t):
    """Fit P(t=1|x). One-class target -> ConstantModel (flag returned)."""
    if t.min() == t.max():
        return ConstantModel(float(t[0])), True
    m = make_family(name, cfg).fit(X, t)
    return m, False


def proba1(model, X) -> np.ndarray:
    if isinstance(model, ConstantModel):
        return model.predict_proba1(X)
    p = model.predict_proba(X)
    classes = list(model.classes_)
    return p[:, classes.index(1)]


def fit_candidate_models(name, cfg, X, y, m):
    """One binary model per candidate j: target 1[y <= j]."""
    models, one_class = [], 0
    for j in range(m):
        mod, oc = fit_binary(name, cfg, X, (y <= j).astype(int))
        models.append(mod)
        one_class += int(oc)
    return models, one_class


def predict_candidates(models, X) -> np.ndarray:
    """(n, M) probabilities with monotonicity enforced by a cumulative max."""
    P = np.stack([proba1(mod, X) for mod in models], axis=1)
    return np.maximum.accumulate(P, axis=1)


# -------------------------------------------------------- decision rule -----

def choose(P: np.ndarray, tau: float):
    """Cheapest candidate with P >= tau; if none, the highest candidate
    (approved gap resolution 1). Returns (choice index, fallback mask)."""
    ok = P >= tau - 1e-12
    any_ok = ok.any(axis=1)
    first = np.argmax(ok, axis=1)
    choice = np.where(any_ok, first, P.shape[1] - 1)
    return choice, ~any_ok


def calibrate_tau(P, quality_by_cand, target, step=0.001):
    """Smallest tau on a `step` grid whose policy quality (mean over queries
    of quality_by_cand[i, choice_i]) >= target. quality_by_cand is the per-
    query success indicator (per-query contract) or recall (mean contract).
    Returns (tau, achieved quality) or (None, best achievable)."""
    taus = np.round(np.arange(0, 1 + step / 2, step), 10)
    idx = np.arange(len(P))
    best = -1.0
    for tau in taus:
        c, _ = choose(P, tau)
        qv = float(quality_by_cand[idx, c].mean())
        best = max(best, qv)
        if qv >= target - 1e-12:
            return float(tau), qv
    return None, best


def select_family(rows, tie_rel, order):
    """rows: list of dicts with name, cost, feasible. Lowest cost among
    feasible; candidates within tie_rel (relative) of the best -> simplest."""
    feas = [r for r in rows if r["feasible"]]
    if not feas:
        return None
    best = min(r["cost"] for r in feas)
    near = [r for r in feas if r["cost"] <= best * (1 + tie_rel) + 1e-12]
    return sorted(near, key=lambda r: order.index(r["name"]))[0]["name"]


# ---------------------------------------------------- B1 two-ef mixture ------

def fnv1a64(s: str) -> int:
    h = 1469598103934665603
    for b in s.encode():
        h = ((h ^ b) * 1099511628211) & MASK64
    return h


def splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def mix_uniform(query_set_id: str, row: int, seed: int) -> float:
    """u in [0,1): splitmix64(fnv1a64("<query_set_id>:<row>") XOR seed) / 2^64."""
    return splitmix64(fnv1a64(f"{query_set_id}:{row}") ^ seed) / 2.0 ** 64


def two_ef_mix(level_by_ef, grid, target):
    """Bracketing grid ef values and weight on the upper one so the expected
    train quality equals `target` exactly (w = 1 if a grid ef hits it)."""
    ok = np.where(level_by_ef >= target - 1e-12)[0]
    if len(ok) == 0:
        return None
    j = int(ok[0])
    if j == 0 or abs(level_by_ef[j] - target) < 1e-15:
        return {"ef_lo": int(grid[j]), "ef_hi": int(grid[j]), "w_hi": 1.0}
    lo = j - 1
    w = (target - level_by_ef[lo]) / (level_by_ef[j] - level_by_ef[lo])
    return {"ef_lo": int(grid[lo]), "ef_hi": int(grid[j]), "w_hi": float(w)}


def assign_mix(mix, rows, query_set_id, seed):
    u = np.array([mix_uniform(query_set_id, int(r), seed) for r in rows])
    return np.where(u < mix["w_hi"], mix["ef_hi"], mix["ef_lo"])


# --------------------------------------------------------------- export -----

def export_binary(model, features):
    if isinstance(model, ConstantModel):
        return {"type": "constant", "p": model.p}
    if isinstance(model, DecisionTreeClassifier):
        t = model.tree_
        val = t.value[:, 0, :]
        frac = val / val.sum(axis=1, keepdims=True)
        return {"type": "decision_tree", "features": list(features),
                "classes": [int(c) for c in model.classes_],
                "children_left": t.children_left.tolist(),
                "children_right": t.children_right.tolist(),
                "feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
                "leaf_p1": frac[:, list(model.classes_).index(1)].tolist()}
    sc, lr = model.named_steps["scale"], model.named_steps["lr"]
    return {"type": "logistic_regression", "features": list(features),
            "scale_mean": sc.mean_.tolist(), "scale_std": sc.scale_.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


def proba_from_export(e, X) -> np.ndarray:
    """sklearn-free re-implementation of P(t=1|x) from export_binary()."""
    if e["type"] == "constant":
        return np.full(len(X), e["p"])
    if e["type"] == "decision_tree":
        # sklearn trees cast inputs to float32 before comparing with the
        # (float64) split thresholds; mirror that exactly.
        X = np.asarray(X, dtype=np.float32).astype(np.float64)
        out = np.empty(len(X))
        for i, x in enumerate(X):
            node = 0
            while e["children_left"][node] != -1:
                node = (e["children_left"][node]
                        if x[e["feature"][node]] <= e["threshold"][node]
                        else e["children_right"][node])
            out[i] = e["leaf_p1"][node]
        return out
    z = (X - np.array(e["scale_mean"])) / np.array(e["scale_std"])
    return 1.0 / (1.0 + np.exp(-(z @ np.array(e["coef"]) + e["intercept"])))
