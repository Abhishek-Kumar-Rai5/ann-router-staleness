"""Shared router training / evaluation logic (Phase 4).

Pure functions on DataFrames so that every piece of routing and evaluation
logic is unit-tested (python/tests/test_router_lib.py). Definitions follow
the Phase 4 pre-registration in docs/notes.md.

Conventions
  tiers      ordered ("low", "med", "high") -> frozen ef values from
             derived/sift1m_s0_effort_tiers.json (18 / 48 / 327).
  curves     per-query search curves: DataFrame indexed by query_id with
             MultiIndex-free wide tables recall[q, ef] and dc[q, ef].
  cost       router total cost = probe distance computations + search
             distance computations at the chosen ef. Fixed-ef baselines and
             the oracle pay no probe.
  failure    tie-aware recall@10 < target (0.95).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TIERS = ("low", "med", "high")
TIER_INDEX = {t: i for i, t in enumerate(TIERS)}
CORE_FEATURES = ["knn_dist", "centroid_dist", "score_concentration"]
LID_FEATURE = "lid"
TOL = 1e-9

# Simplicity order used for tie-breaking (fewest parameters first).
SIMPLICITY = ["DT2", "DT3", "LR", "DT4"]


# --------------------------------------------------------------- data ------

@dataclass
class Curves:
    """Wide per-query curves: rows = query_id, columns = ef."""
    recall: pd.DataFrame          # tie-aware recall@10
    recall_id: pd.DataFrame       # id-based recall@10
    dc: pd.DataFrame              # search distance computations

    def lookup(self, query_ids, efs):
        q = np.asarray(query_ids)
        e = np.asarray(efs)
        rows = self.recall.index.get_indexer(q)
        cols = self.recall.columns.get_indexer(e)
        if (rows < 0).any() or (cols < 0).any():
            raise KeyError("query or ef missing from curves")
        return (self.recall.to_numpy()[rows, cols],
                self.recall_id.to_numpy()[rows, cols],
                self.dc.to_numpy()[rows, cols])


def load_curves(oracle_curves_csv, split: str) -> Curves:
    """Curves for ONE split only; rows of the other split are never kept."""
    df = pd.read_csv(oracle_curves_csv)
    df = df[df["split"] == split]
    assert set(df["split"]) == {split}
    return Curves(
        recall=df.pivot(index="query_id", columns="ef", values="recall_tie_aware"),
        recall_id=df.pivot(index="query_id", columns="ef", values="recall"),
        dc=df.pivot(index="query_id", columns="ef", values="distance_computations"))


def load_split_frame(features_csv, strata_csv, split: str) -> pd.DataFrame:
    """Features + tier label + difficulty for ONE split. The other split's
    rows are dropped before the merge, so they cannot reach the caller."""
    f = pd.read_csv(features_csv)
    s = pd.read_csv(strata_csv)
    f = f[f["split"] == split]
    s = s[s["split"] == split]
    df = f.merge(s[["query_id", "tier_label", "difficulty", "censored",
                    "oracle_ef", "reached"]], on="query_id", validate="1:1")
    assert set(df["split"]) == {split} and len(df) == len(f)
    return df.sort_values("query_id").reset_index(drop=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- models ------

def make_candidate(name: str, seed: int):
    if name.startswith("DT"):
        return DecisionTreeClassifier(max_depth=int(name[2:]),
                                      min_samples_leaf=100, random_state=seed)
    if name == "LR":
        return Pipeline([("scale", StandardScaler()),
                         ("lr", LogisticRegression(C=1.0, max_iter=2000))])
    if name == "SINGLE":
        return DecisionTreeClassifier(max_leaf_nodes=3, random_state=seed)
    raise ValueError(f"unknown candidate {name}")


def export_model(model, features) -> dict:
    """Plain-JSON description sufficient to re-evaluate the model without
    sklearn (used by predict_from_export for independent validation)."""
    if isinstance(model, DecisionTreeClassifier):
        t = model.tree_
        return {"type": "decision_tree", "features": list(features),
                "classes": [str(c) for c in model.classes_],
                "children_left": t.children_left.tolist(),
                "children_right": t.children_right.tolist(),
                "feature": t.feature.tolist(),
                "threshold": t.threshold.tolist(),
                "value": t.value[:, 0, :].tolist()}
    if isinstance(model, Pipeline):
        sc, lr = model.named_steps["scale"], model.named_steps["lr"]
        return {"type": "logistic_regression", "features": list(features),
                "classes": [str(c) for c in lr.classes_],
                "scale_mean": sc.mean_.tolist(), "scale_std": sc.scale_.tolist(),
                "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()}
    raise TypeError(type(model))


def predict_from_export(export: dict, X: np.ndarray) -> np.ndarray:
    """Independent re-implementation of prediction from export_model() JSON."""
    classes = np.array(export["classes"])
    if export["type"] == "decision_tree":
        out = []
        for x in X:
            node = 0
            while export["children_left"][node] != -1:
                f = export["feature"][node]
                node = (export["children_left"][node]
                        if x[f] <= export["threshold"][node]
                        else export["children_right"][node])
            out.append(classes[int(np.argmax(export["value"][node]))])
        return np.array(out)
    z = (X - np.array(export["scale_mean"])) / np.array(export["scale_std"])
    logits = z @ np.array(export["coef"]).T + np.array(export["intercept"])
    return classes[np.argmax(logits, axis=1)]


def tree_rules(export: dict, node: int = 0, depth: int = 0) -> list[str]:
    """Human-readable if/else rules of an exported decision tree."""
    pad = "  " * depth
    if export["children_left"][node] == -1:
        v = np.array(export["value"][node])
        cls = export["classes"][int(np.argmax(v))]
        frac = ", ".join(f"{c}={x:.3f}" for c, x in
                         zip(export["classes"], v / v.sum()))
        return [f"{pad}-> {cls}  (train class fractions: {frac})"]
    f = export["features"][export["feature"][node]]
    th = export["threshold"][node]
    return ([f"{pad}if {f} <= {th:.6g}:"]
            + tree_rules(export, export["children_left"][node], depth + 1)
            + [f"{pad}else:  # {f} > {th:.6g}"]
            + tree_rules(export, export["children_right"][node], depth + 1))


# ------------------------------------------------------------ routing ------

def route(pred_tiers, query_ids, probe_dc, curves: Curves, tier_ef: dict,
          target: float) -> pd.DataFrame:
    """Per-query outcome of a routing decision (one tier per query)."""
    efs = np.array([tier_ef[t] for t in pred_tiers])
    rec, rec_id, dc = curves.lookup(query_ids, efs)
    probe = np.asarray(probe_dc, dtype=float)
    return pd.DataFrame({
        "query_id": np.asarray(query_ids), "tier": list(pred_tiers), "ef": efs,
        "recall": rec, "recall_id": rec_id, "search_dc": dc,
        "probe_dc": probe, "total_dc": dc + probe,
        "failure": rec < target - TOL})


def fixed(query_ids, ef: int, curves: Curves, target: float) -> pd.DataFrame:
    efs = np.full(len(query_ids), ef)
    rec, rec_id, dc = curves.lookup(query_ids, efs)
    return pd.DataFrame({
        "query_id": np.asarray(query_ids), "tier": "", "ef": efs,
        "recall": rec, "recall_id": rec_id, "search_dc": dc,
        "probe_dc": 0.0, "total_dc": dc.astype(float),
        "failure": rec < target - TOL})


def fixed_curve(curves: Curves, query_ids, target: float) -> pd.DataFrame:
    """Mean recall, mean search cost and failure rate of every fixed grid ef
    on a subset of queries."""
    r = curves.recall.loc[query_ids]
    d = curves.dc.loc[query_ids]
    fails = (r < target - TOL).mean()
    return pd.DataFrame({"ef": r.columns.to_numpy(),
                         "mean_recall": r.mean().to_numpy(),
                         "mean_dc": d.mean().to_numpy(),
                         "failure_rate": fails.to_numpy()})


def cheapest_fixed_ef(curve: pd.DataFrame, recall: float):
    """Smallest grid ef whose mean recall >= `recall` (None if unreachable)."""
    ok = curve[curve["mean_recall"] >= recall - TOL]
    return None if ok.empty else int(ok["ef"].iloc[0])


def interp_cost_at_recall(curve: pd.DataFrame, recall: float) -> float:
    """Fixed-ef mean cost at a target mean recall, linearly interpolated
    between the last grid point below and the first at/above it."""
    c = curve.sort_values("ef").reset_index(drop=True)
    idx = np.where(c["mean_recall"].to_numpy() >= recall - TOL)[0]
    if len(idx) == 0:
        return float("inf")
    i = int(idx[0])
    if i == 0:
        return float(c.loc[0, "mean_dc"])
    r0, r1 = c.loc[i - 1, "mean_recall"], c.loc[i, "mean_recall"]
    d0, d1 = c.loc[i - 1, "mean_dc"], c.loc[i, "mean_dc"]
    w = 0.0 if r1 == r0 else (recall - r0) / (r1 - r0)
    return float(d0 + w * (d1 - d0))


def regret(router: pd.DataFrame, oracle_ef, reached, curves: Curves) -> pd.DataFrame:
    """Per-query routing regret vs the oracle. Censored queries (oracle effort
    unknown, > max grid ef) get NaN regret and are reported separately."""
    oracle_ef = np.asarray(oracle_ef, dtype=float)
    reached = np.asarray(reached, dtype=bool)
    oracle_dc = np.full(len(router), np.nan)
    if reached.any():
        _, _, d = curves.lookup(router["query_id"].to_numpy()[reached],
                                oracle_ef[reached].astype(int))
        oracle_dc[reached] = d
    return pd.DataFrame({
        "query_id": router["query_id"].to_numpy(),
        "oracle_dc": oracle_dc,
        "regret_total_dc": router["total_dc"].to_numpy() - oracle_dc,
        "regret_search_dc": router["search_dc"].to_numpy() - oracle_dc})


def tier_error(pred_tiers, true_tiers) -> np.ndarray:
    """Ordinal tier error: > 0 conservative (over-spend), < 0 aggressive."""
    return np.array([TIER_INDEX[p] - TIER_INDEX[t]
                     for p, t in zip(pred_tiers, true_tiers)])


# ---------------------------------------------------------- selection ------

def select_candidate(scores: pd.DataFrame, tie_tol: float) -> str:
    """scores: columns name, score (matched-recall cost ratio), feasible.
    Pre-registered rule: drop infeasible; best score; within tie_tol of the
    best -> simplest by SIMPLICITY order."""
    ok = scores[scores["feasible"]]
    if ok.empty:
        raise RuntimeError("no feasible candidate")
    best = ok["score"].max()
    near = ok[ok["score"] >= best - tie_tol]["name"].tolist()
    return sorted(near, key=lambda n: SIMPLICITY.index(n)
                  if n in SIMPLICITY else len(SIMPLICITY))[0]


# -------------------------------------------------------- statistics -------

def wilcoxon_paired(a, b) -> dict:
    """Paired Wilcoxon signed-rank on a - b, with matched-pairs rank-biserial
    correlation r = (W+ - W-) / (W+ + W-) over non-zero differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return {"n_nonzero": 0, "p_value": 1.0, "rank_biserial": 0.0}
    ranks = stats.rankdata(np.abs(nz))
    w_plus, w_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    res = stats.wilcoxon(nz, zero_method="wilcox")
    return {"n_nonzero": int(len(nz)), "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "rank_biserial": float((w_plus - w_minus) / (w_plus + w_minus))}


def bootstrap_ci(values, n: int, seed: int, stat=np.mean) -> tuple:
    """Percentile 95% CI of `stat` over resampled queries (seeded)."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    s = np.array([stat(v[i]) for i in idx])
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(lo), float(hi)


def to_json(obj) -> str:
    return json.dumps(obj, indent=2, default=lambda o: o.item()
                      if hasattr(o, "item") else str(o))
