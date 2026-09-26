"""Fixes the effort tiers and difficulty thresholds from the S0 training split,
then applies those frozen numbers to every query. Test rows are only touched
after the thresholds are fixed.

Usage: python python/define_effort_tiers.py --oracle <oracle_run> --out derived
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TIER_QUANTILES = {"low": 1 / 3, "med": 2 / 3, "high": 0.99}
TERTILE_QUANTILES = {"T1": 1 / 3, "T2": 2 / 3}


def grid_quantile(oracle_ef: pd.Series, grid: list, p: float) -> int:
    cdf = np.array([(oracle_ef <= e).mean() for e in grid])
    return int(grid[int(np.argmax(cdf >= p - 1e-12))])


def fit_thresholds(train: pd.DataFrame, grid: list) -> dict:
    assert set(train["split"]) == {"train"}, "thresholds must use train only"
    assert train["reached"].all(), "train split is expected to be uncensored"
    ef = train["oracle_ef"].astype(int)
    tiers = {t: grid_quantile(ef, grid, p) for t, p in TIER_QUANTILES.items()}
    tert = {t: grid_quantile(ef, grid, p) for t, p in TERTILE_QUANTILES.items()}
    assert tiers["low"] < tiers["med"] < tiers["high"], tiers
    return {"tiers": tiers, "tertiles": tert,
            "train_floor_share": float((ef == grid[0]).mean())}


def tier_label(oracle_ef, reached, tiers):
    if not reached:
        return "high"
    for name in ("low", "med", "high"):
        if oracle_ef <= tiers[name]:
            return name
    return "high"  # needs more than ef_high, so it fails even at the top tier


def stratum(oracle_ef, reached, tert):
    if not reached or oracle_ef > tert["T2"]:
        return "hard"
    return "easy" if oracle_ef <= tert["T1"] else "medium"


def assign(labels: pd.DataFrame, fit: dict) -> pd.DataFrame:
    out = labels[["query_id", "split", "reached", "oracle_ef"]].copy()
    out["tier_label"] = [tier_label(e, r, fit["tiers"])
                         for e, r in zip(out["oracle_ef"], out["reached"])]
    out["exceeds_high_tier"] = [(not r) or e > fit["tiers"]["high"]
                                for e, r in zip(out["oracle_ef"], out["reached"])]
    out["difficulty"] = [stratum(e, r, fit["tertiles"])
                         for e, r in zip(out["oracle_ef"], out["reached"])]
    out["censored"] = out["reached"] == 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.oracle)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "metadata.json").read_text())
    grid = meta["oracle"]["ef_grid"]
    labels = pd.read_csv(run / "oracle_labels.csv")

    train = labels[labels["split"] == "train"].copy()
    fit = fit_thresholds(train, grid)
    train_bytes = train[["query_id", "oracle_ef"]].to_csv(index=False).encode()

    curves = pd.read_csv(run / "oracle_curves.csv")
    ctr = curves[curves["split"] == "train"]
    baselines_train = {}
    for name, ef in fit["tiers"].items():
        c = ctr[ctr["ef"] == ef]
        baselines_train[name] = {
            "ef": ef,
            "mean_recall_tie_aware": float(c["recall_tie_aware"].mean()),
            "frac_queries_below_target": float(
                (c["recall_tie_aware"] < 0.95 - 1e-9).mean()),
            "mean_distance_computations": float(
                c["distance_computations"].mean())}

    a = assign(labels, fit)
    counts = {
        s: {"tier_label": a[a["split"] == s]["tier_label"].value_counts()
            .reindex(["low", "med", "high"], fill_value=0).to_dict(),
            "difficulty": a[a["split"] == s]["difficulty"].value_counts()
            .reindex(["easy", "medium", "hard"], fill_value=0).to_dict(),
            "exceeds_high_tier": int(a[a["split"] == s]["exceeds_high_tier"].sum()),
            "censored": int(a[a["split"] == s]["censored"].sum())}
        for s in ("train", "test")}

    result = {
        "source_oracle_run": run.name,
        "fitted_on": "train split only",
        "train_rows_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "rules": {"tier_quantiles": TIER_QUANTILES,
                  "tertile_quantiles": TERTILE_QUANTILES,
                  "quantile_definition": "smallest grid ef with train CDF >= p",
                  "ties": "<= threshold goes to the lower tier/group",
                  "censored": "tier 'high' (fails), difficulty 'hard'"},
        "fixed_ef_baselines": fit["tiers"],
        "tertile_thresholds": fit["tertiles"],
        "train_floor_share_at_ef10": fit["train_floor_share"],
        "baselines_on_train_split": baselines_train,
        "counts": counts,
        "censored_query_ids": a.loc[a["censored"], "query_id"].tolist(),
    }
    (out / "sift1m_s0_effort_tiers.json").write_text(json.dumps(result, indent=2) + "\n")
    a.to_csv(out / "sift1m_s0_query_strata.csv", index=False)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
