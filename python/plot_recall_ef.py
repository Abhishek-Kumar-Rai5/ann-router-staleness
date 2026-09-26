"""Plots recall against efSearch for a Phase 1 sweep. results.csv stays the record;
the summary it writes is just for convenience.

Usage: python python/plot_recall_ef.py results/<experiment_id>
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ef")
    return pd.DataFrame({
        "mean_recall": g["recall"].mean(),
        "mean_recall_tie_aware": g["recall_tie_aware"].mean(),
        "p05_recall": g["recall"].quantile(0.05),
        "frac_queries_recall_lt_0.9": g["recall"].apply(lambda r: (r < 0.9).mean()),
        "mean_distance_computations": g["distance_computations"].mean(),
        "median_latency_us": g["latency_us"].median(),
        "qps_single_thread": 1e6 / g["latency_us"].mean(),
    }).reset_index()


def main() -> int:
    run_dir = Path(sys.argv[1])
    df = pd.read_csv(run_dir / "results.csv")
    s = summarise(df)
    s.to_csv(run_dir / "sweep_summary.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(s.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    k = int(df["k"].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(s["ef"], s["mean_recall"], marker="o", label="mean")
    ax.plot(s["ef"], s["p05_recall"], marker=".", ls="--", label="5th percentile")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("efSearch")
    ax.set_ylabel(f"recall@{k}")
    ax.set_title(f"Recall@{k} vs efSearch (S0)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(s["mean_distance_computations"], s["mean_recall"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("mean distance computations / query")
    ax.set_ylabel(f"recall@{k}")
    ax.set_title("Recall vs effort")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(s["mean_recall"], s["qps_single_thread"], marker="o")
    ax.set_yscale("log")
    ax.set_xlabel(f"recall@{k}")
    ax.set_ylabel("QPS (single thread)")
    ax.set_title("Throughput vs recall")
    ax.grid(alpha=0.3)

    fig.suptitle(run_dir.name, fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "recall_vs_ef.png", dpi=130)
    print(f"wrote {run_dir / 'recall_vs_ef.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
