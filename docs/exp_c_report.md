# Experiment C (C1) — DARTH-style runtime actuator at S0: result report

**Outcome: C1 FAILS.** All correctness, provenance and determinism checks pass. The frozen
DARTH policy uses **11.0% more** distance computations than the frozen B1 baseline (95% CI
8.9–13.0% more), at higher mean recall (0.963 vs 0.948).

This is a statement about this one frozen DARTH-style policy, in this configuration (SIFT1M,
L2, hnswlib M16 / efC200, k = 10, mean Recall@10 = 0.95). It is **not** evidence that adaptive
ANN search fails in general.

## 1. What DARTH is
DARTH (Chatzakis, Papakonstantinou, Palpanas, *DARTH: Declarative Recall Through Early
Termination for Approximate Nearest Neighbor Search*, PACMMOD 3(4) art. 242, SIGMOD 2026;
arXiv:2505.19001; official FAISS-based code: https://github.com/MChatzakis/DARTH) is
declarative-recall early termination:
- inside the HNSW base-layer search, a LightGBM regressor periodically predicts the
  query's current recall from 11 live search features;
- the search stops once the prediction reaches the recall target;
- predictor calls are spaced adaptively, more often as the prediction nears the target.

## 2. What we independently re-implemented
- **No DARTH or FAISS source is copied** (THIRD_PARTY.md).
- **Search:** a stop condition on hnswlib v0.8.0's public `searchStopConditionClosest` API;
  hnswlib is unmodified (`include/ars/darth.h`, `src/darth.cpp`,
  `HnswIndex::SearchWithStopCondition`).
- **Pipeline:** the driver `apps/exp_c_darth.cpp` (verify / trace / eval / predict_check),
  plus `python/exp_c_train.py` and `python/exp_c_eval.py`.
- **Mechanism, as published and frozen in design_doc.md C1.9:**
  - 11 features;
  - one observation per base-layer distance;
  - LightGBM regressor (100 trees, defaults, seed 42) on all rows;
  - recall@10-by-id labels;
  - adaptive interval pi = mpi + (ipi − mpi)(Rt − Rp);
  - published heuristic ipi = dists_Rt/2 = 367, mpi = dists_Rt/10 = 73;
  - stop when Rp ≥ 0.95;
  - search cap ef 142 (the paper's "plain search reaches recall ≥ 0.99" rule, on the
    training split).

## 3. Differences from the published implementation, and why
1. **Feature order (deliberate).** The reference runtime passes the features in an order
   that differs from the model's training order (avg ↔ furthest swapped; median ↔ p25
   swapped). We use the training order, so the intended DARTH policy is evaluated rather
   than an apparent defect.
   - Our C++ predictions equal Python LightGBM's to 2.2e-16.
   - The misaligned order would shift predictions by 0.017 on average, so this choice is
     not inert.
2. **Search loop.** hnswlib replaces FAISS:
   - hnswlib's own beam termination at ef 142 replaces FAISS's efSearch-step rule;
   - after a stop decision, hnswlib finishes the current node's neighbour distances, which
     are counted in the cost (mean 8.6 extra base-layer distances per stopped query).
3. **Training queries.** We use the frozen 8,000-query Phase-2 training split, not a
   benchmark learning set (sift_learn is consumed by A1 and confirmation v2).
4. **Index and k.** M16 / efC200 / k = 10. The paper's SIFT runs are SIFT100M / M32 /
   efC500 / efS500, so no published number is comparable, and no reproduction gate applies.
5. **Training reproducibility.** Multi-threaded LightGBM is not bit-reproducible. An
   identical refit differs by ≤ 2.2e-16 in predictions, with 0 changed stop decisions. The
   first model is the frozen one.

## 4. Configuration (frozen; design_doc.md C1, C1.9)
| Item | Value |
|---|---|
| Data and index | SIFT1M, L2; S0 seed-42 index `data/cache/hnsw_S0_c86ffb2a63e7325b.bin`, sha256 4282e2e2279fb17c… |
| Training | 8,000 Phase-2 train queries; 17,726,998 observations |
| Evaluation | 2,000 Phase-2 test queries, once |
| Target | mean Recall@10 = 0.95 (quality measured tie-aware; id recall also reported) |
| Baseline | Phase-4b mean-recall B1 at R = 0.95, seed 42: ef 50/53, w_hi 0.538 (read-only) |
| Model | `derived/exp_c/darth_lgbm_s0.txt`, sha256 d291a2f1f8cbd64f… |
| Policy | `derived/exp_c/darth_policy_s0.json`, sha256 bcbe2720044b51ed… |

## 5. Results (2,000 test queries)
| | DARTH | B1 |
|---|---|---|
| Mean recall@10, tie-aware | **0.9633** | 0.9480 |
| Mean recall@10, by id | 0.9625 | – |
| Mean distance computations / query | **1,172** | 1,057 |
| Median | 1,040 | 1,090 |
| p10 / p90 / p99 | 500 / 2,020 / 2,913 | 766 / 1,275 / 1,423 |
| Min / max | 432 / 3,386 | 314 / 1,664 |

- **DARTH's per-query recall:** 1,465 queries at 1.0, 388 at 0.9, 108 at 0.8, 29 at 0.7,
  6 at 0.6 and 4 at 0.5; 7.4% of queries are below 0.9.
- **95.1% of queries terminated early;** the rest ran to the ef-142 cap (cap mean cost
  2,324, recall 0.993).
- **Paired cost difference (B1 − DARTH):**
  - median +35, so DARTH is cheaper on 53% of queries;
  - mean −116, because the costly tail dominates: p5 −1,288, p1 −1,647.

DARTH is cheaper on easy queries but spends far more on the hard tail. It overshoots the
recall target (0.963 vs the 0.95 asked for), which is how the reference method behaves by
design (each query stops at a predicted ≥ 0.95).

## 6. C1 gate (C1.5; unchanged)
| Criterion | Result | Pass |
|---|---|---|
| G1 saving vs B1 ≥ 10% | **−11.0%** | ✗ |
| G2 cost-difference CI lower bound > 0 | mean B1 − DARTH = −115.8, 95% CI [−137.9, −94.0] (saving CI [−13.0%, −8.9%]) | ✗ |
| G3 Wilcoxon p < 0.01 in favour of DARTH | p = 8.5e-5, but in the opposite direction (rank-biserial −0.10) | ✗ |
| G4 recall non-inferiority (CI lb ≥ −0.01) | +0.0152, 95% CI [+0.0119, +0.0187] | ✓ |
| Mechanism / correctness / plausibility checks | all 16 pass | ✓ |

**C1: FAIL.**

## 7. Predictor and runtime overhead (separate from the cost metric)
- **Predictor use:** 8.0 calls per query on average (median 7, max 30), each with an
  inference. Mean predictor time 0.094 ms per query, 21% of DARTH's wall time.
- **Wall-clock** (single thread, sequential, warm cache): DARTH 0.436 ms per query vs
  0.074 ms for plain hnswlib at the B1-assigned ef, so 5.9×.
- **The non-predictor part is mostly our implementation path,** not DARTH:
  - the general hnswlib stop-condition loop with a virtual callback per distance, versus
    the bare-bone `searchKnn` fast path;
  - per-observation feature building.

  Wall time is therefore reported, not interpreted as DARTH's intrinsic overhead.

## 8. Checks (all pass)
- **Correctness:** with prediction disabled, the search matches `searchKnn` on all 10,000
  queries at ef 10/50/53/142: 0 mismatches in ids, order, distances and distance counts.
  - Two earlier verify runs failed from tie-handling in the new wrapper's
    post-processing: the same ids, distances and counts, but a different order or choice
    among equidistant duplicates.
  - They were fixed by mirroring `searchKnn`'s selection, then re-verified (docs/notes.md).
  - The same 0 mismatches hold after `clang-format`.
- **Unit tests:** gtests 68/68, including plain-mode equality with lazily deleted vectors;
  pytest 33/33.
- **Data separation:** no train/test overlap; the eval set is exactly the test split; the
  trace contains only training queries.
- **Determinism:** a repeat evaluation run is byte-identical apart from timing columns.
- **Hashes:** policy, model and index hashes are unchanged.
- **Cost accounting:** B1 per-query costs and recalls equal the Phase-2 oracle curves, and
  live plain searches reproduce the B1 costs exactly.
- **Frozen artifacts:** no Phase-2 or Phase-4b artifact was modified (none newer than
  2026-10-04).

## 9. Limitations
- One configuration, one index seed, one dataset; 2,000 test queries.
- The policy is a re-implementation on hnswlib. FAISS-specific behaviour (search-loop
  termination, the k-sized result heap) is approximated by hnswlib's semantics.
- The training data are the Phase-2 queries rather than a benchmark learning set, and fewer
  of them (8,000 vs the paper's 10,000).
- The interval parameters come from the published heuristic, with no tuning (as frozen). A
  different recall target or B1 contract was not explored, and must not be to rescue C1.

## 10. Status
- C1 failed, so the freeze rule for Experiment E (C1.6) is **not** triggered: this actuator
  is not established as a credible S0 runtime actuator.
- **No Experiment E work has been started.**

## Artifacts
| Artifact | Location |
|---|---|
| Correctness (final) | results/9cb179548eb1_20261004T202453Z; post-format re-check results/9cb179548eb1_20261004T211435Z |
| Trace run | results/02a1c5c22faa_20261004T202634Z; data/exp_c/trace_train.f64 (gitignored, 1.84 GB) |
| Training | results/exp_c_train_20261004T202826Z |
| C++/Python prediction check | results/2aaf3176b512_20261004T211200Z |
| Evaluation | results/b089d7599961_20261004T211319Z; determinism repeat results/b3b930d34d46_20261004T211322Z; post-format re-run results/82bf1f070903_20261004T211444Z |
| Analysis | results/exp_c_eval_20261004T211328Z (c1_report.json, per_query_darth_vs_b1.csv) |
