# Phase 4b — Evaluation State Record

Status as of 2026-10-02: **the corrected confirmation set (v2) is constructed and
validated. NO evaluation has been performed on any query set.** The final
evaluation proceeds only after user approval of this set.

## Frozen routers (unchanged)
- `results/phase4b_train_311634ee3011_20261002T145649Z/`
- 1,142 / 1,142 manifest SHA-256 hashes re-verified after the v2 set was built.
- No router, model, τ, feature, candidate, B1, seed or contract parameter changed.

## Attempt 1 (v1) — REJECTED, never evaluated
- Definition: 2,000 rows drawn from **all** 100,000 `sift_learn` rows. Seed 20261005;
  Fisher–Yates via `ars::MakeQuerySplit`.
  - Config: `configs/phase4b/confirmation_set_v1_REJECTED.yaml`
  - Run: `results/572a53726e42_20261002T172041Z/`
  - Files archived in `splits/_invalid/` and `data/confirm/_invalid/`
- **Integrity failure** (`results/phase4b_confirmation_integrity.json`):
  - 190 exact duplicates of original SIFT queries: 150 from the TRAIN split, 40 from
    the old TEST split;
  - 1 internal duplicate pair (rows 402 and 532);
  - 0 duplicates against the base set.
- **Root cause:** `sift_query` is contained in `sift_learn`. 9,979 of 10,000 queries occur
  byte-for-byte in the learn file (`results/phase4b_confirmation_overlap_characterisation.json`).
  The redesign's claim that `sift_learn` is disjoint from the query set was unverified
  and false.
- **Not done on v1:** no ground truth, no oracle curves, no features, no evaluation. It
  was not repaired by deleting rows; it is kept only as a record
  (`results/phase4b_confirmation_STOP.md`).

## Attempt 2 (v2) — corrected Option A (user-approved definition)
**Population (fixed before drawing):**
1. Take the rows of `data/sift/sift_learn.fvecs` in ascending row order.
2. Exclude every row whose vector is a byte-exact copy of any of the 10,000 original
   SIFT queries (`data/sift/sift_query.fvecs`). This removes 10,017 rows.
3. Keep each distinct remaining vector once, at its lowest learn row. This removes 195
   rows (195 duplicate pairs).

This implements "no exact duplicates within the sample" at the population stage, so
nothing is replaced after drawing.

**Eligible population: 89,788** (eligible row-list FNV-1a64 `4e6db04fbdaf0fb6`).

**Sampling (fixed before drawing):**
- `ars::MakeQuerySplit(89788, 2000, seed)`: Fisher–Yates over **population positions**
  0..89,787 with `std::mt19937_64(seed)` and rejection-sampled bounded integers.
- The first 2,000 positions are selected.
- Output is in ascending `sift_learn` row order.
- **The seed is applied to the filtered population.**

**Seed: 20261007.** New; never used before (20261005 = rejected v1, 20261006 = B1 hash).

**Artefacts:**
- Config: `configs/phase4b/confirmation_set_v2.yaml`
- Run: `results/b0f0dabdfeea_20261002T172854Z/`
- Query-set id: `sift_learn_confirm_v2_seed20261007`

| File | SHA-256 |
|---|---|
| `data/confirm/sift_learn_confirm_v2_seed20261007_n2000.fvecs` | `69070a20fc5344f3f510c476b4b33c1867037489e044f352c03230fe4d78d91d` |
| `splits/confirmation_v2_sift_learn_seed20261007_n2000_ids.csv` (row, source_row, population_position) | `aa22a5f00d2e4989586503e1adb92e838c62f0d7faa31cf24ea3fe21d67a510f` |
| `splits/confirmation_v2_sift_learn_seed20261007_n2000_tagging.csv` (all "test") | `9c8504c622d69d9b00708d7427496637d6188e88a01d2faa9d052455396a386a` |

**Integrity** (`results/phase4b_confirmation_v2_integrity.json`): **PASS**

| Requirement | Result |
|---|---|
| Exactly 2,000 vectors | 2,000 |
| Exact duplicates vs the 1M base vectors | 0 |
| Exact duplicates vs the 10,000 original queries | 0 |
| Internal duplicate pairs | 0 |
| Deterministic regeneration | C++ re-run byte-identical (vectors, ids, tagging); independent Python re-implementation of population + Fisher–Yates gives the identical population (89,788) and ids; vectors equal the source rows |
| Population ≥ 2,000 | 89,788 |

**Label-free distribution-shift report (descriptive; not an acceptance criterion):**
- Vector-norm quantiles (5/25/50/75/95%): original queries 507.62 / 508.16 / 508.61 /
  509.09 / 509.74; v2 set 507.61 / 508.14 / 508.61 / 509.11 / 509.77. KS D = 0.0235,
  p = 0.31.
- Mean coordinate 26.71 vs 26.74. Zero-coordinate fraction 0.2261 vs 0.2257.
- Per-dimension mean absolute difference: 0.568 (v2 vs original queries), against
  0.682 between the original train and test splits (scale reference).

**Near-duplicate diagnostic (descriptive only; no threshold; nothing removed).**
Euclidean distance to the nearest original query, at the 1 / 5 / 25 / 50 / 75 / 95th
percentiles:

| From → to | 1% | 5% | 25% | 50% | 75% | 95% |
|---|---|---|---|---|---|---|
| v2 → nearest original query | 97.6 | 151.7 | 213.0 | 244.8 | 272.3 | 305.6 |
| v2 → nearest train query | 107.9 | 156.5 | 215.5 | 247.0 | 275.8 | 307.9 |
| Reference: old test → nearest train | 94.7 | 148.9 | 212.9 | 247.2 | 274.6 | 311.1 |
| Reference: train → nearest other train | 110.5 | 156.9 | 215.1 | 247.0 | 275.0 | 309.5 |

Minimum distance from any v2 vector to any original query: 28.27.

## Not yet done (awaiting approval)
- Ground truth, oracle curves and features for v2. Router, B1 and B2 evaluation.
- The old-test second look.
- For the evaluation stage, `configs/phase4b/evaluate.yaml` must point its
  `confirmation` entry to the v2 query-set id (`sift_learn_confirm_v2_seed20261007`)
  and to the v2 runs. The id feeds the frozen B1 hash and is part of the set's identity,
  not a frozen router parameter.

## Final evaluation attempt 1 — STOPPED at the artefact-consistency check (2026-10-02)
Pre-flight (10 checks) passed: v2 query-set id and hashes, n = 2,000, router directory,
manifest verified (1,142), ladder, S*/R levels, seeds, contract.
Confirmation v2 runs (identical to the frozen configs except query file, tagging and name):
- oracle: seed 42 `results/6cc721cd28dc_20261002T173819Z`, 43 `results/c4eb66527ca2_20261002T174027Z`,
  44 `results/dc50f5d75ebc_20261002T174241Z`;
- features: 42 `results/152d65619a3d_20261002T174455Z`, 43 `results/0424aedbaf85_20261002T174457Z`,
  44 `results/09a4fe4d7f3a_20261002T174459Z`.

Ground truth (`data/cache/gt_S0_b43ae8f836a05de9`) was checked independently: NumPy float64
brute force gives identical top-100 ids for all 2,000 queries (PASS). No query is censored
on the full grid.

The evaluation stopped on its first router at the assertion "fitted model (joblib)
probabilities == sklearn-free JSON export probabilities". No metric was computed.
Output quarantined in `results/_invalid/`.

Diagnosis (features only; no labels or outcomes read):
- 2 of 54 frozen specs disagree: seed 42 single-feature ablation, knn_dist,
  per_query 0.95 (DT3) and per_query 0.99 (DT2).
- 3 queries in total; max |ΔP| 0.12.
- Cause: sklearn trees cast X to float32 before comparing with split thresholds; the
  export re-implementation compares in float64. A float32 cast removes all
  mismatches (0 remaining).
- 0 routing decisions differ.
- No primary, LID or mean-recall router is affected.

Awaiting the user's decision before re-running.

## Final evaluation — COMPLETED (2026-10-02)
- Checker fix approved and applied (`proba_from_export`: trees round inputs to float32,
  as sklearn does). 0/54 mismatches; 0 routing decisions changed; 1,142 hashes unchanged.
- Primary confirmation evaluation: `results/phase4b_eval_confirmation_20261002T175418Z/`.
  Validated by 36 C++ re-search runs (72,000 rows matching). The determinism re-run is
  identical.
- Old-test second look (after primary): `results/phase4b_eval_old_test_second_look_20261002T175740Z/`.
- **Frozen all-three gate: FAIL.** S\* = 0.90 / 0.95 / 0.99: −20.9% / −8.9% / +1.1%
  saving vs B1. Report: `docs/phase4b_final_report.md`. Provenance:
  `results/phase4b_final_manifest.json`.
