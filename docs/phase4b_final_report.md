# Phase 4b — Final Report

Date: 2026-10-02. Phase 4b is complete. **Phase 5 has not been started.**

Every number below comes from the frozen artefacts listed in §16. Full tables:
- `results/phase4b_eval_confirmation_20261002T175418Z/tables.md` (primary)
- `results/phase4b_eval_old_test_second_look_20261002T175740Z/tables.md` (secondary)

Notation:
- **dc** = distance computations per query (the cost metric).
- **Success** = tie-aware recall@10 == 1.0 for that query.
- **B1** = the frozen, training-calibrated two-ef fixed baseline.
- **B2** = the hindsight-optimal fixed baseline on the evaluation set.
- Savings are relative to B1; a negative saving means the router costs more.

## 1. Frozen experimental setup
- **Data and index:** SIFT1M with 1M base vectors. HNSW via hnswlib v0.8.0, M = 16,
  ef_construction = 200, single-threaded builds. Index seeds 42, 43 and 44.
- **Primary quality contract:** per-query success = tie-aware recall@10 == 1.0.
  Co-primary success-rate targets S\* ∈ {0.90, 0.95, 0.99}. Phase 4b passes **only if
  the gate passes at all three**.
- **Secondary contract:** mean recall@10 with R ∈ {0.95, 0.97, 0.99}, using the
  identical procedure. R = 0.95 carries an ex-ante candidate-set limitation: even
  perfect information could save at most about 1–1.5% there.
- **Candidate efforts:** {probe result (ef 10), 20, 40, 83, 164, 327, 647, 1343, 2661}.
  Fallback: 2661.
- **Router:**
  - Per candidate, a model of P(query succeeds at that effort | features), using
    logistic regression or a small tree (depth 2–4).
  - Probabilities are made non-decreasing across efforts (cumulative maximum).
  - Decision: the cheapest effort whose probability ≥ τ. τ was calibrated on training
    out-of-fold predictions to reach S\*.
  - Model family chosen by out-of-fold cost on training data.
- **Features:** knn_dist, centroid_dist and score_concentration from an ef = 10 probe.
  The LID ablation uses only an ef = 20 probe. The single-feature ablations are also
  frozen.
- **Cost:** probe + routed search, fully additive. Returning the probe's own result
  costs only the probe.
- **B1:** a two-ef mix calibrated on training data to S\* (or R), with a deterministic
  splitmix64 hash assignment per query (seed 20261006).
- **B2:** hindsight-optimal two-ef fixed policy on the evaluation set, at the router's
  realised quality. It favours the baseline by design.
- **Gate, per level (frozen, bias audit §10):**
  1. saving vs B1 ≥ 10% **and** 95% CI lower bound > 0 **and** Wilcoxon p < 0.01;
  2. 95% CI lower bound of the quality difference (router − B1) ≥ −0.01;
  3. saving vs B2: 95% CI lower bound > 0;
  4. every seed: saving vs B1 > 0 and quality difference ≥ −0.01.
- **Statistics:**
  - query-clustered bootstrap (2,000 resamples, seed 20261002; all seeds of a query
    resampled together);
  - Wilcoxon signed-rank on per-query seed-averaged paired cost differences, with
    rank-biserial effect size;
  - exact McNemar test per seed.

## 2. Confirmation-set provenance
- **v1 (rejected, never evaluated):** `sift_learn`, seed 20261005. It held 190 exact
  copies of original queries (150 from the training split) and 1 internal duplicate
  pair. Cause: the original query file is contained in `sift_learn`. Archived in
  `splits/_invalid/`.
- **v2 (used):** `sift_learn` (100,000 rows) minus 10,017 exact copies of the
  original 10,000 queries, giving 89,983 rows. Deduplicated (195 pairs), giving an
  eligible population of 89,788. Fisher–Yates via `mt19937_64` over population
  positions, **seed 20261007**, n = **2,000**. Query-set id
  `sift_learn_confirm_v2_seed20261007`.
- **Hashes:** vectors SHA-256 `69070a20…d91d`; ids SHA-256 `aa22a5f0…510f`.
- **Integrity:**
  - 0 duplicates vs base, 0 vs the original queries, 0 internal;
  - C++ regeneration byte-identical; independent Python regeneration identical.
- **Shift (label-free):** vector norms indistinguishable (KS p = 0.31). Per-dimension
  mean difference 0.57, against 0.68 between the original train and test splits.
- **Ground truth:** independent NumPy brute force gives identical top-100 ids for all
  2,000 queries. No query is censored on the full ef grid, for any seed.

## 3. Router provenance
- `results/phase4b_train_311634ee3011_20261002T145649Z/`: 54 routers, 540 models,
  1,142 hashed files.
- Re-verified before and after every step; unchanged.
- Trained on the 8,000 training queries only.
- The selected primary family is logistic regression everywhere except seed 44 at
  mean-recall R = 0.95 (DT2).
- **Before evaluation:** the export-consistency checker was fixed to round tree inputs
  to float32, as sklearn does (approved). Effect: 0/54 mismatches, max |ΔP| = 0,
  0 routing decisions changed.

## 4–5. Primary results (confirmation v2; pooled over 3 seeds × 2,000 queries)

| S\* | Router cost mean / median | B1 cost mean / median | Abs. saving | Saving vs B1 [95% CI] | Router / B1 success | Δsuccess [95% CI] | Saving vs B2 [95% CI] | Wilcoxon p (rank-biserial) | Gate |
|---|---|---|---|---|---|---|---|---|---|
| 0.90 | 2,379 / 2,035 | 1,968 / 2,044 | −411 | **−20.9%** [−23.3, −18.5] | 0.9102 / 0.9135 | −0.0033 [−0.0162, +0.0092] | −22.1% [−28.3, −14.6] | 2.1e-16 (+0.212) | **FAIL** (✗ ✗ ✗ ✗) |
| 0.95 | 2,886 / 2,670 | 2,651 / 2,767 | −235 | **−8.9%** [−11.1, −6.8] | 0.9478 / 0.9537 | −0.0058 [−0.0157, +0.0043] | −16.7% [−27.1, −5.4] | 7.1e-05 (−0.103) | **FAIL** (✗ ✗ ✗ ✗) |
| 0.99 | 4,436 / 3,802 | 4,485 / 4,692 | +48 | **+1.1%** [−0.7, +2.9] | 0.9915 / 0.9892 | +0.0023 [−0.0017, +0.0067] | +6.2% [−6.8, +19.7] | 1.0e-25 (−0.271) | **FAIL** (✗ ✓ ✗ ✓) |

- The rank-biserial sign is for router − B1 cost; positive means the router is
  costlier per query.
- At 0.95 and 0.99 the router is cheaper in the **median** but costlier in the
  **mean** (or level). It saves on most queries and overspends heavily on a minority
  (see §10).
- **Cost split:**
  - probe: 393 dc on average;
  - router search only: 1,985 / 2,493 / 4,043 at S\* = 0.90 / 0.95 / 0.99;
  - router total p90 / p99: 3,761 / 6,651, 5,830 / 6,900 and 6,777 / 11,499 at
    S\* = 0.90 / 0.95 / 0.99;
  - B1 total p99: 2,618 / 3,576 / 6,205 at the same levels.
- **Fallback** (no candidate reaches τ): 0% of queries.
- **Quality:** median recall@10 is 1.00 for every policy. At S\* = 0.95 the router's
  failures are 278 at R = 0.9, 21 at R = 0.8 and 14 at R ≤ 0.7; B1's are 230 / 38 / 10.

## 6. Ablations (confirmation)

| Variant | S\* = 0.90 | S\* = 0.95 | S\* = 0.99 | Gate |
|---|---|---|---|---|
| LID (ef = 20 probe only) | −24.8% | −12.5% | +0.2% | FAIL at all three |
| Single feature | −24.5% | −11.6% | −1.7% | FAIL at all three |

**Mean-recall family** (primary router): −43.5% (R = 0.95, the documented
limitation), −31.6% (0.97), −15.3% (0.99). The quality criterion is met at every R;
the cost criteria fail. The LID and single-feature variants show the same pattern.

## 7. Baselines
- **B1 mixes, identical across seeds:**
  - S\* = 0.90: ef 111/117;
  - S\* = 0.95: ef 164/172;
  - S\* = 0.99: ef 311/327;
  - with w_hi per seed in the router specs.
- **B1 realised success on confirmation:** 0.9135 / 0.9537 / 0.9892. That is close to
  S\*, so there is no calibration-shift flag.
- **Pure smallest-ef B1 (secondary):** cost 2,005 / 2,730 / 4,584. Router saving
  against it: −18.6% / −5.7% / +3.2%.
- **B2 (hindsight):** pooled cost 1,948 / 2,473 / 4,728 at S\* = 0.90 / 0.95 / 0.99.
  Per seed (42 / 43 / 44): [1937, 1982, 1927], [2488, 2467, 2463], [4726, 4731, 4727]. The router does not beat B2 at any
  level (criterion 3).

## 8. Oracle references (confirmation; kept separate)

| Seed | 1. Grid min-ef oracle: censored / mean search dc | 2. Cheapest succeeding candidate: achievable / mean total dc | 3. Perfect-knowledge policy at S\* = 0.90 / 0.95 / 0.99 (dc) |
|---|---|---|---|
| 42 | 0 / 1,109 | 2,000 / 1,737 | 1,234 / 1,397 / 1,617 |
| 43 | 0 / 1,119 | 2,000 / 1,749 | 1,247 / 1,410 / 1,629 |
| 44 | 0 / 1,113 | 2,000 / 1,750 | 1,247 / 1,411 / 1,630 |

**Perfect information over the same candidates, with the probe charged, would cost
about 47% less than B1 at S\* = 0.95 (1,397 vs 2,651).** The headroom exists; the
router does not capture it.

## 9. Statistics and regret decomposition (primary router; mean over seeds)

| S\* | Avoidable failures | Overspend on successful queries (dc) | Underspend on failures (dc, not netted) | Probe overhead (dc) | Contract regret (dc) | Legacy regret (dc) |
|---|---|---|---|---|---|---|
| 0.90 | 0.0898 | 905 | 2,124 | 393 | 1,104 | 1,265 |
| 0.95 | 0.0522 | 1,332 | 2,333 | 393 | 1,488 | 1,772 |
| 0.99 | 0.0085 | 2,756 | 4,887 | 393 | 2,797 | 3,323 |

- **Sensitivities** (non-gating), saving vs B1 at S\* = 0.90 / 0.95 / 0.99:
  - shared descent: −14.8% / −4.4% / +3.8%;
  - probe-free: −0.9% / +6.0% / +9.8%.

  Even with a free probe, no level reaches the +10% requirement.
- **Spearman ρ, features vs grid-oracle ef** (descriptive, confirmation set; per
  seed): knn_dist +0.48, centroid_dist −0.42, score_concentration +0.41 to +0.44,
  LID +0.60. Same as on the training data (Phase 3).
- **McNemar** (success, per seed): no significant router-vs-B1 difference at any
  per-query level (p 0.28–1.0).
- **Censored or excluded cases:**
  - Confirmation: none. Every query reaches the target on the full grid for every
    seed, so no exclusions or censoring.
  - Old test: 2 censored queries on seed 42 only (ids 4659, 8318); 0 on seeds 43/44.
    Handled by the pre-specified rule: they count as failures for every policy and
    are left out of the grid-oracle cost.
  - One-class constant models: 0 in the frozen final models.
  - NaN / Inf / duplicate / missing rows: 0.

## 10. Difficulty strata (frozen easy ≤ 18 < medium ≤ 48 < hard; primary router; query × seed)

| S\* | Easy: router / B1 cost (success) | Medium | Hard |
|---|---|---|---|
| 0.90 | 1,512 / 1,710 (0.997 / 1.000) | 2,349 / 2,002 (0.964 / 1.000) | 3,323 / 2,203 (0.762 / 0.730) |
| 0.95 | 1,804 / 2,284 (0.999 / 1.000) | 2,886 / 2,695 (0.985 / 1.000) | 4,025 / 2,989 (0.854 / 0.855) |
| 0.99 | 2,746 / 3,818 (1.000 / 1.000) | 4,433 / 4,556 (1.000 / 1.000) | 6,219 / 5,110 (0.974 / 0.966) |

**Failure mode:**
- On easy queries the router saves 11–28%.
- On hard queries it spends 34–51% more, with little or no gain in success (0.854 vs
  0.855 at S\* = 0.95).
- Medium queries both cost more and occasionally fail where B1 succeeds.

## 11. Three-seed consistency

| S\* | Seed 42 | Seed 43 | Seed 44 | Pattern |
|---|---|---|---|---|
| 0.90 | −20.3% | −22.0% | −20.3% | Consistent negative |
| 0.95 | −9.0% | −9.4% | −8.2% | Consistent negative |
| 0.99 | +0.4% | +0.7% | +2.2% | Consistent small positive, below threshold; CI spans 0 |

**No seed-specific behaviour.** The mean-recall family is consistently negative on all
seeds.

## 12. Old-test second look (SECONDARY / DISCLOSED SECOND-LOOK ANALYSIS)
- This split was seen once before (original Phase 4). It was evaluated **after** the
  primary results were frozen and is not combined with them.
- **Primary router saving vs B1:**

  | S\* | Saving [95% CI] | Gate |
  |---|---|---|
  | 0.90 | −19.3% [−21.6, −17.0] | FAIL |
  | 0.95 | −9.3% [−11.5, −7.2] | FAIL |
  | 0.99 | +0.8% [−1.1, +2.6] | FAIL |

- **Mean-recall family:** −42.3% / −31.9% / −14.7%.
- Same direction and size as the confirmation set; same easy-save, hard-overspend
  pattern.

## 13. Gate-by-gate result

| S\* | 1. Saving ≥ 10%, CI > 0, p < 0.01 | 2. Δsuccess CI lb ≥ −0.01 | 3. vs B2 CI lb > 0 | 4. Every seed | Gate |
|---|---|---|---|---|---|
| 0.90 | ✗ (−20.9%) | ✗ (lb −0.0162) | ✗ | ✗ (all seeds negative) | **FAIL** |
| 0.95 | ✗ (−8.9%) | ✗ (lb −0.0157) | ✗ | ✗ (all seeds negative) | **FAIL** |
| 0.99 | ✗ (+1.1%, CI spans 0) | ✓ (lb −0.0017) | ✗ | ✓ (all seeds > 0) | **FAIL** |

- **A. Frozen decision (all three co-primary): Phase 4b gate = FAIL.**
- **B.** Per level: FAIL / FAIL / FAIL.
- **C. Informational S\* = 0.95-only reading (not the decision rule):** also FAIL. It
  costs 8.9% more than B1, CI entirely below 0.

## 14. Limitations and caveats
- **Single dataset and scale:** SIFT1M (128-d), k = 10, one HNSW configuration.
- **Cost measure:** distance computations, not latency. The additive probe is a
  conservative accounting; the probe-free bound is reported.
- **Confirmation set:** drawn from `sift_learn`, a different file than the training
  queries. It matches them closely in distribution, and the old-test second look
  agrees.
- **One fixed router design:** three cheap features, per-candidate LR/tree success
  models, a single global τ, and a fixed doubling ladder. Other designs were not
  tested, by contract.
- **Process deviations before the final run, all disclosed:**
  - rejected confirmation v1;
  - checker fix (no decision changed);
  - quarantined aborted runs.

## 15. Final research interpretation
**What the experiment directly demonstrates.** On SIFT1M with frozen routers that use
three cheap probe features, allocating search effort per query **does not** reach a
per-query success target with less computation than a fixed effort.
- At S\* = 0.90 and 0.95 it needs **9–21% more** distance computations.
- At 0.99 it is at **parity** (+1.1%, CI spans 0).
- Under mean-recall targets it needs **15–44% more**.
- The result is consistent across three index seeds, two query sets, and the LID and
  single-feature ablations.

**What it does not demonstrate.**
- That query-adaptive effort allocation cannot work: perfect information over the
  same candidates would cost about 47% less than B1.
- That other features, models, decision rules, datasets or index configurations
  would fail.
- Anything about latency.

**Support for the Phase 4b research question.** **Not supported** for this frozen
configuration. The design's prerequisite (§21: "router clearly outperforms the best
fixed-ef baseline on S0 — must succeed before proceeding") is **not met**.

**What the failure modes imply for the next phase.**
- **The probe matters, but isn't the whole story.** Its 393 dc is about 15% of B1's
  cost at S\* = 0.95. Even probe-free, savings stay below 10%.
- **The core failure is discrimination among harder queries.** To reach S\*, the
  calibrated threshold routes many medium and hard queries to expensive efforts that
  rarely change their outcome. Overspend on successes is 1,332 dc at 0.95, while
  hard-query success is unchanged versus B1.
- **Feature signal is real but weak** (|ρ| ≈ 0.4–0.6), and it transferred unchanged
  to new queries.
- **The design's stated precondition for studying staleness is not met.** Phase 5,
  which asks whether an S0 routing advantage degrades, has no S0 advantage to degrade
  with this router. Whether and how to continue is a research decision for the user;
  nothing was started.

## 16. Artefact locations
- **Frozen routers:** `results/phase4b_train_311634ee3011_20261002T145649Z/` (manifest
  SHA-256 in the final manifest).
- **Confirmation v2:**
  - set: `data/confirm/sift_learn_confirm_v2_seed20261007_n2000.fvecs`,
    `splits/confirmation_v2_*`, `configs/phase4b/confirmation_set_v2.yaml`;
  - creation run: `results/b0f0dabdfeea_20261002T172854Z/`;
  - integrity: `results/phase4b_confirmation_v2_integrity.json`.
- **Confirmation oracle runs:** `results/6cc721cd28dc_20261002T173819Z`,
  `results/c4eb66527ca2_20261002T174027Z`, `results/dc50f5d75ebc_20261002T174241Z`.
- **Confirmation feature runs:** `results/152d65619a3d_20261002T174455Z`,
  `results/0424aedbaf85_20261002T174457Z`, `results/09a4fe4d7f3a_20261002T174459Z`.
- **Ground-truth check:** `results/phase4b_confirm_v2_logs/gt_verification.json`.
- **Primary evaluation:** `results/phase4b_eval_confirmation_20261002T175418Z/`
  (`eval_report.json`, `per_query_rows.csv`, `tables.md`, `validation/` with 36 C++
  re-search runs, all matching).
- **Determinism re-run** (identical report and per-query table):
  `results/_determinism_checks/`.
- **Old-test second look:** `results/phase4b_eval_old_test_second_look_20261002T175740Z/`.
- **Provenance:** `results/phase4b_final_manifest.json` (185 file hashes, configs,
  seeds, versions, commands, timing, git state).
- **History:**
  - `docs/phase4b_evaluation_state.md`;
  - `results/phase4b_confirmation_STOP.md`;
  - `results/_invalid/README.txt`;
  - `splits/_invalid/README.md`.

**No router, model, threshold, feature, candidate, baseline, oracle, difficulty
threshold or statistical procedure was changed after evaluation began.**
