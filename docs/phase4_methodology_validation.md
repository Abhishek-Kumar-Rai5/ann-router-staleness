# Phase 4 Methodology Validation (second-order audit)

Date: 2026-10-02. This document validates `docs/methodology_audit.md` and
closes the five open questions. It is read-only throughout:
- nothing canonical changed: no router change, no retraining, no design-doc
  change, and the test-evaluation protocol is untouched;
- every new number uses the **training split only** (8,000 queries); scripts
  assert that no test row is read;
- test-split figures are neither recomputed nor used for any choice.

Sources of new numbers:
- `python/validate_audit_train.py` →
  `results/audit_validation_20261002T122651Z/validation_train.json`
- the per-query-contract sensitivity and descent measurement (same directory)
  → `per_query_contract_and_descent.json`
- earlier audit diagnostics: `results/audit_phase4_20261002T105613Z/audit_train.json`

Notation:
- **R** = mean tie-aware recall@10.
- **"Fails"** = per-query tie-aware recall@10 below 0.95, which means not 10/10.
- **dc** = distance computations per query.
- **R_op = 0.951575** is the DT2 out-of-fold (OOF) mean recall on train, the
  operating point every row of the audit table is matched to.

---

## 1. Validation objective

Check whether the audit's central claims hold, and on exactly which
definitions. The central claims are:
- (C1) at the router's operating point, even a perfect-knowledge router
  restricted to tiers 18/48/327 *plus the probe* costs more than fixed ef;
- (C2) the trained DT2 adds the largest loss.

The five questions:
1. seed and variance;
2. the recall contract behind each number;
3. probe-cost accounting;
4. where the fixed-ef baseline number comes from;
5. an independent distance-count check.

Then (6) rebuild the five-row audit table with full provenance.

## 2. Seed/variance findings

**Existing replicates.**

| Source of randomness | Values with completed runs | Design requirement |
|---|---|---|
| HNSW level seed (index construction) | 42 only, one S0 index | §13: "core conditions are repeated across 2–3 seeds" |
| Query split | 20261001 only | Fixed by design (never resampled) |
| CV fold assignment | 20261002 only | Not specified |

**UNRESOLVED.** There are no independent index-seed replicates, so
index-construction variance of every Phase 4 quantity is unknown. Measuring it
needs new index builds, about 554 s each, plus an oracle recomputation of about
13 minutes per seed. That is a new protocol step; it was not done.

**What can be estimated from existing data, without a new protocol.**

(a) **Query-sampling variance.** A bootstrap over the 8,000 training queries,
200 resamples, seed 20261004. The operating point is re-matched in every
resample: each row is evaluated at that resample's DT2 mean recall.

| Quantity | Mean | SD | 95% CI |
|---|---|---|---|
| Fixed ef (interpolated) | 1,078 | 15.9 | [1,051, 1,109] |
| Perfect knowledge, any ef | 669 | 5.5 | [658, 680] |
| Perfect knowledge, tiers 18/48/327 | 890 | 14.4 | [863, 920] |
| … + probe | 1,281 | 14.7 | [1,254, 1,310] |
| DT2 OOF (probe included) | 2,580 | 23.9 | [2,530, 2,625] |
| **Gap: fixed − (tiers + probe)** | **−203** | 5.2 | **[−212, −192]** |
| Gap: DT2 − fixed | +1,502 | 19.2 | [1,468, 1,538] |
| Gap: fixed − perfect any-ef | +408 | 11.4 | [388, 432] |

The key structural gap, fixed versus perfect-knowledge tiers plus probe, is
negative in every resample, with a CI about 40 standard errors from zero. It is
**not** a query-sampling artefact. **CONFIRMED** (for query-sampling variance only).

(b) **CV-fold spread of the DT2 result.** The five folds were reconstructed
from the same `StratifiedKFold` seed; no fitting.

| Fold | n | DT2 mean recall | DT2 cost | Fixed cost at that recall | Fixed / DT2 |
|---|---|---|---|---|---|
| 0 | 1,600 | 0.9521 | 2,491 | 1,062 | 0.426 |
| 1 | 1,600 | 0.9389 | 2,355 | 961 | 0.408 |
| 2 | 1,600 | 0.9508 | 2,475 | 1,073 | 0.434 |
| 3 | 1,600 | 0.9598 | 2,821 | 1,170 | 0.415 |
| 4 | 1,600 | 0.9563 | 2,773 | 1,181 | 0.426 |

The ratio ranges only 0.408–0.434 across folds. **CONFIRMED stable**
(within-seed). CV-seed variance itself is untested (one CV seed).

## 3. Recall-contract findings

Two different recall contracts appear in the project:
- **Per-query contract (Phase 2 oracle label):** a query is "solved" if its
  tie-aware recall@10 ≥ 0.95, which with k = 10 means **10/10**. The oracle ef
  is the smallest grid ef from which 10/10 holds at every larger grid ef.
- **Mean-recall contract (Phase 4 matched comparison):** policies are compared
  at equal **mean** tie-aware recall@10 over the query set. A 9/10 query counts
  as 0.9 and is not a "failure" for this purpose.

Contract of each figure (train split):

| Policy / figure | Contract used | Achieved mean R | Per-query fail rate | Share 10/10 |
|---|---|---|---|---|
| Fixed ef, 1,082 | Mean recall = R_op (interpolated, §5) | 0.951575 | 29.1% (interpolated between 30.9% at ef 50 and 29.0% at ef 53) | ≈ 71% |
| "Perfect knowledge, any ef", 670 | Mean recall ≥ R_op (Lagrangian allocation over all 121 grid ef) | 0.951575 | 32.1% | 67.9% |
| "Perfect knowledge, tiers", 893 | Mean recall ≥ R_op (Lagrangian over {18, 48, 327}) | 0.951613 | 31.0% | 69.1% |
| Same + probe, 1,284 | As above | 0.951613 | 31.0% | 69.1% |
| DT2 OOF, 2,583 | Not matched: *defines* R_op | 0.951575 | 27.7% | 72.3% |
| Phase 2 oracle (not in the table), 1,137 | Per-query 10/10 for every query | 1.000 | 0% | 100% |

**Findings.**
- **CONFIRMED: all five table rows use the same mean-recall contract.**
  Achieved mean recall is equal to R_op within 4×10⁻⁵. The tier-restricted row
  overshoots slightly because of discreteness, which can only overstate its
  cost, against the audit's own claim.
- **CONFIRMED: the rows are not equal under the per-query contract.** The two
  perfect-knowledge rows fail 31–32% of queries, more than DT2 (27.7%) and
  fixed ef (29.1%). They partly buy their low cost by accepting more 9/10
  queries, which the mean-recall contract permits.
- **CONFIRMED: the "perfect-knowledge any-ef" row (670) is not the Phase 2
  oracle (1,137).** The oracle meets the per-query contract for every query
  (R = 1.0). The audit's wording ("perfect-knowledge router") is accurate, but
  the two must not be conflated.
- **Sensitivity under the per-query contract** (failure rate ≤ 27.725%, the
  DT2 OOF rate). This is explicitly a sensitivity, not a redefinition:

  | Policy | Mean cost |
  |---|---|
  | Fixed ef (cheapest grid ef with failure ≤ 27.725%: ef 56, failure 27.34%) | **1,130** |
  | Perfect knowledge, any ef (cheapest success at own oracle ef, failures at ef 10, failing the queries that save most) | 578 |
  | Perfect knowledge, tiers (success at own tier ef; queries above 327 forced to fail; failures at ef 18) | 857 |
  | … + probe | **1,248** |
  | DT2 OOF | 2,583 |

  The audit's central structural claim, that perfect-knowledge tiers plus probe
  lose to fixed ef, **holds under both contracts** (1,284 vs 1,082 by mean
  recall; 1,248 vs 1,130 by failure rate). So does DT2's loss on train: 2.4× by
  mean recall (2,583 / 1,082), 2.3× by failure rate (2,583 / 1,130).
- **CONFIRMED: the 0.95 target is effectively 10/10 per query.** Under the
  mean-recall contract fixed ef reaches R ≈ 0.95 with about 29% of queries at
  9/10 or below. That is why the label objective (10/10) and the evaluation
  objective (mean) diverge (audit M3). The divergence is documented and
  quantified here; it is not a measurement error.

## 4. Probe-cost findings

**Code-path trace** (Phase 4 and audit code):

| Policy / figure | Code path | Probe |
|---|---|---|
| Router (test eval, train CV) | `router_lib.route()`: `total_dc = search_dc + probe_dc`, with `probe_dc` = Phase 3 `probe_distance_computations` | Included once per query |
| LID ablation | `route(..., probe_plus_lid_dc)` | Both probes (see m2) |
| Fixed ef | `router_lib.fixed()`: `probe_dc = 0` | Excluded (needs no features) |
| Oracle | `evaluate_router.py`: `probe_dc = 0` | Excluded |
| 670 / 893 | `audit_phase4_train.lagrange_alloc(..., extra_cost=0)` | Excluded |
| 1,284 | `lagrange_alloc(..., extra_cost=mean(probe))` | Included as the mean |
| 1,082 | `interp_cost_at_recall(fixed curve)` | Excluded |
| 2,583 | `route()` on OOF DT2 predictions | Included per query |

**Findings.**
- **NOT A PROBLEM: counting is consistent.** Every policy that needs the
  features pays the probe exactly once; every policy that doesn't need them
  doesn't. Verified on train:
  - `total − search == probe` for every one of 8,000 queries;
  - probe dc equals the ef = 10 curve dc for every query;
  - adding the mean probe to the tier-restricted allocation is exact, because
    the probe cost is the same for every option of a query and cannot change
    the allocation.
- **No double counting of the counter.** Probe and routed search are separate
  `searchKnn` calls with a counter reset in between (`SearchAtCurrentEf`).
  Confirmed independently in §6.
- **The four figures do not share one convention, by design.** 670, 893 and
  1,082 exclude the probe; 1,284 and 2,583 include it. That is correct for
  their meaning:
  - 670 and 893 are routers that receive information for free (no-probe bounds);
  - 1,284 is the bound for a router that must compute its features;
  - 1,082 is fixed ef, which never computes features.

  **CONFIRMED wording issue in the audit:** its headline headroom figures
  ("38% … 59% … 76%") are no-probe numbers. The probe-inclusive headroom of a
  perfect-knowledge any-ef router is **+2% / +16% / +41% / +68%** at
  R = 0.952 / 0.97 / 0.99 / 0.999. The audit tabulates both, but its
  executive summary leads with the no-probe values.
- **CONFIRMED: duplicated search *work*,** distinct from double counting. The
  upper-layer greedy descent in hnswlib's `searchKnn` (hnswalg.h:1275–1303)
  does not depend on ef, so the probe and the routed search perform an
  identical descent. Measured with the independent replica on the 5 §6
  queries: 115–135 dc including the entry-point distance, 20–35% of each
  probe. Additive accounting therefore charges at least about 120 dc per query
  for work that is repeated identically.
- **Could probe results be reused?** Not implemented; answered from code only.
  - *Descent reuse:* `HierarchicalNSW::searchBaseLayerST` is a public member
    (class `public:` at hnswalg.h:19). Calling it with the descent's endpoint
    would skip the second descent without modifying hnswlib. It needs the
    endpoint, which `searchKnn` does not return, so the descent would have to
    be re-implemented outside hnswlib. **Feasible, but a scope decision.**
  - *Base-layer state reuse* (continuing the ef = 10 search to a larger ef):
    not possible through the public API; the visited list and heaps are local
    to `searchBaseLayerST`. It would require a custom search loop, which is
    **outside the "hnswlib as black box" boundary**.
  - *Result reuse:* the probe's top-10 is a valid ef = 10 answer
    (mean tie-aware recall 0.71). A router could return it at zero extra cost
    for an "ef ≤ 10" tier, but the frozen tiers have no such option.

## 5. Fixed-ef baseline findings

The 1,082 figure (train) is **interpolated, not measured at one ef value**:
- Source: `rl.fixed_curve` over the 8,000 train queries, from Phase 2
  `oracle_curves.csv` (train rows, tie-aware recall, distance computations).
- Bracketing rows: **ef 50** (mean R 0.947975, 1,037.94 dc, fail 30.86%) and
  **ef 53** (mean R 0.951738, 1,083.59 dc, fail 29.03%).
- Method: linear interpolation in mean recall. Weight on ef 53 =
  (R_op − 0.947975) / (0.951738 − 0.947975) = 0.957, giving **1,081.62**.
- Directly measured at the cheapest grid ef reaching R_op (ef 53): 1,083.59.
  The difference is 1.97 dc (0.18%).
- Maximum possible interpolation error is bounded by the bracket's cost width,
  45.65 dc (4.2%). The actual deviation from the measured neighbour is 1.97.
- The interpolated point is **realisable**. It is the expected cost of a
  randomised fixed policy running ef 53 on 95.7% of queries and ef 50 on 4.3%
  (same per-query curves, linear in expectation). It does not give the fixed
  baseline any unattainable advantage.
- (Test split, quoted only: interpolated 1,075 vs ef 53 measured 1,078.)

**NOT A PROBLEM.** The approximation is at most 4.2% by construction and 0.18%
in practice, against a 2.4× effect.

Related: the perfect-knowledge rows (670, 893) are **derived**, not measured.
- They are Lagrangian allocations: each query picks the option minimising
  cost − λ·recall, with λ bisected until the mean recall reaches R_op.
- The resulting allocation is **feasible** (it reaches R_op on the stored
  per-query curves), so its cost is achievable *with perfect information*. It
  lies within roughly one query's option switch (≤ ~2.5 dc on n = 8,000) of the
  exact optimum.
- **CONFIRMED wording issue:** the audit calls these a "lower bound".
  Precisely, they are achievable perfect-information allocations, essentially
  equal to the optimum. They are lower bounds for any *real* router (which lacks
  perfect information), within that ~2.5 dc.

## 6. Independent distance-count verification

**Method.** No code is shared with the C++ search or evaluation path:
- the saved hnswlib index file is parsed directly in Python (layout of
  `saveIndex`, hnswalg.h:685–713; 96-byte header, verified to consume the file
  exactly);
- hnswlib v0.8.0 `searchKnn` is re-implemented in pure Python: upper-layer
  greedy descent, then `searchBaseLayerST<bare_bone=true>` with its stop rule
  `candidate_dist > lowerBound`;
- libstdc++'s `push_heap` / `pop_heap` are reproduced exactly (comparator on
  distance only, as in `CompareByFirst`), so tie order matches
  `std::priority_queue`;
- distances are computed in float64, which is exact for integer-valued SIFT;
- ground truth comes from NumPy brute force over the 1M vectors read from the
  index file;
- every distance evaluation is counted.

**Queries.** The first training query (lowest id) in each of five DT2 OOF
cells: correct low (q 2), low→high conservative (q 35), high→low aggressive
(q 26), correct med (q 23), correct high (q 1).

**Results.** **20 / 20** search distance counts identical, and 20 / 20
tie-aware recalls identical. **5 / 5** policy totals (probe + routed search)
identical.

| q | Case | Probe (ef 10) replica / pipeline | Routed ef | Search replica / pipeline | Policy total replica / pipeline |
|---|---|---|---|---|---|
| 2 | low → low | 383 / 383 | 18 | 513 / 513 | 896 / 896 |
| 35 | low → high | 472 / 472 | 327 | 5,683 / 5,683 | 6,155 / 6,155 |
| 26 | high → low | 446 / 446 | 18 | 556 / 556 | 1,002 / 1,002 |
| 23 | med → med | 526 / 526 | 48 | 1,048 / 1,048 | 1,574 / 1,574 |
| 1 | high → high | 564 / 564 | 327 | 5,607 / 5,607 | 6,171 / 6,171 |

All four ef values matched for every query (see the JSON).

**CONFIRMED correct:** probe count, final search count and total policy cost.
q 35 also illustrates the cost of a conservative error: it was already 10/10 at
the ef = 10 probe and was sent to ef 327 (+5,683 dc).

## 7. Reconstructed five-row table (training split, R_op = 0.951575)

| # | Row | Mean dc | Recall contract | Probe | Split | Source | Measured or derived |
|---|---|---|---|---|---|---|---|
| 1 | Fixed ef at same recall | **1,081.6** | Mean R = R_op; per-query fail ≈ 29.1% | No (not needed) | Train | `fixed_curve` over Phase 2 train curves; ef 50 ↔ 53 | **Interpolated** between two measured fixed-ef points (realisable as a randomised mix); ef 53 measured = 1,083.6 |
| 2 | Perfect knowledge, arbitrary ef | **670.3** | Mean R ≥ R_op (achieved 0.951575); fail 32.1% | No | Train | Lagrangian over 121 grid ef on per-query curves | **Derived** (perfect-information allocation; per-query values are measured curve points) |
| 3 | Perfect knowledge, tiers 18/48/327 | **892.8** | Mean R ≥ R_op (achieved 0.951613); fail 31.0% | No | Train | Lagrangian over {18, 48, 327} | **Derived** |
| 4 | Row 3 + probe | **1,283.8** | As row 3 | **Yes** (+390.96 mean, exact) | Train | Row 3 + mean Phase 3 probe dc | **Derived** |
| 5 | Actual DT2 router | **2,583.1** | Its own mean R (= R_op by definition); fail 27.7% | **Yes** (per query) | Train (5-fold OOF) | `route()` on saved OOF DT2 predictions + Phase 2 curves + Phase 3 probe | **Measured** (OOF predictions × measured per-query curves) |

Query-bootstrap 95% CIs for every row are in §2. Row 5 on the **test** split
(quoted only, canonical): 2,480 dc at R = 0.950, against fixed 1,075
(interpolated) / 1,078 (ef 53).

## 8. Confirmed issues

| ID | Issue | Status | Severity |
|---|---|---|---|
| V-1 | No index-seed (or CV-seed) replicates; §13 asks for 2–3 seeds for core conditions | UNRESOLVED (missing data) | Major for the strength of the claim; unlikely to flip it (see §10) |
| V-2 | The audit's perfect-knowledge rows use the mean-recall contract and fail 31–32% of queries per query, more than DT2 and fixed | CONFIRMED (documented, quantified) | Minor; conclusions hold under the per-query contract too |
| V-3 | Label objective (10/10 per query) differs from the evaluation objective (mean recall) | CONFIRMED (= audit M3) | Major |
| V-4 | Audit headline headroom (38/59/76%) excludes the probe; probe-inclusive is 2/16/41/68% | CONFIRMED (wording) | Minor |
| V-5 | Probe and routed search repeat an identical upper-layer descent (115–135 dc on 5 queries; 20–35% of the probe) | CONFIRMED (work duplication, not a counting error) | Major for probe accounting (= audit M2) |
| V-6 | The audit calls the perfect-knowledge rows a "lower bound"; precisely, they are achievable perfect-information allocations within ~2.5 dc of the optimum | CONFIRMED (wording) | Minor |
| V-7 | Audit §4 quotes test-split numbers (1,075 / 1,078) beside train-split statements | CONFIRMED (wording) | Minor |
| C1 | Perfect-knowledge 3-tier router + probe loses to fixed ef at the operating point | CONFIRMED: gap −203 dc, CI [−212, −192] (mean contract); 1,248 vs 1,130 (per-query contract) | Critical (unchanged) |
| C2 | DT2 adds the largest loss beyond that | CONFIRMED: +1,299 dc; DT2 − fixed = +1,502 [1,468, 1,538]; fold ratio 0.408–0.434 | Critical (unchanged) |

## 9. Issues that are NOT actually problems

- **Distance counting.** Correct and independently reproduced (20/20 searches,
  5/5 policy totals). No double counting of the counter in any policy.
- **Probe inclusion convention.** Consistent: included exactly where features
  are needed, once per query, exact under aggregation.
- **Fixed-ef interpolation.** 0.18% from the measured neighbour, bounded by
  4.2%, realisable as a randomised fixed policy.
- **Same population.** All rows use the same 8,000 training queries (test
  comparison: the same 2,000), paired by query id.
- **Recall implementation.** The independent replica + NumPy ground truth
  reproduce the pipeline's tie-aware recall exactly at every checked
  (query, ef).
- **Operating-point matching.** All table rows reach R_op within 4×10⁻⁵ (any
  overshoot only raises the bound's cost).
- **Query-sampling noise.** Cannot explain any key gap: every CI is far from 0.

## 10. What remains uncertain

1. **Index-seed variance (V-1).** Unmeasured. The structural gap is −203 dc
   (≈ 19% of the fixed cost) with query-level SD ≈ 5, and the DT2 gap is
   +1,502 dc. Flipping either would need index-to-index variation of a much
   larger order than query sampling shows. That is plausible but unverified;
   only new index builds can settle it.
2. **CV-seed variance.** One fold assignment; fold-to-fold spread is small
   (ratio 0.408–0.434).
3. **Achievable performance of a cost-aware router on these features.** Not
   tested, by instruction; it is what Phase 4b would measure.
4. **How much probe work is legitimately reusable at scale.** The descent was
   measured on 5 queries only (about 120 dc). Base-layer reuse is outside the
   black-box boundary.
5. **Test-split behaviour of any alternative.** Deliberately not examined.
6. **Generalisation to the secondary dataset.** Not yet examined (Phase 9).

## 11. Exact methodological decisions that must be made before Phase 4b

1. **Recall contract.** Mean recall (R*) or per-query success rate (S*) as the
   primary comparison contract, and the same contract for labels, model
   selection and evaluation (V-2, V-3).
2. **Operating point.** The value of R* (or S*), declared from train-only
   evidence before any fitting. Probe-inclusive perfect-knowledge headroom is
   +2% at 0.952, +16% at 0.97, +41% at 0.99 and +68% at 0.999.
3. **Probe accounting.** One of:
   - (a) fully additive (current);
   - (b) additive minus the shared upper-layer descent, via the public
     `searchBaseLayerST`, which needs the descent re-implemented outside
     hnswlib (scope ruling required);
   - (c) report additive and search-only bounds;
   - (d) allow returning the probe's own ef = 10 result as an option (changes
     the tier set).
4. **Tier set.** Keep 18/48/327 frozen, or reopen with a train-only cost-aware
   rule. It must be settled before Phase 5 (§13: identical baselines across
   states).
5. **Training target and decision rule.** Keep tier classification, or switch
   to per-tier success probabilities / expected recall with a cost-aware
   choice (audit §10.3). The model family stays DT/LR.
6. **Seed replication (V-1).** Whether the Phase 4b gate must be shown on 2–3
   independent HNSW index seeds, as §13 requires for core conditions (cost:
   about 554 s build + 13 min oracle per seed), or on seed 42 with replication
   deferred.
7. **Test-set protocol.** A single disclosed second evaluation on the same
   2,000 queries, or a fresh held-out set (a design change).
8. **Regret definition** (audit M4), before Phases 6–8.
9. **Audit wording corrections** (V-4, V-6, V-7): apply to
   `methodology_audit.md`, or leave it as-is with this document as the
   correction record.

---

## Conclusion

**Is the Phase 4 negative result currently strong enough to justify changing
the research methodology?**

**Yes, for changing how the Phase 4 gate is operationalised. No, for changing
the research question.**

What the evidence supports:
- **The measured result is correct.** Distance counts and recall are
  independently reproduced (20/20, 5/5). Probe accounting is consistent with no
  double counting. The fixed baseline is within 0.2% of a measured point.
- **It is robust to query sampling and CV folds.** DT2 − fixed = +1,502 dc
  [1,468, 1,538], and the fold ratio is 0.408–0.434. It holds under both recall
  contracts: on train 2.4× by mean recall and 2.3× by failure rate (test,
  canonical: 2.3× by mean recall).
- **The structural cause is real.** With the frozen tiers and additive probe,
  even perfect knowledge loses to fixed ef at the router's operating point:
  −203 dc [−212, −192] by mean recall, 1,248 vs 1,130 by failure rate. Under
  the current operationalisation, the gate cannot be passed by any model. A
  test that cannot be passed carries no information about whether routing
  works, which is sufficient reason to change the operationalisation (contract,
  operating point, probe accounting, tier set, decision rule).
- **The evidence does not show routing is futile.** Perfect-information
  headroom with the probe reaches +41% at R = 0.99. So there is no basis for
  abandoning or reframing the research question.

The main caveat: everything rests on **one index seed** (V-1). The margins
are large relative to the measured query-level variance, but index-seed
variance is unmeasured. If the methodology is changed, the replication
requirement (decision 6) should be settled at the same time.
