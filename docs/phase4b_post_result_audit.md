# Phase 4b — Post-Result Methodology Audit

Date: 2026-10-02. **Audit only.** No code, config, router, threshold, ladder, contract,
baseline, gate, query set or seed was changed. No experiment was re-run. Phase 5 has
not been started.

**Data used.** New numbers come from read-only calculations on frozen artefacts:
- training-split curves, used for every calibration;
- the saved confirmation per-query rows, used for **post-hoc diagnostics only**.

Post-hoc diagnostics can explain the result. They never feed back into it.

**Notation.**
- **dc** = distance computations per query (the cost metric).
- **Success** = tie-aware recall@10 == 1.0 for a query; **S\*** = the target success rate.
- **B1** = the frozen fixed baseline: a training-calibrated mix of two adjacent ef
  values. **B2** = the hindsight-optimal fixed baseline on the evaluation set.
- **τ** = the router's probability threshold.
- **Ladder** = the frozen candidate efforts {probe result (ef 10), 20, 40, 83, 164, 327,
  647, 1343, 2661}.
- Classifications: **A** = methodology bug, **B** = methodology choice, **C** = no issue,
  **D** = research limitation.

## 1. Audit question and answer in one line

Phase 4b fairly tested whether **this** router class (three probe features, per-candidate
success models, a single global τ, the frozen 9-effort ladder, an additive probe) beats a
fine-grained fixed effort. It did not test query-adaptive effort allocation in general.

The negative result is **real and robust for that class**. One genuine gate-design defect
was found; it cannot change the outcome. Two interpretation statements in the final
report need correcting.

## 2. Reconstruction: design vs implementation vs report

| Item | Design (`design_doc.md`, redesign, bias audit) | Implemented | Report claims | Discrepancy |
|---|---|---|---|---|
| Research question | Per-query effort allocation meets a quality target with less computation than a fixed effort | Tested for one router class | Stated correctly | None |
| Index | hnswlib, M = 16, efC = 200, seeds 42/43/44 | Same; single-threaded builds; distinct hashed index files | Same | None |
| Populations | Train 8,000 / old test 2,000 (seed 20261001); fresh confirmation set | Train 8,000; confirmation v2 (2,000, seed 20261007); old test as second look | Same | v1 rejected and disclosed |
| Oracle | Per-query minimum ef reaching 10/10 (stable reach) | Same; grid 10–4096, ×1.05 | Same | None |
| Features | Design §9: knn, centroid, concentration, "**reused** from a shallow initial graph probe" | A separate ef = 10 `searchKnn` probe; its result is reused only if chosen | "Probe … fully additive" | Design says reused, implementation is additive (known since audit M2; disclosed; sensitivity reported) |
| Router output | Design §9: discrete tier (low/med/high) | 9-value ladder (approved redesign) | Same | Approved change |
| Router model | DT / LR | P(success at c \| x) per candidate, cumulative max, cheapest c with P ≥ τ | Same | None |
| τ | Train OOF, smallest τ with OOF quality ≥ target | Same | Same | None |
| Fixed baselines | Design §7.6: "three constant efSearch values chosen from the S0 curve" | B1 = training-calibrated **randomised mix of two adjacent grid ef** per S\* | "B1 two-ef mix" | B1 is not a single constant ef; a per-S\* fixed policy (approved; disclosed here) |
| Quality contract | Per-query success, co-primary S\* ∈ {0.90, 0.95, 0.99}; mean recall secondary | Same | Same | None |
| Cost | dc, additive probe | Same | Same | None |
| Gate | Bias audit §10 criteria 1–4, all three S\* | Same code; criteria applied exactly | Same | Criterion 2 has a power defect (§15) |
| Statistics | Query-clustered bootstrap, seed-averaged Wilcoxon, McNemar | Same | Same | None |
| Interpretation | — | — | §15: "core failure is discrimination among harder queries … routes many medium and hard queries to expensive efforts that rarely change their outcome" | **Partly inaccurate** (§8, §11 below) |

## 3. Baseline fairness

**What B1 is.** Two adjacent grid ef values (for example 164/172 at S\* = 0.95),
calibrated on the **training** split to reach S\* in expectation. Each query is assigned
by a content-independent hash.

- **Realisable** without any query information.
- **Not tuned** to the confirmation set: its realised confirmation success is 0.9537, the
  right level.
- Gets **no** information the router lacks. It uses less: no features.

**What B1 has that the router lacks.**
1. **Fine resolution of the fixed operating point.** B1 can use any of the 121 grid
   values. The router's fixed-like choices are limited to the 9 ladder values.
2. **No probe.**

So the comparison is **"training-optimised fine-grained fixed policy" vs "adaptive
policy restricted to a coarse ladder, plus probe"**. The router's policy class does not
contain B1: it cannot emulate "everyone at ef ≈ 166".

**Decomposition** (post-hoc, confirmation, pooled over seeds; calibrations on training):

| S\* | B1 | Ladder-only fixed mix (train-calibrated) | + probe | Router | Router − B1 | from ladder restriction | from probe | from adaptive allocation |
|---|---|---|---|---|---|---|---|---|
| 0.90 | 1,968 | 2,147 (83/164) | 2,540 | 2,379 | +411 | +179 | +393 | **−161 (router 6.3% cheaper than same-class fixed)** |
| 0.95 | 2,651 | 2,673 (164/327) | 3,066 | 2,886 | +235 | +22 | +393 | **−180 (5.9%)** |
| 0.99 | 4,485 | 4,544 (164/327) | 4,937 | 4,436 | −48 | +59 | +393 | **−501 (10.1%)** |

**Verdict.**
- B1 is a **legitimate strong baseline, not an unfair one.** It is non-adaptive,
  training-calibrated, realisable, and "fixed search effort" naturally means "any ef the
  operator chooses".
- The policy-class asymmetry is a **methodology choice (B)**. Its cost is the
  ladder-restriction column. Removing it would still leave the router at −11.8% (0.90)
  and −8.0% (0.95), so it **cannot explain the failure**.

## 4. Candidate ladder

- **Fixed before evaluation:** yes. It was pre-declared and data-independent, before the
  redesign feasibility numbers existed.
- **Spacing.** The 164 → 327 gap straddles B1's 0.95 operating point. 83 → 164 straddles
  B1's 0.90 point (111/117). That is ordinary **discretization loss**, not unfairness:
  +179 / +22 / +59 dc (§3).
- **Oracle headroom** (47%) is computed on the **same ladder and probe**, so it is
  internally consistent. The full-grid oracle would be lower still (training feasibility:
  +54% vs +46.5% at 0.95).
- **Verdict: B (choice) for the spacing, D (limitation) for scope.** Ladder granularity
  cannot explain the negative result at 0.90 or 0.95.

## 5. Probe

**Work performed.**
- **Probe:** a full `searchKnn(k = 10, ef = 10)`: upper-layer descent (mean ≈ 118 dc)
  plus a base-layer search. 393 dc on average, identical to the ef = 10 curve for every
  query.
- **Routed search:** a fresh `searchKnn` at the chosen ef. The descent is repeated
  identically; base-layer work is partly re-done.
- **B1:** one `searchKnn`, no probe.

**Reuse.**
- The probe's **result** is reused when ef 10 is chosen (0.5–0.9% of queries).
- Descent reuse is possible through the public `searchBaseLayerST`, which needs the
  descent re-implemented outside hnswlib (a scope question).
- Base-layer state reuse is impossible through the public API.

**Assessment.**
- Charging the full probe is **consistent with "black-box hnswlib"**: it is decision
  overhead a deployable router built this way would really incur.
- It is conservative relative to design §9's "reused" wording.
- **Verdict: B (choice), disclosed.** Shared-descent and probe-free sensitivities were
  reported.

**Probe-free sensitivity check.**
- It charges 0 for queries routed to the probe result, which counts that search as free.
  Strictly, that search is the answer's own search.
- Affected share: 0.9% / 0.5% / 0.0% of queries. Corrected values −1.0% / +5.9% / +9.8%
  (reported −0.9% / +6.0% / +9.8%).
- **A, minor, non-gating.** It favours the router slightly and cannot change any
  conclusion.

## 6. Quality contract

- **Per-query 10/10 success rate** follows from design §2.3 and §12 ("fraction of failed
  queries"). It is discontinuous: 9/10 counts as a failure.
- Router and B1 are judged identically. The router's training labels use the **same
  success definition**.
  - Small difference: labels use *stable* success from candidate c onward, while
    evaluation uses success *at* c. This is only slightly conservative, because
    non-monotone recall curves are rare (about 1 in 10,000 on the original grid).
- **Mean recall** was pre-registered as the secondary contract. It tells the same story,
  more strongly (−15% to −44%).
- No post-hoc selection was possible: both families were evaluated and both are negative.
- **Verdict: C**, with a **D** note (claims are scoped to these contracts and k = 10).

## 7. Router-training objective

**What is optimised.**
- Each candidate model maximises the likelihood of P(success at c | x) (LR, or gini
  trees).
- Model selection: lowest out-of-fold total cost subject to out-of-fold success ≥ S\*.
- τ: the smallest value meeting S\* out-of-fold.

These are aligned with the evaluation objective at the **aggregate** level.

**Structural mismatch (B, material).** The decision rule "cheapest c with
P(success at c) ≥ τ" applies **one probability threshold to every query**. It is not a
cost-minimising allocation under a global success constraint, which would compare each
query's marginal success gain per dc.

Consequences:
- Queries whose probability rises slowly are pushed up the ladder whether or not that is
  cost-effective.
- Weakly identified easy queries are kept well above their need.
- Post-hoc evidence at S\* = 0.95: **easy** queries (oracle ef ≤ 18) are routed to 83
  (43.5%) or 164 (19.2%). Overspend on successful queries averages 1,332 dc.

**Is the router "systematically conservative"?** Not in aggregate quality. Realised
success is 0.948 against S\* = 0.95 and B1's 0.954. Its inefficiency is **mis-allocation
under weak per-query information**, not over-delivery of quality.

**Could useful probabilities still fail to save computation?** Yes. Within-stratum
feature signal is weak (Spearman 0.10–0.34 inside easy, medium and hard), so a
per-query probability guarantee forces wide safety margins.

**Verdict.** The decision rule is a **B (approved formulation)** whose effect on the
conclusion is **unknown**. Resolving it would need a new, pre-registered decision rule
evaluated on fresh data. It cannot be tested post hoc on the confirmation set.

## 8. Threshold selection

- **Data:** 5-fold out-of-fold predictions on training only, with folds stratified on the
  candidate-oracle index. Confirmation and old-test data were untouched (verified by data
  access records and hashes).
- **Nesting:** τ and family selection both use the same out-of-fold predictions, which
  gives a mild selection optimism. On new data the router reached 0.9478 (out-of-fold
  0.9504), while B1 reached 0.9537. This costs the router on quality criterion 2, not on
  cost.
- **What τ optimises:** τ meets the success constraint at the cheapest **global
  threshold**. It does **not** optimise cost subject to quality across queries (§7).
- **Verdict: B.** Correctly executed and leakage-free. Its optimality is limited by the
  formulation.

## 9. Oracle and headroom

**Calculation.** The contract oracle uses the same 9 candidates, the same additive probe
and the same success definition. With perfect per-query knowledge, it makes exactly
⌈S\*·n⌉ achievable queries succeed at their cheapest stable candidate, and serves the
rest at the cheapest candidate.

At 0.95: 1,397 (seed 42) vs B1 2,651, i.e. **47%** lower.

**Validity.**
- It is a valid **achievable lower bound for any policy in the router's class with
  perfect information**: ladder-restricted, probe-charged, same contract.
- Its advantages over the router are pure information, including **knowing which 5% to
  fail** (the most expensive ones).
- It is not directly a "fixed vs adaptive" comparison, since the class differs from B1's,
  but both are valid policies at the same S\*.

**Conclusion.**
- "Substantial theoretical headroom exists" is **justified** as a perfect-information
  bound.
- "The current router cannot exploit it" is **justified**: 2,886 vs 1,397.
- It does **not** follow that a *realistic* router could exploit it. The achievable
  fraction depends on information content (§10).
- **Verdict: C.**

## 10. Features and information

- **Availability.** All features are computed from the probe search and the base-set
  centroid before the routed search. No ground truth or labels are involved:
  **leakage-free (C)**.
- **Correlation transfer.** Training Spearman (knn +0.46, centroid −0.45, concentration
  +0.36, LID +0.56) matches confirmation (+0.48, −0.42, +0.41 to +0.44, +0.60):
  **comparable (C)**.
- **Probe artefacts.** knn_dist is the probe's approximate 10th-neighbour distance, so
  part of its signal comes from the probe's own difficulty. That is legitimate: a real
  router has exactly this information.
- **Within-stratum information is weak.** Spearman inside easy / medium / hard is
  0.10–0.34. The features order queries coarsely but barely discriminate within a
  difficulty band.

  At S\* = 0.95, hard queries were routed 16% → 83, 51% → 164, 29% → 327, 1.5% → ≥ 647.
  Their success equals B1's (0.854 vs 0.855). The few sent to ≥ 647 do succeed (1.000
  vs B1's 0.517), but they are rare.
- **Verdict: D (research limitation).** The observed failure is consistent with the
  information content of these features under this decision rule.

**Correction to the final report.** §15 says the router "routes many medium and hard
queries to expensive efforts that rarely change their outcome". Post hoc, expensive
routing (≥ 647) is **rare** (≤ 1.5%). Where it happens, it does change the outcome. The
larger inefficiencies are:
- the probe (393 dc);
- conservative routing of easy and medium queries to 83–164;
- an adaptive gain (6–10% versus a same-class fixed policy) that is too small to cover
  the probe.

**Classification: A** (an error in the report's interpretation). **Minor**: it does not
bias the measured comparison and does not change the conclusion.

## 11. Query set

**Construction.**
- Filtered (exact copies of the original queries removed), deduplicated, sampled with
  Fisher–Yates on the filtered population (seed 20261007).
- Independently regenerated.
- 0 exact overlap with the base set, the original queries and itself.

**Diagnostics.**
- Norms indistinguishable from the original queries.
- Per-dimension mean difference smaller than the original train-vs-test difference.
- Nearest-original-query distances match the same-population reference.

**Consistency.**
- Difficulty mix (34 / 34 / 32%) matches training (33 / 34 / 32%).
- B1 calibration transferred (success 0.954).
- The old-test second look agrees (−19.3% / −9.3% / +0.8%).

**Verdict: C.** No selection mechanism plausibly disadvantages the router. The source
file differs (`sift_learn`), but every label-free and outcome check is consistent.

## 12. Index seeds

- Only the level RNG differs; there are three distinct index files.
- Results are consistent:
  - per-seed savings −20.3 / −22.0 / −20.3% at 0.90;
  - −9.0 / −9.4 / −8.2% at 0.95;
  - +0.4 / +0.7 / +2.2% at 0.99.
- Seed-specific censoring (2 old-test queries on seed 42 only) is immaterial.
- **Verdict: C**, with a **D** note (three seeds of one configuration).

## 13. Cost metric

- dc is hardware-independent and deterministic, and it was design-specified as primary.
- Not measured: latency and QPS.
- The router's real-time overhead goes beyond dc: two `searchKnn` calls (two
  visited-list setups), feature computation and model inference (9 small models). That
  makes it **unlikely to be better in latency** while being worse in dc.
- A router better in latency but worse in dc is implausible here. The reverse (better in
  dc, worse in latency) is plausible, but did not occur.
- **Verdict: D.**

## 14. Statistics

- **Correct:**
  - query-clustered bootstrap (seeds of a query resampled together, 2,000 resamples);
  - Wilcoxon on seed-averaged per-query differences (n = 2,000);
  - exact McNemar test per seed;
  - B2 recomputed inside every bootstrap resample.
- **Criterion 1 interpretation.** It requires point saving ≥ 10% **and** CI lower bound
  > 0. That is a lenient formulation (CI lower bound ≥ 10% would be strict). For this
  *negative* result the **upper** bounds matter: −18.5%, −6.8% and +2.9% all exclude
  +10%. Savings of 10% are statistically implausible at every level in this
  configuration. **C.**
- **Wilcoxon estimand.** It tests the pseudo-median of paired differences, while
  criterion 1 is about the mean. At 0.95 and 0.99 the router is cheaper in median and
  pseudo-median (rank-biserial −0.10 / −0.27) but not meaningfully in mean. The
  two-sided p < 0.01 is not direction-checked against the mean. **B, minor.** Here mean
  and median disagree in direction, which the report discloses.

## 15. Gate-design defect: criterion 2 is underpowered (A, material for gate validity; does not change this conclusion)

Criterion 2 requires the **95% CI lower bound** of Δsuccess (router − B1) ≥ −0.01. The
observed bootstrap standard errors are 0.0065 / 0.0051 / 0.0021 at S\* = 0.90 / 0.95 /
0.99. A router with **exactly equal true success** would therefore pass criterion 2 with
probability about **0.34 / 0.50 / 1.00**.

The bias audit's justification ("margin ≈ 2 SE at n = 2,000") was **wrong**: it ignored
that the lower bound, not the point estimate, must clear the margin.

- **Effect here: none on the conclusion.** At 0.90 and 0.95 criterion 1 fails with CIs
  entirely below 0. At 0.99 criterion 1 fails (+1.1%, CI upper bound +2.9% < 10%) while
  criterion 2 passes.
- **Required for any future gate:** a pre-registered power analysis for the
  non-inferiority margin (larger n, a wider margin, or a point-estimate-based criterion).
  Not applied here.

## 16. Reproducibility: what it does and does not establish

**Establishes.**
- Ground truth is correct (independent brute force).
- The evaluation reads search outcomes correctly (72,000 C++ re-search matches).
- The code is deterministic (re-run identical).
- Frozen artefacts were used unchanged (hashes).
- Units behave as specified (tests).

**Does not establish.**
- Fairness of the design (§3, §7, §15).
- Optimality of the router class.
- External validity (other datasets, k, index configurations, latency).
- That the gate thresholds are well calibrated (§15).

## 17. Claim audit

| Claim | Supported? | Why |
|---|---|---|
| A. The specific Phase 4b router implementation failed to achieve the required 10% savings | **Yes, strongly** | All three S\*, all seeds, two query sets. CI upper bounds exclude +10%. Robust to ladder restriction, probe-free accounting and the criterion-2 defect. |
| B. The selected feature set + probability-routing formulation failed to achieve the required savings | **Yes, with scope** | True for: these three features; per-candidate LR/DT success models; a single global τ; the frozen ladder; an additive black-box probe; SIFT1M; k = 10. Not established for other decision rules over the same features (e.g. a cost-aware allocation), §7. |
| C. Query-level ANN search-effort routing does not provide computational savings | **No, too strong** | Within its own class the router **does** save 6–10% versus a same-class fixed policy. Perfect information would save 47%. The deficit against B1 is mainly the probe overhead plus weak per-query information under one decision rule. |
| D. Query-level ANN search-effort routing is not viable for this problem | **No, too strong** | Only one router class, one probe accounting and one dataset were tested. Viability depends on cheaper or reusable decision information and richer per-query signals, none of which were tested. |

## 18. Final audit table

| Area | Finding | Class | Severity | Could change conclusion? | Evidence |
|---|---|---|---|---|---|
| Research question | Tested for one router class, not in general | D | material | No (scopes it) | §2, §17 |
| Index / seeds | Correct; consistent across seeds | C | — | No | §12 |
| Confirmation set | Clean, representative; old test agrees | C | — | No | §11 |
| Oracle | Correct, consistent with design | C | — | No | §2, §9 |
| Headroom claim (47%) | Valid perfect-information bound in the router's class | C | — | No | §9 |
| B1 fairness | Legitimate strong baseline; realisable; training-only | C | — | No | §3 |
| Policy-class asymmetry (fine-grid fixed vs coarse-ladder adaptive) | Real; costs +179 / +22 / +59 dc | B | material | No (router still −11.8% / −8.0% / +2.4% without it) | §3 decomposition |
| Ladder spacing | Fixed ex ante; discretization loss only | B | minor | No | §4 |
| Probe accounting (additive) | Conservative vs design's "reused"; realistic for black-box hnswlib; 393 dc | B | material | No (probe-free: −1.0 / +5.9 / +9.8%, all < 10%) | §5 |
| Probe-free sensitivity | Charges 0 for probe-result queries | A | minor | No | §5 (≤ 0.1 point) |
| Decision rule (global τ on P(success)) | Not cost-optimal across queries; drives conservative easy/medium routing | B | material | Unknown; would need a new pre-registered experiment | §7 |
| τ selection | Leakage-free, train-only, mild out-of-fold optimism | B | minor | No | §8 |
| Training labels vs evaluation success | Stable-reach vs at-c success; negligible | C | — | No | §6 |
| Feature information | Weak within-stratum signal (ρ 0.10–0.34) | D | material | No (explains it) | §10 |
| Quality contract | Appropriate; mean recall agrees | C | — | No | §6 |
| Cost metric | dc only; no latency | D | minor | No (latency would not favour the router) | §13 |
| Gate criterion 2 power | P(pass \| equal success) ≈ 0.34 / 0.50 | A | material (gate validity) | No (criterion 1 fails independently) | §15 |
| Wilcoxon estimand | Pseudo-median vs mean; not direction-checked | B | minor | No | §14 |
| Criterion 1 leniency | Point ≥ 10% with CI > 0, not CI ≥ 10% | B | minor | No (CI upper bounds exclude 10%) | §14 |
| Report §15 interpretation | Mis-describes the failure mode; under-states the probe's role and the router's in-class gain | A | minor | No | §10 |
| B1 vs design §7.6 wording | Randomised two-ef mix per S\*, not "three constant values" | B | minor | No | §2 |
| Reproducibility | Execution verified; fairness and external validity not | C (with D note) | — | No | §16 |

### A. Must-fix methodology issues (genuine bugs)
1. **Gate criterion 2 is underpowered.** A router with equal success fails it 50–66% of
   the time at S\* = 0.90 and 0.95. Must be fixed (power analysis) before any future gate
   relies on it. It does **not** affect the Phase 4b outcome.
2. **The final report's §15 failure-mode statement needs correcting** (documentation
   only).
3. **The probe-free sensitivity definition** should charge the returned probe search
   (documentation; at most 0.1 point).

None of these requires re-running Phase 4b.

### B. Important but defensible choices (disclose)
- **Policy-class asymmetry:** the fixed baseline is fine-grained, while the router is
  restricted to the ladder plus a probe.
- **Additive black-box probe** against the design's "reused" wording.
- **Single global probability threshold** as the decision rule; not cost-optimal across
  queries.
- **B1 as a randomised two-ef mix** rather than a single constant ef.
- **Wilcoxon estimand and criterion-1 leniency.**

### C. Robust aspects
- Ground truth and oracle correctness.
- Leakage-free features and training.
- Training-only calibration of the router and B1.
- Integrity and representativeness of the confirmation set.
- Seed consistency.
- Query-clustered statistics.
- Determinism.
- The negative cost result itself:
  - CI upper bounds −18.5% / −6.8% / +2.9%;
  - robust to removing the ladder restriction;
  - robust to a free probe;
  - robust to the criterion-2 defect.

### D. What the negative result legitimately establishes
On SIFT1M at k = 10, with black-box hnswlib, a router that:
- pays a full ef = 10 probe for three cheap features;
- predicts per-candidate success with LR or small trees;
- routes with a single training-calibrated probability threshold over the frozen
  9-effort ladder

**does not** reduce distance computations relative to a training-calibrated fixed effort
at per-query success targets of 0.90, 0.95 or 0.99. It costs 9–21% more at 0.90 and
0.95, and is at parity (≤ 2.9% saving) at 0.99. Mean-recall targets are worse.

Within its own class, the router's adaptive allocation **does** save 6–10% versus a
comparable fixed policy, but that gain is smaller than the probe's overhead (393 dc,
about 15% of B1's cost at 0.95), except at 0.99.

### E. What it does NOT establish
- That query-adaptive effort allocation cannot save computation in general.
- That these features carry no usable signal: they do, but coarsely.
- That a different decision rule over the same features would fail, e.g. a cost-aware
  allocation instead of a per-query probability guarantee.
- That a probe whose work is reused (shared descent or a resumed search) would fail.
- Anything about latency, other datasets, other k, or other index configurations.
- That Phase 4b's gate is well calibrated for quality non-inferiority.

### F. Recommended decision (no experiment designed here)
1. **Accept Phase 4b as a valid negative result**, scoped as in D and E. No genuine bug
   invalidates it, and none of the found defects can flip it.
2. **No methodological correction or re-run of Phase 4b is required.** The gate defect
   (criterion 2) would only matter for a borderline pass, which this is not.
3. **A limited sensitivity experiment is not required to support the Phase 4b
   conclusion.** It would only be justified to answer a *different* question that
   Phase 4b cannot answer: how much of the deficit is decision-rule and probe-design
   choice versus feature information. If pursued, it must be a new, pre-registered
   experiment on fresh data, with the criterion-2 power defect fixed first. That is a
   research decision for you.
4. **Phase 5 remains blocked.** Design §21 requires an S0 advantage, and Phase 4b found
   none.
