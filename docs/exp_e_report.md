# Experiment E report — calibration staleness of S0-frozen policies under index evolution

Design: docs/design_doc.md Addendum E1 (= docs/exp_e_design_freeze_v2.md, FROZEN 2026-10-05).
Rows: results/exp_e_rows_20261005T110522Z. Analysis: results/exp_e_analysis_20261005T110646Z
(repeat results/exp_e_analysis_20261005T110753Z, byte-identical). Precheck:
results/exp_e_precheck_20261005T105734Z.

Scope: SIFT1M, L2, hnswlib M16/efC200, k = 10; the 30 A1 states (1–8% insertion churn,
in-distribution and regional "OOD"; 2–8% lazy deletion); 2,000 Phase-2 test queries;
3 index seeds. "OOD" is a narrow construction (regionally concentrated real SIFT vectors). At
8% insertion magnitude, the available pool forces the regional-OOD construction to overlap
approximately 91% with the matched ID set, so this level does not provide a clean ID-vs-OOD
comparison.

## A. Execution integrity
- **Freeze:** FROZEN — FINAL. Implementation sha256 recorded before execution
  (docs/notes.md, "Experiment E — FINAL FREEZE").
- **Policies:** policy and model hashes verified in every stage:
  - seed 42: policy bcbe2720…, model d291a2f1…;
  - seed 43: policy 6b2275ef…, model 9003ba8b…;
  - seed 44: policy df0bd726…, model 21a27a77….
- **Precheck: ALL PASS, all seeds:**
  - REF(S0) ≡ frozen B1 (mix and per-query ef);
  - B1 ef, cost and recall equal the Phase-4b rows;
  - DARTH E-pipeline S0 rows equal the frozen S0 runs (non-timing columns).
- **Cells:** 33 DARTH cells (3 seeds × S0 + 10 states), plus B1 and REF rows for all 33. Each
  has 2,000 queries, and the §16 checks pass for every cell:
  - recall in [0, 1];
  - 10 labels returned;
  - no deleted or out-of-range label.
- **Analysis:** two runs on identical inputs are byte-identical.

## B. H-E1 — cost staleness (primary D, margin ±0.10)
| State | B1 D̄ [95% CI] | class | DARTH D̄ [95% CI] | class |
|---|---|---|---|---|
| id 1% | −0.0005 [−0.001, −0.000] | STABLE | −0.0055 [−0.008, −0.003] | STABLE |
| id 2% | +0.0027 [+0.002, +0.003] | STABLE | −0.0064 [−0.010, −0.003] | STABLE |
| id 4% | −0.0004 [−0.001, −0.000] | STABLE | −0.0127 [−0.017, −0.008] | STABLE |
| id 8% | +0.0005 [+0.000, +0.001] | STABLE | −0.0210 [−0.027, −0.015] | STABLE |
| ood 1% | −0.0001 [−0.000, −0.000] | STABLE | −0.0021 [−0.004, +0.000] | STABLE |
| ood 2% | +0.0018 [+0.002, +0.002] | STABLE | −0.0044 [−0.007, −0.001] | STABLE |
| ood 4% | +0.0040 [+0.004, +0.005] | STABLE | −0.0070 [−0.011, −0.003] | STABLE |
| ood 8%¹ | +0.0016 [+0.001, +0.002] | STABLE | −0.0211 [−0.027, −0.015] | STABLE |
| del 2% | +0.0157 [+0.015, +0.017] | STABLE | −0.0034 [−0.006, −0.001] | STABLE |
| del 8% | +0.0524 [+0.051, +0.054] | STABLE | −0.0224 [−0.027, −0.018] | STABLE |

- **Per seed:** all 60 per-seed cells are STABLE.
- **Family verdicts:** STABLE-FAMILY for both policies × both families.
- **Mean cost, B1 / REF (seed 42):**
  - S0: 1057 / 1057;
  - del 8%: 1125 / 1068;
  - id 8%: 1064 / 1064.
- **Mean cost, DARTH (seed 42):** S0 1172, id 8% 1157, del 8% 1158.
- **Raw R (descriptive):** B1 equals D. DARTH's raw R is +0.084 to +0.112 at every state, which
  is the S0 C1 gap (+0.103 to +0.115 at S0); anchoring removes it.
- **Additive ΔR (descriptive):** within ±0.003 of D everywhere.
- **Mean tie-aware recall@10** (seed means):

  | | S0 | id 8% | ood 8% | del 8% |
  |---|---|---|---|---|
  | B1 | 0.9490 | 0.9491 | 0.9497 | 0.9540 |
  | REF | 0.9490 | 0.9490 | 0.9496 | 0.9487 |
  | DARTH | 0.9633 | 0.9614 | 0.9617 | 0.9579 |

  The largest B1 drift (del 8%, +5.2%) comes with higher B1 recall than REF (0.954 vs 0.949):
  after deletion the reference recalibrates to a cheaper mix.

¹ 8% OOD ≈ ID by construction (see the scope note).

## C. H-E2 — contract staleness (excess newly failing, margin ±0.05)
| State | B1 Ē [95% CI] | class | DARTH Ē [95% CI] | class |
|---|---|---|---|---|
| id 1% | +0.0000 [0.000, 0.000] | STABLE | +0.0134 [+0.007, +0.020] | STABLE |
| id 2% | −0.0010 [−0.002, −0.000] | STABLE | +0.0202 [+0.011, +0.030] | STABLE |
| id 4% | −0.0002 [−0.001, 0.000] | STABLE | +0.0286 [+0.017, +0.040] | STABLE |
| id 8% | −0.0005 [−0.001, 0.000] | STABLE | +0.0326 [+0.018, +0.046] | STABLE |
| ood 1% | +0.0000 [0.000, 0.000] | STABLE | +0.0129 [+0.007, +0.019] | STABLE |
| ood 2% | −0.0007 [−0.002, 0.000] | STABLE | +0.0212 [+0.013, +0.030] | STABLE |
| ood 4% | −0.0031 [−0.006, −0.001] | STABLE | +0.0117 [+0.002, +0.022] | STABLE |
| ood 8%¹ | −0.0012 [−0.002, −0.000] | STABLE | +0.0243 [+0.010, +0.037] | STABLE |
| del 2% | −0.0075 [−0.012, −0.004] | STABLE | +0.0022 [−0.003, +0.007] | STABLE |
| del 8% | −0.0075 [−0.012, −0.004] | STABLE | +0.0288 [+0.020, +0.038] | STABLE |

- **Family verdicts:** STABLE-FAMILY for both policies × both families.
- **Per seed:** no seed-specific STALE. DARTH per-seed INCONCLUSIVE at id 8% (seeds 42, 44)
  and del 8% (seed 42), with CI upper bounds just above 0.05 (E = 0.034–0.038).
- **Example newly-failing rates** (seed 42, frozen / reference):
  - B1, id 8%: 0.0792 / 0.0792; transitions 1268 / 109 / 510 / 113 (stable-pass /
    newly-failing / stable-fail / improved);
  - DARTH, id 8%: 0.1154 / 0.0792; transitions 1296 / 169 / 394 / 141.
- **Cohort sizes:** K_B1 = 1377 / 1388 / 1393; K_DARTH = 1465 / 1472 / 1465.
- **Caveat (frozen):** DARTH's cohort differs in composition from K_REF = K_B1.

## D. H-E3 — structural predictors (insertion, 24 cells; Holm over 16)
| Policy | Outcome | update fraction | overlap decay | density drift | distributional drift |
|---|---|---|---|---|---|
| B1 | D | +0.22 (p_Holm 0.008) ✓ | +0.16 (0.008) ✓ | −0.22 (0.008) ✓ | +0.12 (0.008) ✓ |
| B1 | E | −0.26 (0.048) ✓ | −0.18 (0.054) | +0.22 (0.048) ✓ | −0.18 (0.136) |
| DARTH | D | −0.89 (0.008) ✓ | −0.93 (0.008) ✓ | +0.91 (0.008) ✓ | +0.67 (0.008) ✓ |
| DARTH | E | +0.65 (0.063) | +0.72 (0.060) | −0.69 (0.063) | −0.64 (0.060) |

- ✓ = associated (Holm-adjusted p < 0.05). Raw bootstrap p-values and 95% CIs are in
  analysis.json; the p floor is 1/2000.
- **10 of 16 tests are associated.** All four DARTH-E tests miss after Holm (raw p 0.012–0.028).
- **Structural measures vs update fraction (secondary):** no structural measure showed a
  detectably stronger association than update fraction; distributional drift was detectably
  weaker for B1 cost drift. (|ρ| differences −0.22 to +0.07; every CI includes 0 or is
  negative.) This does not show that update fraction is the best predictor.
- **Per-query (secondary):** overlap decay vs newly failing within the S0 cohort ρ = 0.21 (B1)
  and 0.24 (DARTH). vs per-query cost difference |ρ| ≤ 0.08.
- **Deletion (descriptive, 6 cells, 2 magnitudes):** ρ is mostly ±0.88, which with 2
  magnitude levels is uninformative.
- **What these associations concern:** sub-threshold drift only. No state was STALE (§B, §C),
  so H-E3 says nothing about predicting actual policy invalidation.
- **Dependence caveats (H-E3 remains secondary):**
  - the predictors are highly collinear with evolution magnitude;
  - update fraction and distributional drift are state-level constants repeated across seeds,
    and overlap decay is seed-independent (it comes from ground truth);
  - so the 24 cells carry far fewer independent predictor values (about 8);
  - the query-bootstrap p-values ignore that dependence and should be read cautiously.

  The frozen Holm correction is applied as specified.

## E. H-E4 — DARTH predicted vs realised recall (descriptive; stopped queries only)
- **Stopped fraction:** 0.943 at S0; 0.942–0.946 at every state.
- **ID recall (primary):**
  - signed bias −0.0013 at S0 (seed mean); ranges −0.0010 to +0.0046 across states;
  - MACE 0.0537 at S0; 0.0537–0.0569 across states.
- **Largest change from S0:** del 8%, bias +0.0059 [+0.0044, +0.0075] and MACE +0.0032
  [+0.0021, +0.0042]. All insertion changes are ≤ 0.002 in bias and ≤ 0.001 in MACE.
- **Tie-aware (secondary):** the same pattern (del 8%: bias change +0.0062).
- **Small-n:** no cell is flagged.
- **Disclosure:** the comparison is conditional on early-stopped queries (selection at
  predicted ≥ 0.95), so it is not probabilistic calibration. Realised recall is discrete, so
  MACE has an irreducible floor. Cap-run queries are excluded.

## F. Secondary trends (§11)
- **Insertion mixed models (ID slope per doubling of magnitude, 95% Wald CI):**
  - B1 D: +0.00002 [−0.001, +0.001]; no change;
  - B1 E: −0.00007 [−0.0006, +0.0005]; no change;
  - **DARTH D: −0.0053 [−0.0071, −0.0035]; changes with magnitude;**
  - **DARTH E: +0.0066 [+0.0040, +0.0092]; changes with magnitude.**
- **Fit status:**
  - all four mixed-effects fits converged;
  - all four carry the "MLE may be on the boundary" warning (seed variance ≈ 0);
  - the two DARTH fits additionally required statsmodels' built-in retry with L-BFGS after the
    first optimiser did not converge. This is part of statsmodels' default fitting; the frozen
    model specification (y ~ L + T + L:T, seed random intercept, REML) is unchanged.
  - Warnings are reported, not acted on, and are not scientific evidence.
- **Robustness OLS slopes:** these agree with the mixed models. DARTH D: id −0.0053, ood −0.0060.
- **Deletion (del 8% − del 2%):** B1 D +0.037; DARTH D −0.019; DARTH E +0.027; B1 E 0.000.
- **Overhead (descriptive):** about 7.9–8.0 predictor calls per query. Wall-clock times
  (0.98–4.2 ms per query) were higher than in C1 for reasons not established here and are not
  interpreted. The experimental cost metric is distance computations.

## G. Interpretation (scope: the frozen design only)
**Results under the frozen decision rules:**
- **H-E1:** both frozen policies are STABLE at every state, in both families. Within 1–8%
  insertion churn and 2–8% lazy deletion, neither the S0-frozen B1 mix nor the S0-frozen DARTH
  policy drifts in cost by more than 10% relative to a freshly calibrated B1-style reference.
  - Observed drift: B1 ≤ +5.2% (del 8%); DARTH −2.2% to −0.2%.
- **H-E2:** both are STABLE at every state, in both families. The excess newly-failing rate
  stays within ±5 pp: B1 ≈ 0; DARTH +0.2 to +3.3 pp.
  - Three DARTH per-seed cells are INCONCLUSIVE (upper bound just above 5 pp); none is STALE.
- **H-E3 (secondary):** structural measures are associated with D for both policies (all 8
  tests) and with B1's E (2 of 4, both at p_Holm = 0.048).
  - For DARTH's E, no association survives Holm.
  - All of these concern sub-threshold drift: the B1 effects are very small (|D| ≤ 0.053) and
    DARTH's |D| ≤ 0.026. H-E3 does not establish prediction of actual policy invalidation.
  - The predictors are collinear with magnitude and partly repeated across seeds (§D), so the
    p-values should be read cautiously.
  - Secondary: no structural measure showed a detectably stronger association than update
    fraction; distributional drift was detectably weaker for B1 cost drift.
- **H-E4 (descriptive):** among early-stopped queries only (not all queries; not probabilistic
  calibration), DARTH's frozen recall predictor stays near-unbiased: |bias| ≤ 0.005 and MACE within +0.003 of S0, with the largest shift under 8% deletion.

**Descriptive pattern:**
- **DARTH:** as the index grows or loses vectors, it becomes slightly cheaper relative to the
  reference (D slope −0.5 pp per doubling) while losing contract passes slightly faster (E slope
  +0.7 pp per doubling). Both stay inside the frozen margins.
- **The state-local B1 reference:**
  - **Under insertion** it remains at ef 50/53, with the high-ef weight moving approximately from
    0.48 to 0.58 (seed 42; 0.48–0.66 across all seeds, vs frozen B1 at 0.538 / 0.596 / 0.637).
    B1 insertion-state comparisons are therefore close to a mechanical comparison with the
    frozen B1 policy.
  - **Under deletion,** 2% keeps ef 50/53 with weights 0.17–0.29, and 8% moves to ef 48/50 with a
    high-ef weight of approximately 0.048 (seed 42; 0.048–0.24 across seeds).
  - So deletion is where state-local recalibration meaningfully changes. Frozen B1 keeps its S0
    ef there and costs up to 5% more than the recalibrated reference, at slightly higher recall.

**Not established:**
- staleness beyond 8% churn, under physical deletion or graph repair, on other datasets, index
  parameters or k;
- behaviour under a genuinely independent OOD source;
- anything about router cost savings: DARTH's absolute S0 inefficiency is the C1 result and is
  not re-tested here.

A STABLE verdict means that, under these conditions, the frozen calibrations did not become
stale by the pre-declared margins. It is not a claim that adaptive ANN policies are immune to
staleness in general.

## H. Final evidence table
| Analysis | Result | Status | Correct interpretation |
|---|---|---|---|
| H-E1 B1 | D̄ −0.0005 to +0.0040 (insertion); +0.0157 (del 2%), +0.0524 [0.051, 0.054] (del 8%) | STABLE (all states, all seeds; STABLE-FAMILY ×2) | No cost staleness beyond ±10%. Detectable sub-threshold drift under deletion; insertion comparisons are near-mechanical (reference stays at ef 50/53). |
| H-E1 DARTH | D̄ −0.0021 to −0.0224 (e.g. id 8% −0.0210 [−0.027, −0.015]); raw R +0.084 to +0.112 (descriptive) | STABLE (all states, all seeds; STABLE-FAMILY ×2) | No cost staleness beyond ±10%. DARTH becomes slightly cheaper relative to the reference. Its C1 S0 inefficiency persists in raw R and is not re-tested. |
| H-E2 B1 | Ē −0.0075 to +0.0000 | STABLE (STABLE-FAMILY ×2) | No excess loss of S0 contract passes vs the reference. |
| H-E2 DARTH | Ē +0.0022 to +0.0326 (e.g. id 8% +0.0326 [0.018, 0.046]) | STABLE pooled (STABLE-FAMILY ×2); seed-level INCONCLUSIVE at id 8% (seeds 42, 44) and del 8% (seed 42); none STALE | Detectable, magnitude-related excess loss of S0 passes, below the ±5 pp margin. INCONCLUSIVE cells are not failures. Cohort-composition caveat applies. |
| H-E3 | 10/16 associated after Holm (B1 D 4/4, B1 E 2/4, DARTH D 4/4, DARTH E 0/4) | Secondary | Associations with sub-threshold drift only. Predictors collinear with magnitude and partly repeated across seeds. Does not establish prediction of policy invalidation. No measure detectably stronger than update fraction. |
| H-E4 | Stopped fraction 0.942–0.946; bias −0.0013 (S0) to ≤ +0.0046; MACE 0.0537 → ≤ 0.0569 | Secondary (descriptive) | Among early-stopped queries only, the frozen predictor's ID-recall error barely changes (largest at del 8%). Not calibration over all queries. |
| Trends | DARTH cost −0.53 pp/doubling [−0.71, −0.35]; newly-failing +0.66 pp/doubling [+0.40, +0.92]; B1 flat | Secondary | Small monotone DARTH drift. Mixed models on the boundary (seed variance ≈ 0); CIs from 24 correlated cells are likely optimistic. |

**Summary:** measurable, policy-specific drift occurred, but it remained below the
preregistered practical invalidation margins within the tested evolution regime.

## I. Interpretation across the project (evidence chain)
1. **Phase 4 / 4b:** the original adaptive-routing efficiency objective was not achieved. The
   frozen Phase-4b router failed its gate (−20.9 / −8.9 / +1.1% vs B1 at S* 0.90 / 0.95 /
   0.99), despite about 47% perfect-information headroom.
2. **Literature audit:** it motivated the shift toward policy validity under index evolution
   (runtime-adaptive actuators: Ada-ef, rejected for L2; DARTH, selected). It took place in the
   research discussion; the repository records only its consequences (docs/notes.md, Ada-ef and
   DARTH audits), not a literature-review document.
3. **Phase 5′:** the feature→oracle-effort relationship stayed stable (H1′ supported: every Δ CI lower bound
   ≥ −0.05, all 40 pooled cells STABLE) within the tested regime, although query-level effort reshuffled (per-query effort rank
   correlation 0.84 at 8%).
4. **C1:** the DARTH-style policy did not beat B1 on S0 computational efficiency (−11.0% saving,
   replicated at −11.5% / −10.8% for seeds 43 / 44), at higher recall (0.963 vs 0.948).
5. **Experiment E:** both S0-frozen policies stayed within the predefined cost (±10%) and
   contract (±5 pp) validity margins in every tested state.
6. **Measurable sub-threshold drift remained,** policy-specific and clearest for DARTH: slightly
   cheaper relative to the reference, with a growing excess loss of S0 contract passes. B1 drift
   appeared mainly under deletion.
7. **Structural change was associated with some of that drift,** but H-E3 does not establish
   prediction of actual policy invalidation, and its p-values carry dependence caveats.
8. **All of this is bounded to the exact tested regime (§J).** It is not a successful
   adaptive-routing result: neither adaptive policy outperformed B1 at S0 in this project.

## J. Limitations (retained)
- **Evolution:** 1–8% insertion churn (in-distribution and regionally concentrated "OOD") and
  2–8% lazy deletion only. No physical deletion or graph repair, and no evidence beyond the
  tested churn.
- **Configuration:** one dataset (SIFT1M, L2); one hnswlib/HNSW configuration (M16, efC200);
  k = 10.
- **Queries and seeds:** the 2,000-query Phase-2 test set; three index seeds.
- **Reference:** a B1-style state-local reference. For DARTH this is comparison with a freshly
  calibrated simple policy, not a recalibrated DARTH. Under insertion it stays close to frozen B1.
- **DARTH:** an independent re-implementation on hnswlib, with a documented intent-faithful
  feature-order deviation, and C1-failed at S0. H-E2 has a cohort-composition caveat.
- **H-E3:** predictors collinear with magnitude and partly seed-independent, so few independent
  predictor values. Query-bootstrap p-values should be read cautiously.
- **H-E4:** early-stopped queries only (selection at predicted ≥ 0.95); discrete recall gives
  MACE a floor.
- **Cost metric:** distance computations. Wall-clock times are not interpreted.
- **OOD:** no genuine independent out-of-distribution source. 8% "OOD" ≈ ID by construction.
- **Margins:** the ±10% and ±5 pp margins were fixed a priori; STABLE is relative to them.
- **Novelty:** any novelty claim requires separate literature verification.

