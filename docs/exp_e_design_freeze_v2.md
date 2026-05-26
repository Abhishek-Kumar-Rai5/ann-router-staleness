# Experiment E — Design Freeze v2: Canonical Implementation Specification

**Status: FROZEN — FINAL (user-approved 2026-10-05; appended to docs/design_doc.md as Addendum E1).**
- Becomes binding only when the user approves it. On approval it is appended verbatim to
  docs/design_doc.md as Addendum E1.
- Written before any Experiment E estimand existed. No B1 or DARTH outcome at any evolved state
  has been computed or inspected.
- Known beforehand, and disclosed: the Phase 5′ results (docs/phase5b_report.md).
- Margins and policies are as frozen by the user. Every other choice comes from the A1 /
  Phase-4b / C1 conventions or from S0 evidence, and the source is cited.
- **One genuine logical contradiction was found** (§6.3). It is resolved in §6.3
  (user-approved 2026-10-05).

## 1. Research question
When the ANN index evolves through insertions and lazy deletions, does a policy calibrated once
at S0 become *stale*, i.e. cost and contract quality drifting relative to a policy freshly
calibrated at the evolved state? And do the A1 structural index-change measures track that
staleness?

Two S0-calibrated policies are studied:
- the static fixed-effort B1;
- the runtime-adaptive DARTH (C1 failed; studied as a calibrated actuator, not as a winner).

No direction is presupposed.

## 2. Experimental units and states
- **Index seeds:** 42, 43, 44.
- **States per seed:** the existing A1/Phase-5′ states, unchanged (manifest
  data/phase5/sift/states/run_manifest.json):
  - **insertion family (8 states):** id_{10000, 20000, 40000, 80000} and
    ood_{10000, 20000, 40000, 80000};
  - **deletion family (2 states):** del_{20000, 80000}.
  - Total 30 (seed, state) cells, plus S0 per seed.
- **Magnitude:** m = update fraction = count / 1,000,000. Insertion m ∈ {0.01, 0.02, 0.04,
  0.08}; deletion m ∈ {0.02, 0.08}.
- **Evaluation unit:** query q ∈ the 2,000 Phase-2 test-split queries
  (splits/sift1m_query_split_seed20261001.csv, split == "test").

## 3. Frozen policies (per seed σ; no retuning, no retraining)
- **B1(σ):** the Phase-4b mean-recall R = 0.95 two-ef mix for seed σ (results/phase4b_train_311634ee3011_20261002T145649Z):

  | Seed | ef_lo / ef_hi | w_hi |
  |---|---|---|
  | 42 | 50 / 53 | 0.538 |
  | 43 | 50 / 53 | 0.596 |
  | 44 | 50 / 53 | 0.637 |

  - Per-query ef = ef_hi if splitmix64(fnv1a64("sift_query:<query_id>") XOR 20261006)/2⁶⁴ < w_hi,
    else ef_lo (router4b_lib.assign_mix).
  - **At every state each query keeps its S0-assigned ef.**
- **DARTH(σ):** the frozen policy files (no other DARTH policy may be used):

  | Seed | Policy file |
  |---|---|
  | 42 | derived/exp_c/darth_policy_s0.json |
  | 43 | derived/exp_e/darth_policy_s0_seed43.json |
  | 44 | derived/exp_e/darth_policy_s0_seed44.json |

  - Each policy is run unchanged on seed σ's evolved index with apps/exp_c_darth `eval`.
- **Cost:** total distance computations per query (CountingL2Space, upper layers included), the
  Phase-2/4b/C1 accounting.
  - B1 and reference costs and recalls are read from each state's existing oracle_curves.csv at
    the assigned ef (identical accounting, verified at S0 in C1).
  - DARTH cost is measured live.
- **Quality:** tie-aware recall@10 against the state's own ground truth. Contract success ⇔
  tie-aware recall@10 = 1 (10/10).

## 4. Evaluation split
- All hypotheses are evaluated on the 2,000 test queries only.
- The 8,000 training queries are used only to calibrate the state-local reference (§5).
- No estimand, test or descriptive result is computed on training queries.

## 5. State-local reference REF(σ, s)
- **Per (σ, s)**, independently, seeds never pooled:
  `router4b_lib.two_ef_mix(level_by_ef, grid, 0.95)`.
  - `grid` = the canonical 121-ef grid.
  - `level_by_ef` = the mean over the 8,000 training queries of tie-aware recall@10 at each grid
    ef, from (σ, s)'s oracle_curves.csv.
- **Assignment:** the same hash as B1 (query_set_id "sift_query", seed 20261006).
- **Test-query cost and recall** come from the test rows of the same oracle_curves.csv.
- **Integrity requirement:** REF(σ, S0) must equal B1(σ) exactly (ef_lo, ef_hi, w_hi to 1e-12
  and per-query ef). Otherwise STOP.
- **For DARTH, REF is a freshly calibrated simple B1-style policy, not a state-local DARTH.**
- If two_ef_mix returns None (no grid ef reaches 0.95) → failed cell (§16).

## 6. H-E1 — cost staleness

### 6.1 Cost estimand: ratio of means (option A), preferred cost-ratio construction (the primary H-E1 estimand is D, §6.2)
For policy P ∈ {B1, DARTH}, seed σ and state s:

  R(P, σ, s) = ( C̄_P(σ, s) − C̄_REF(σ, s) ) / C̄_REF(σ, s)

with C̄ = the mean cost over the evaluation queries (in a bootstrap replicate, over the
resampled queries).

**Why:**
- it is the Phase-4b/C1 "saving vs B1" definition (sign flipped);
- it is stable when per-query costs are small (B1 minimum 314);
- it is a smooth function of paired means, so the paired percentile bootstrap applies.

**Secondary (descriptive, no decision):** the mean over queries of (C_P,q − C_REF,q)/C_REF,q,
with its CI.

### 6.2 Decision quantity (primary H-E1 estimand): proportional S0-anchored drift D

  D(P, σ, s) = [ C̄_P(σ, s) / C̄_REF(σ, s) ] / [ C̄_P(σ, S0) / C̄_REF(σ, S0) ] − 1

- **C̄** = the mean distance computations per query over the 2,000 test queries. In each
  bootstrap replicate (§10), every C̄ (state and S0 terms alike) is recomputed over the same
  resampled query ids.
- **S0 terms** come from seed σ's S0 rows: frozen B1/REF from the S0 oracle curves, DARTH from
  that seed's S0 evaluation.
- **D = 0 at S0 for both policies**, by construction.
- **For B1,** C̄_B1(σ, S0) = C̄_REF(σ, S0) exactly (§5 integrity requirement). Its S0 reference
  ratio is therefore 1, and D(B1, σ, s) equals the ordinary relative regret R(B1, σ, s) of §6.1.
- **For DARTH,** D isolates the proportional change from DARTH's own S0 cost relationship to
  the reference. It does not re-test DARTH's known C1 S0 inefficiency.
- **Descriptive only (no decision):** raw R(P, σ, s) (§6.1), the additive
  ΔR(P, σ, s) = R(P, σ, s) − R(P, σ, S0), and the per-query relative cost of §6.1.

### 6.3 Logical contradiction found, and its resolution (DECIDED: user-approved 2026-10-05)
**The contradiction:**
- The raw R measures staleness only when the frozen policy equals the reference at S0. That
  holds for B1.
- It does not hold for DARTH: at S0, DARTH already costs +11.0% / +11.5% / +10.8% vs the S0
  reference (= B1), per C1 and Stage 1.
- With a ±10 pp margin, DARTH's raw-R verdict would be decided by its known S0 inefficiency (C1),
  not by any change caused by index evolution. That contradicts the research question
  ("calibrated at S0 → stale later").

**Design decision (approved by the user on 2026-10-05):**
- **Raw R is retained as descriptive only.** DARTH has a pre-existing S0 cost disadvantage vs
  the reference, so raw R would mix that known C1 result with evolution-induced change.
- **The additive ΔR = R(s) − R(S0) is rejected.** It scales DARTH's evolution drift by its S0
  relative-cost ratio (≈ 1.11): a proportional drift p appears as ≈ 1.11·p. So the ±10% margin
  would not carry the same proportional meaning for DARTH as for B1.
- **The proportional D (§6.2) is adopted as the primary H-E1 estimand.** It is 0 at S0 for both
  policies, equals R for B1, and gives the same proportional interpretation of the ±10% margin
  for both policies.
- **Basis of the decision:** only the conceptual estimand and already-established S0 evidence
  (C1 and Stage-1 S0 cost ratios 1.110 / 1.115 / 1.108; B1 ≡ REF at S0). No Experiment E
  evolved-state result existed or was consulted. Margins, policies, references, states, queries
  and bootstrap are unchanged.

### 6.4 Decision rule (per pooled state, three-valued; A1.7 style)
- **Pooled estimate:** D̄(P, s) = (1/3) Σ_σ D(P, σ, s), equal weights.
- **95% CI:** percentile, query bootstrap (§10).
- **Classification of D̄(P, s), with margin M₁ = 0.10:**
  - **STABLE:** the 95% CI lies entirely within [−0.10, +0.10];
  - **STALE-COSTLIER:** CI lower bound > +0.10;
  - **STALE-CHEAPER:** CI upper bound < −0.10;
  - **INCONCLUSIVE:** otherwise.
- **Per seed:** D(P, σ, s) is classified the same way and always reported. Any seed-specific
  STALE-COSTLIER / STALE-CHEAPER is listed explicitly next to the pooled verdict and is never
  averaged away.
- **Quality context (required):** mean tie-aware recall@10 is reported for every (P, σ, s),
  including REF and S0, and for the seed means. Cost drift is then interpreted alongside quality
  drift: D can partly reflect a change in recall rather than in efficiency.
- **Family verdict**, per policy × family (insertion, deletion separately):
  - STABLE-FAMILY if every pooled state in the family is STABLE;
  - STALE-FAMILY if any is STALE;
  - INCONCLUSIVE-FAMILY otherwise.
- **Trend across magnitude:** secondary, not part of the verdict (§11).
- **S0 power context** (computed before E, S0 only): the paired per-query SD of DARTH − B1 cost
  is 494 (SE of mean ≈ 1.05% of cost per seed). Seed-to-seed B1 cost differences have SD ≈ 55
  (≈ 0.12% relative SE). Expected CI half-widths are ≈ 1–2 pp, well inside the ±10 pp margin.

## 7. H-E2 — contract staleness (newly failing)

### 7.1 Cohorts and transitions
- **Pass:** pass_P(σ, s, q) = 1 iff tie-aware recall@10 = 1 (implemented as ≥ 1 − 1e-9).
- **S0 cohorts:**
  - K_P(σ) = { q : pass_P(σ, S0, q) = 1 } for P = B1 and DARTH;
  - **K_REF(σ) = K_B1(σ).** The reference at S0 is B1(σ) exactly (§5 integrity requirement), so
    the reference's S0 cohort is B1's S0 cohort. This applies whichever frozen policy it is
    compared with.
- **Transition categories per (P, σ, s, q)** (S0 → s), for P ∈ {B1, DARTH, REF}:
  stable-pass, newly-failing, stable-fail, improved.
  - All four category counts are reported for every cell.

### 7.2 Estimand: conditional rate, denominator option B
  NF(P, σ, s) = #{ q ∈ K_P(σ) : pass_P(σ, s, q) = 0 } / |K_P(σ)|

The same formula applies to REF with K_REF(σ). Then:

  E(P, σ, s) = NF(P, σ, s) − NF(REF, σ, s)   (absolute; fraction units, 0.05 = 5 pp)

**Why denominator B:**
- The frozen policies have different S0 pass rates (B1 ≈ 0.69, DARTH ≈ 0.73).
- With denominator A (all 2,000 queries), the excess would mechanically mix in the base pass
  rate.
- B is the probability that an S0 success is lost, which is the staleness quantity.
- At S0, E = 0 by construction, so H-E2 needs no anchoring.

**Disclosed caveat (DARTH):** DARTH's cohort and the reference cohort differ in composition
(C1: 192 queries pass under DARTH but fail under B1). E(DARTH) therefore includes a cohort-mix
component. It is reported, not corrected.

### 7.3 Decision rule
Identical to §6.4, applied to Ē(P, s) = mean over σ of E(P, σ, s), with:
- margin M₂ = 0.05: STABLE if CI ⊂ [−0.05, +0.05]; STALE if CI lower bound > +0.05
  (STALE-WORSE) or CI upper bound < −0.05 (STALE-BETTER); INCONCLUSIVE otherwise;
- 95% percentile CI;
- per-seed classification reported;
- family verdicts per policy × family;
- trend secondary (§11).

**S0 power context:**
- the B1 seed-to-seed newly-failing rate is 1.4–2.0% of ≈ 1,380 S0 passers;
- the binomial SE per seed is ≈ 0.35 pp;
- expected CI half-widths are ≲ 1 pp, inside ±5 pp.

## 8. H-E3 — structural predictors of staleness
- **Predictors:** exactly the A1.5 (§8) measures, taken from the existing Phase-5′ artifact
  results/phase5b_analysis_20261004T230023Z. Nothing is recomputed, and no new measure is added.
  1. update fraction m (state level);
  2. neighbourhood-overlap decay = 1 − overlap_top10_with_S0/10 (per query; column
     overlap_top10_with_S0 of rows_query_state_seed.csv.gz);
  3. local-density drift = (knn_dist_s − knn_dist_S0)/knn_dist_S0 (per query; column
     local_density_drift). This is the A1 "local-density drift". Centroid drift is not an A1
     structural measure and is not used.
  4. distributional drift D (state level; analysis.json → distributional_drift_D).
- **Representation at the (σ, s) level:** measures 2 and 3 are averaged over the 2,000 test
  queries of (σ, s) (in a bootstrap replicate, over the resampled queries). Measures 1 and 4 are
  constants per state.
- **Outcomes:** D(P, σ, s) (§6.2, the primary H-E1 quantity) and E(P, σ, s) (§7.2), for
  P ∈ {B1, DARTH}. Raw R and additive ΔR are descriptive only and are not H-E3 outcomes.
- **Unit and statistic (primary, insertion family only):** Spearman ρ between predictor and
  outcome over the 24 (σ, s) insertion cells (3 seeds × 8 states; ties by average ranks).
  - **CI:** 95% percentile, query bootstrap (§10); every replicate recomputes outcomes,
    query-level predictors, then ρ.
  - **Two-sided bootstrap p:** p = min(1, 2 · min(Pr*(ρ* ≤ 0), Pr*(ρ* ≥ 0))), with a floor of
    1/2000.
- **Multiple comparisons:** Holm–Bonferroni over the family of 16 tests (2 policies × 2
  outcomes × 4 predictors), at familywise α = 0.05.
  - **"Associated"** ⇔ Holm-adjusted p < 0.05.
- **Deletion family:** descriptive only, because n = 6 cells with only 2 values of m. Spearman ρ
  and CI are reported; there is no test.
- **Secondary descriptive (A1 H4′ spirit; no decision):**
  - |ρ(structural)| − |ρ(update fraction)| for predictors 2–4, with bootstrap CI;
  - a per-query analysis: Spearman over all insertion (σ, s, q) test rows of overlap decay and
    density drift vs the per-query cost difference C_P − C_REF and vs the newly-failing
    indicator (S0-pass cohort).

## 9. H-E4 — DARTH predicted vs realised recall (descriptive)
- **Per (σ, s):**
  - Q = queries with stopped_early = 1;
  - stopped fraction f = |Q|/2000 (always reported with the metrics);
  - bias = mean_{q∈Q}(pred_q − realised_id_q);
  - MACE = mean_{q∈Q} |pred_q − realised_id_q|.
  - pred_q = last_predicted, the clipped regression output at the stop decision. realised_id_q
    = recall@10 by id.
- **Per-query MACE, no binning:** the predictor is a point regression of id recall, not a
  probability, so a reliability diagram is not the estimand.
- **Pooled:** the mean over σ, with 95% bootstrap CI (§10; Q recomputed per replicate).
- **Change from S0:** bias(σ, s) − bias(σ, S0) and MACE(σ, s) − MACE(σ, S0) are reported, with
  CIs. S0 values come from the Stage-1/C1 evaluations.
- **Secondary (disclosed):** the same metrics against realised tie-aware recall.
- **Required disclosures:**
  - it is conditional on the stopped subset (selection at pred ≥ 0.95), so it is not
    probabilistic calibration;
  - realised recall is discrete (tenths), so MACE has an irreducible floor;
  - cap-run queries are excluded (their last_predicted is a stale check value).
- **No margin and no decision rule** (none was frozen). If |Q| < 30 for a cell, that cell's
  metrics are reported but flagged "small-n".

## 10. Bootstrap specification
- **Resamples:** B = 2,000; numpy.random.default_rng(20261002).
- **Index matrix:** generated once: an integer matrix I of shape (2000, 2000) with entries
  uniform on the 2,000 test-query positions (sorted query_id order).
- **Sharing:** the same I is reused for every seed, state, policy, reference, hypothesis and
  outcome (A1 convention: a query and all its seeds/states/policies move together, so pairing
  is preserved).
- **Unit resampled:** query id; there is no other clustering level. Seeds are fixed (not
  resampled), and their variability enters through the per-seed reporting and §11.
- **Interval:** 95% percentile [2.5%, 97.5%].
- **Ratios and conditional rates** are recomputed from the resampled numerators and
  denominators in each replicate.

## 11. Mixed-effects trend specification (secondary; not part of the H-E1/H-E2 verdicts)
- **Package:** statsmodels (`smf.mixedlm`), version pinned in python/requirements.txt at install.
  Fit with REML = True and default optimiser settings.
- **Outcomes:** D(P, σ, s) (§6.2, the primary H-E1 quantity; raw R and additive ΔR are not
  modelled) and E(P, σ, s), separately for P = B1 and DARTH. That is 4 models,
  each with point estimates on the full test set.
- **Insertion model** (24 cells; S0 excluded):
  `y ~ L + T + L:T`, groups = seed, random intercept only (`re_formula="1"`).
  - L = log2(m / 0.01) ∈ {0, 1, 2, 3}.
  - T = 1 for OOD, 0 for ID.
- **Inferential trend quantity:** the coefficient of L (the ID slope).
  - Reported with its 95% Wald CI.
  - **"Changes with magnitude"** ⇔ the CI excludes 0.
  - The T and L:T terms are descriptive only (ID-vs-OOD is not an E hypothesis, and 8% OOD is
    ≈ ID by construction).
- **Deletion:** no model (2 magnitudes). Report the pooled difference
  y(del_80000) − y(del_20000) with its bootstrap CI (descriptive).
- **Failure handling:**
  - a boundary variance estimate (0) is reported as fitted;
  - a non-converged or failed fit is reported as "fit failed" with the error;
  - no alternative model is substituted.
- **Robustness (descriptive):** the bootstrap CI of the slope of the pooled-over-seed estimate
  on L (ordinary least squares on 4 points per trajectory).

## 12. Multiple-comparison policy (frozen)
- **H-E1 and H-E2 per-state three-valued classifications:** no α adjustment. The family verdict
  is an intersection rule, which is conservative. All cells are reported (A1.7 convention).
- **H-E3:** Holm–Bonferroni over the 16 primary insertion tests (§8).
- **Trend (§11) and H-E4:** secondary/descriptive; unadjusted 95% CIs, labelled secondary.

## 13. Treatment of 8% OOD states
- **H-E1, H-E2, H-E3:** ood_80000 is an ordinary insertion state. It counts in the per-state
  classifications, the family verdicts, the trend model and the H-E3 correlations. It is a valid
  index-change state.
- **ID vs OOD:** any contrast (T and L:T in §11, or a reported ID/OOD difference) is descriptive
  only. At 8% the mandatory A1 wording is attached: "At 8% insertion magnitude, the available
  pool forces the regional-OOD construction to overlap approximately 91% with the matched ID
  set, so this level does not provide a clean ID-vs-OOD comparison."

## 14. Confidence intervals and significance rules (summary)
- **All CIs:** 95%.
- **H-E1/H-E2:** query-bootstrap percentile CIs with the three-valued margin rules (§6.4,
  §7.3).
- **Trend:** MixedLM Wald CI excluding 0 (secondary).
- **H-E3:** Holm-adjusted bootstrap p < 0.05.
- **H-E4:** CIs only, no test.

## 15. Seed handling
- Each seed's frozen policies run only on that seed's states. The reference is calibrated per
  (σ, s).
- Pooled estimates are equal-weight means over σ = 42, 43, 44.
- Per-seed results are always reported, and a seed-specific STALE is listed explicitly.
- Seeds are a random intercept only in §11.
- DARTH seeds 43/44 have caps 142/135 and intervals from their own S0 traces (Option A, Stage 1).

## 16. Missing or failed cells
- A (σ, s) cell fails if any of these hold:
  - a DARTH run errors;
  - a run returns ≠ 2,000 rows;
  - any recall lies outside [0, 1];
  - any deleted or out-of-range label is returned;
  - two_ef_mix returns None;
  - the §5 integrity check fails;
  - any oracle-curve test row is missing.
- **Any failed cell → STOP before analysis and report.** No imputation, no partial analysis, no
  dropping of states or queries.

## 17. Anti-p-hacking rules
- **Nothing is retuned, retrained or changed:**
  - no retuning of B1 and no retraining of DARTH;
  - no change of caps, intervals, target, margins, states, split, cohorts, predictors, models
    or bootstrap;
  - no removal of outlier queries or difficult states;
  - no metric switching;
  - no choosing between policies or estimands after seeing results.
- **The analysis script is committed and its sha256 is recorded before** any evolved-state
  DARTH run.
- **One run only:** the analysis is run once. A re-run is allowed only for a reproducibility
  check (identical output required).
- **Report findings as obtained:** negative, null or inconclusive results are reported as such.
  B1 and DARTH, and insertion and deletion, are reported separately.

## 18. Execution order
1. User approves this document. Append it to design_doc.md as Addendum E1, with any §6.3
   decision recorded.
2. Pin statsmodels in python/requirements.txt.
3. Implement (no outcomes yet):
   - python/exp_e_b1.py: frozen B1 + REF per (σ, s) from the oracle curves, with the §5
     integrity check;
   - python/exp_e_run.py: per-(σ, s) DARTH eval configs and runs;
   - python/exp_e_analysis.py: §6–§13.
   Then record the analysis-script sha256 in notes.
4. **Pre-run checks** (no evolved-state outcomes):
   - REF(σ, S0) == B1(σ) for σ = 42/43/44;
   - the DARTH eval via the E pipeline at S0 reproduces the Stage-1/C1 S0 rows exactly
     (non-timing columns);
   - existing gtests and pytest pass.
5. Run DARTH(σ) on all 30 (σ, s) cells: single-threaded eval, the state's index and ground
   truth, and the frozen policy hash verified.
6. Compute B1 and REF per-query rows for all cells from the oracle curves.
7. Check §16 completeness. If anything fails, STOP.
8. Run python/exp_e_analysis.py once; write results/exp_e_analysis_<ts>/.
9. Determinism check: re-run the analysis; the output must be identical.
10. Write the report (docs/exp_e_report.md) with the A1.10 limitations and the 8% wording.

## 19. Implementation checklist for Claude Code
- [ ] **B1 assignment** via router4b_lib.assign_mix(mix, query_ids, "sift_query", 20261006),
  using the seed's frozen mix from the Phase-4b training record.
- [ ] **REF(σ, s)** via router4b_lib.two_ef_mix on mean training tie-aware recall over the
  121-ef grid, and the same assign_mix.
- [ ] **Integrity check** REF(σ, S0) ≡ B1(σ).
- [ ] **Costs and recalls** of B1 and REF read from oracle_curves.csv (columns
  distance_computations, recall_tie_aware) of (σ, s) at the assigned ef, test rows only.
- [ ] **DARTH eval config per (σ, s):** index_path = the state index; gt_prefix = the state's
  ground-truth cache path from that state's oracle-run metadata; queries/split unchanged;
  ef_cap from the frozen policy; policy_path = the frozen policy file. The deletion path is
  automatic (DeletedCount > 0 → general hnswlib rule).
- [ ] **DARTH columns used:** distance_computations, recall_tie_aware, recall_id, stopped_early,
  last_predicted (and the overhead columns, reported descriptively).
- [ ] **Bootstrap matrix I** built once (§10) and shared everywhere.
- [ ] **H-E1:** primary D (§6.2) with the S0 terms recomputed per replicate; seed-averaged D̄
  classification (STABLE / STALE-COSTLIER / STALE-CHEAPER / INCONCLUSIVE), per-seed,
  family verdict; mean tie-aware recall@10 per (P, σ, s) incl. REF and S0; descriptive only:
  raw R, additive ΔR, per-query relative cost.
- [ ] **H-E2:** cohorts K_P (K_REF = K_B1), the four transition counts, NF, E, pooled
  classification, per-seed, family verdict.
- [ ] **H-E3:** the 16 Spearman tests with bootstrap p and Holm; deletion descriptive;
  secondary descriptives.
- [ ] **H-E4:** f, bias, MACE, changes from S0, tie-aware secondary, small-n flags, disclosures.
- [ ] **§11 MixedLM:** 4 models plus deletion differences and the robustness slope.
- [ ] **Row-level output:** one row per (policy ∈ {B1, DARTH, REF}, σ, state incl. S0, q), with
  cost, recalls, pass, transition and assigned ef / DARTH fields.
- [ ] **Metadata:** config, script hashes, policy/model hashes, submodule commits, seeds.

## Final Freeze Status
| Item | Status |
|---|---|
| 1 H-E1 decision rule (three-valued, 95% percentile, per pooled state, family intersection, per-seed reported) | FROZEN |
| 1a H-E1 primary estimand = S0-anchored proportional D (§6.2/§6.3); raw R and additive ΔR descriptive only | FROZEN (user-approved 2026-10-05) |
| 2 H-E2 decision rule | FROZEN |
| 3 H-E1 cost definition (ratio of means primary; per-query relative secondary) | FROZEN |
| 4 H-E2 denominator (S0-pass cohort of the same policy) | FROZEN |
| 5 Reference S0 cohort = frozen-B1 S0 cohort (same for both comparisons) | FROZEN |
| 6 8% OOD states (ordinary insertion states; ID-vs-OOD descriptive) | FROZEN |
| 7 H-E4 MACE (per-query) + stopped fraction | FROZEN |
| 8 Mixed-effects specification | FROZEN |
| 9 H-E3 predictors / aggregation / statistic / unit / multiplicity | FROZEN |
| 10 Bootstrap | FROZEN |
| 11–19 remaining sections | FROZEN |
