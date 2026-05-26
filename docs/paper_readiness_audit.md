# Paper readiness audit (read-only; 2026-10-05)

Sources: docs/design_doc.md (incl. Addenda A1, C1, E1), docs/notes.md, CLAUDE.md,
docs/phase4b_final_report.md, docs/methodology_audit.md, docs/phase4_methodology_validation.md,
docs/phase5b_report.md, docs/exp_c_report.md, docs/exp_e_report.md,
docs/exp_e_design_freeze_v2.md, THIRD_PARTY.md, derived/exp_c/README.md.

Every number below is copied from those documents or their cited artifacts. Nothing was
recomputed, rerun or modified. Where a fact is missing from the repository, this audit says so.

---

## 1. Final research question
**On SIFT1M / hnswlib-HNSW (L2, M16, efC200, k = 10), do search-effort policies calibrated once
on a static index (S0) remain valid as the index evolves through bounded insertion (1–8%) and
lazy deletion (2–8%)?** Validity is measured against a freshly calibrated state-local reference,
separately for computational cost and for the per-query recall contract. Two further questions:
do the cheap query features that predict required effort keep their predictive relationship,
and do structural index-change measures track any drift?

The question exists because the preceding prerequisite failed: neither a static query-feature
router (Phase 4/4b) nor a runtime-adaptive early-termination policy (C1, DARTH re-implementation)
beat a calibrated fixed-effort baseline (B1) at S0. The completed project therefore supports a
**policy-validity / calibration-staleness** question, not an adaptive-efficiency question.

## 2. Final project thesis
- **The project began by testing whether per-query adaptive search effort can reduce ANN search
  cost, and whether that advantage survives index evolution.**
  - On SIFT1M/HNSW the adaptive-efficiency premise was not met at S0, under pre-registered gates,
    three index seeds and two query sets.
  - The static feature router cost 9–21% more than the calibrated fixed baseline at success
    targets 0.90/0.95, and was at parity at 0.99 (Phase 4b).
  - The re-implemented runtime-adaptive DARTH policy cost 10.8–11.5% more than B1 at mean recall
    0.95 (C1, three seeds).
  - These are scoped negative results, not proofs that adaptivity cannot help: Phase 4b
    documented about 47% perfect-information headroom that the cheap features could not exploit.
- **Because the robustness question presupposes an S0 advantage, the project was reframed
  (Addenda A1, E1).**
  - The reframed question asks whether the ingredients and calibrations of such policies stay
    valid under index evolution.
  - Phase 5′ showed that the population-level relationship between three cheap features and
    oracle-required effort stays within 0.05 of S0 across 30 evolved states, although individual
    queries' required effort reshuffles substantially (per-query effort rank correlation 0.84 at
    8% churn).
  - Experiment E showed that both an S0-frozen fixed-effort policy (B1) and an S0-frozen
    runtime-adaptive policy (DARTH) stayed within pre-registered practical validity margins
    (±10% cost drift, ±5 pp excess contract loss) in every tested state.
  - Measurable, policy-specific drift nevertheless occurred below those margins:
    - DARTH became about 2% relatively cheaper while losing S0 contract passes up to about 3 pp
      faster;
    - B1 overpaid up to 5.2% after 8% deletion.
  - Structural measures were associated with some of this sub-threshold drift, but none was
    shown to track it better than the plain update fraction, and no actual invalidation occurred
    to be predicted.
- **Why the negative and null results matter.**
  - They are pre-registered, multi-seed, precisely estimated (narrow CIs relative to the margins)
    and reproducible.
  - They bound when per-query adaptivity pays off on this workload.
  - They show that, within modest churn, recalibrating fixed-effort or DARTH-style policies is
    not urgent by these margins.
  - They show that staleness must be decomposed: cost drift and contract drift moved in opposite
    directions for DARTH.
  - Every conclusion is bounded to the tested regime.

## 3. Contributions (novelty NOT claimed; see §14)
- **Methodological:**
  - a pre-registered, phase-gated protocol for evaluating search-effort policies, with frozen
    gates, query-clustered bootstrap and three-valued margin rules (STABLE / STALE /
    INCONCLUSIVE);
  - an S0-anchored proportional staleness estimand
    D = [C_P(s)/C_REF(s)] / [C_P(S0)/C_REF(S0)] − 1, which separates a policy's S0 inefficiency
    from evolution-induced drift;
  - a contract-staleness estimand (excess newly-failing among S0-pass cohorts);
  - a state-local recalibrated reference as the counterfactual.
- **Empirical:**
  - scoped negative results for S0 adaptive efficiency (Phase 4/4b static router; C1 DARTH
    re-implementation);
  - stability of the feature→effort signal under bounded churn (Phase 5′);
  - sub-threshold, policy-specific calibration drift of frozen B1 and DARTH (E);
  - a cost-vs-contract decomposition;
  - a descriptive account of when recalibration changes the reference (deletion, not insertion).
- **Experimental and reproducibility:**
  - byte-reproducible index builds and nested evolved states (V1–V11);
  - independent ground-truth verification;
  - an independent DARTH re-implementation on hnswlib's public stop-condition API, verified
    byte-identical to searchKnn with stopping disabled, with a documented feature-order
    deviation;
  - frozen model/policy hashes;
  - deterministic analyses.

## 4. Complete evidence chain
(Chronology note: Phase 5′ states were produced before C1, but its analysis ran after C1.)

| Phase | Question | Method | Result | Consequence | Supported claim | Unsupported claim |
|---|---|---|---|---|---|---|
| 1 | Is the S0 index/GT infrastructure correct? | SIFT1M, hnswlib M16/efC200, brute-force GT vs TEXMEX | Checkpoint MET: GT verified (0 rows differ beyond ties); recall@10 0.71 @ef10 → 0.997 @256; deterministic | Trusted base for oracle labels | Correct, deterministic S0 infrastructure | Anything about latency (descriptive only) |
| 2 | Is required effort heterogeneous across queries? | Per-query oracle: smallest ef with tie-aware recall@10 = 1 on a 121-ef grid | Checkpoint MET: train median 29, p90 117, max 2794; at the oracle budget (~1,137 dc) the best fixed ef gets recall 0.955 | Premise for routing exists | Strong per-query effort heterogeneity on SIFT1M | That it is exploitable by cheap features |
| 3 | Do cheap features predict effort? | knn_dist, centroid_dist, score_concentration (+LID) from an ef = 10 probe | Checkpoint MET: Spearman +0.459 / −0.445 / +0.363 (LID +0.558) | Router can be attempted | Moderate rank signal (\|ρ\| ≈ 0.36–0.56) | That ρ ≈ 0.4 suffices for profitable routing |
| 4 | Does a 3-tier router beat the best fixed ef? | Pre-registered DT/LR tier router, test once | Checkpoint NOT MET: router 2,480 dc vs matched fixed ef 53 1,078 dc at recall 0.950 | Methodology audit found the gate unattainable by construction at R ≈ 0.95 → redesign | This router fails at S0 | That routing cannot work (design flaw identified) |
| 4b | Does a redesigned per-candidate success router beat B1? | Doubling ladder, per-candidate models, τ calibration, 3 seeds, fresh confirmation set | Gate FAIL at S* 0.90 / 0.95 / 0.99: saving −20.9% / −8.9% / +1.1%; perfect-info headroom ~47% | Robustness question blocked (§21) → reframe | Frozen cheap-feature routers do not beat B1 on SIFT1M | That adaptive allocation cannot help (headroom exists) |
| Literature audit | Which runtime-adaptive actuator fits SIFT1M/L2, and what gap motivates the reframing? | Paper/code audits of Ada-ef and DARTH | Ada-ef rejected (estimator not defined for L2); DARTH selected. **The broader literature audit motivating the reframing is not documented in the repository; only its consequences are (notes, Addendum C1).** | C1 with DARTH | DARTH supports L2; Ada-ef's published estimator does not | Any claim about the wider literature gap (undocumented here) |
| C1 | Does a runtime-adaptive policy beat B1 at S0? | Independent DARTH re-implementation (hnswlib stop-condition API), LightGBM, frozen intervals | C1 FAIL: −11.0% saving [−13.0, −8.9], recall 0.963 vs 0.948; replicated −11.5% / −10.8% (seeds 43/44) | No credible "advantaged" actuator → E reframed to calibration staleness | This DARTH-style policy is costlier than B1 at R = 0.95 on SIFT1M | That DARTH (official) or adaptive search fails generally; any quality-matched comparison |
| 5′ | Does the feature→effort relationship survive evolution? | 30 nested evolved states (1–8% ID/regional insertion, 2–8% lazy deletion), V1–V11, Δ\|ρ\| with ±0.05 rule | H1′ SUPPORTED: all 40 pooled and 120 per-seed cells STABLE; tightest centroid_dist at ood 4% (Δ −0.037, CI lb −0.045); per-query effort rank corr 0.84 at 8% | Signal-level premise holds; per-query reshuffling motivates calibration-level E | Population-level signal stable within tested churn | Per-query calibration stability; router validity |
| E | Do S0-frozen policies stay valid as the index evolves? | Frozen B1 + DARTH (3 seeds), state-local B1 reference, D (±10%), excess newly-failing (±5 pp), H-E3/H-E4/trends | H-E1, H-E2 STABLE everywhere; H-E3 10/16 associated (secondary); H-E4 near-unbiased among early-stopped queries; DARTH trends | Final evidence | No practical staleness under the margins within the tested regime; measurable sub-threshold drift | "No drift"; generality beyond regime; structural prediction of invalidation |

## 5. Final hypothesis / gate status
| Hypothesis / gate | Experiment | Result | Status | Exact evidence | Interpretation |
|---|---|---|---|---|---|
| Phase 1–3 checkpoints | Phases 1–3 | MET | SUPPORTED | GT verified; oracle checks 6/6; feature criteria met (\|ρ\| ≥ 0.2, CI excludes 0) | Infrastructure and signal valid at S0 |
| Phase 4 router gate | Phase 4 | Criteria (i), (ii) fail, (iii) passes | NOT SUPPORTED | 2,480 vs 1,078 dc at recall 0.950 | Tier router dominated by matched fixed ef |
| Phase 4b gate (all three S*) | Phase 4b | FAIL at 0.90 / 0.95 / 0.99 | NOT SUPPORTED | −20.9% / −8.9% / +1.1% vs B1 | No S0 advantage for frozen routers |
| C1 gate | C1 | G1–G3 fail, G4 passes | NOT SUPPORTED | Saving −11.0% [−13.0, −8.9]; Wilcoxon p 8.5e-5 favouring B1 | DARTH re-implementation costlier than B1 |
| H1′ (signal stability, X = 0.05) | Phase 5′ | All cells STABLE | SUPPORTED | 40 pooled + 120 per-seed STABLE | Feature→effort ρ stable within tested churn |
| H2′ (ID vs OOD, 1–4%) | Phase 5′ | centroid_dist differs (e.g. −0.031 at 4%); others CI ∋ 0 | DESCRIPTIVE ONLY (secondary) | Paired CIs | Regional insertion affects the global feature more |
| H3′ (centroid most stable, expectation) | Phase 5′ | Expectation reversed | NOT SUPPORTED (secondary) | centroid − knn −0.014 / −0.016 at id/ood 8% | Global feature least stable, within margin |
| H-E1 (cost staleness, D, ±10%) | E | 20/20 pooled, 60/60 per-seed STABLE | STABLE | B1 D̄ −0.0005…+0.0524; DARTH D̄ −0.0021…−0.0224 | No practical cost staleness; detectable sub-threshold drift |
| H-E2 (excess newly failing, ±5 pp) | E | All pooled STABLE; 3 DARTH per-seed INCONCLUSIVE | STABLE (seed-level INCONCLUSIVE ×3) | B1 Ē −0.0075…0; DARTH Ē +0.0022…+0.0326 | No practical contract staleness; DARTH sub-threshold excess |
| H-E3 (structural association) | E | 10/16 associated after Holm | SUPPORTED for 10/16 associations; NOT SUPPORTED for 6/16 (secondary) | B1 D 4/4, B1 E 2/4 (p_Holm 0.048), DARTH D 4/4, DARTH E 0/4 | Associations with sub-threshold drift only; dependence caveats |
| H-E4 (predicted vs realised ID recall) | E | Small changes | DESCRIPTIVE ONLY | Bias −0.0013 → ≤ +0.0046; MACE 0.0537 → ≤ 0.0569; early-stopped only | Predictor error barely moves among early-stopped queries |
| Trends | E | DARTH D −0.53 pp/doubling, E +0.66 pp/doubling; B1 flat | DESCRIPTIVE ONLY | Mixed-model Wald CIs; boundary warnings | Small monotone DARTH drift |

## 6. Final results inventory
| Finding | Value | Source |
|---|---|---|
| Effort heterogeneity (train oracle ef) | median 29, p90 117, max 2794 | notes Phase 2 |
| Feature–effort Spearman at S0 | knn +0.459, centroid −0.445, conc. +0.363, LID +0.558 | notes Phase 3 |
| Phase 4 router vs matched fixed | 2,480 vs 1,078 dc at recall 0.950 | notes Phase 4 |
| Phase 4b saving vs B1 | −20.9% [−23.3, −18.5] / −8.9% [−11.1, −6.8] / +1.1% [−0.7, +2.9] | phase4b_final_report §4–5 |
| Phase 4b perfect-info headroom (S* 0.95) | ~47% (1,397 vs 2,651 dc) | phase4b_final_report |
| C1 DARTH vs B1 (seed 42) | 1,172 vs 1,057 dc; saving −11.0% [−13.0, −8.9]; recall 0.963 vs 0.948 | exp_c_report |
| C1 replication (seeds 43/44) | −11.5% [−13.5, −9.5] / −10.8% [−12.8, −8.8] | notes (E Stage 1) |
| Phase 5′ largest signal drift | centroid_dist ood 4%: Δ −0.037, CI lb −0.045 | phase5b_report |
| Phase 5′ per-query effort reshuffling | rank corr with S0 0.84 at 8% ID/OOD | phase5b_report |
| E H-E1 B1, 8% deletion | D̄ +0.0524 [0.051, 0.054] (1,125 vs 1,068 dc, seed 42) | exp_e_report |
| E H-E1 DARTH, largest | D̄ −0.0224 (del 8%), −0.0210 (id 8%) | exp_e_report |
| E H-E2 DARTH, largest | Ē +0.0326 [0.018, 0.046] (id 8%) | exp_e_report |
| E state-local reference | insertion ef 50/53 (w_hi 0.48–0.66); del 8% ef 48/50 (w_hi 0.048–0.24) | exp_e_report §G |
| E H-E3 | 10/16 associated (Holm) | exp_e_report §D |
| E H-E4 | bias −0.0013 (S0) → ≤ +0.0046; MACE 0.0537 → ≤ 0.0569; stopped 0.942–0.946 | exp_e_report §E |
| E trends (DARTH) | D −0.0053/doubling [−0.0071, −0.0035]; E +0.0066 [+0.0040, +0.0092] | exp_e_report §F |

## 7. Supported claims (safe for the paper)
1. On SIFT1M/hnswlib (M16, efC200, k = 10), frozen cheap-feature routers did not beat a
   calibrated fixed-effort baseline under pre-registered gates (three seeds, two query sets).
2. A re-implementation of DARTH on hnswlib cost about 11% more distance computations than B1 at
   mean recall@10 = 0.95, at higher realised recall; this replicated across three index seeds.
3. Within 1–8% insertion and 2–8% lazy deletion, the population-level rank association between
   three cheap features and oracle-required effort stayed within 0.05 of S0, while per-query
   required effort reshuffled.
4. Within the same regime, S0-frozen B1 and S0-frozen DARTH stayed within ±10% cost drift and
   ±5 pp excess contract loss relative to a freshly calibrated reference (pre-registered,
   precise CIs).
5. Measurable sub-threshold drift occurred and was policy-specific:
   - DARTH: slightly cheaper but losing S0 passes faster, growing with magnitude;
   - B1: overpaying up to 5.2% after deletion.
6. S0 anchoring (D) is needed to keep a policy's known S0 inefficiency out of staleness
   estimates.

## 8. Forbidden / unsupported claims
- **Adaptive ANN routing:** that it "works" or "fails" in general, or that it cannot beat fixed
  effort (Phase 4b headroom exists).
- **Robustness:** that ANN/HNSW/DARTH is "robust" or "immune" to index evolution, or that
  "no drift" occurred.
- **Generalisation:** to other datasets, metrics, index types or parameters (M, efC), k, query
  workloads, churn beyond 8%, physical deletion or graph repair.
- **Genuine OOD:** the "OOD" states are regionally concentrated pool vectors; 8% ≈ ID by
  construction.
- **Structural predictors:**
  - that they predict staleness or policy invalidation (none occurred);
  - that update fraction is the best predictor;
  - that structural measures are inferior.
- **DARTH calibration:**
  - that DARTH is calibrated over all queries, or probabilistically (H-E4 covers only
    early-stopped queries);
  - anything about official DARTH (this is a re-implementation with a feature-order deviation).
- **Wall-clock performance:** any latency or QPS claim; wall-clock was not interpreted.
- **Quality-matched efficiency:** C1 compared at target, not at matched realised recall.
- **Novelty:** none without separate literature verification (§14).

## 9. Limitations
- **Dataset/index:**
  - one dataset (SIFT1M, 128-d, L2);
  - one hnswlib configuration (M16, efC200);
  - k = 10;
  - the 2,000-query Phase-2 test set (same distribution as base).
- **Evolution process:**
  - 1–8% insertion from a single 87,788-vector pool;
  - 2–8% lazy deletion (markDelete; no repair);
  - nested states;
  - narrow regional "OOD" (≈ ID at 8%).
- **Policy/reference:**
  - B1-style state-local reference (not a recalibrated DARTH); near-identical to frozen B1 under
    insertion;
  - DARTH failed C1 at S0;
  - H-E2 cohort-composition caveat.
- **Statistics:**
  - three index seeds;
  - query-only bootstrap (seeds fixed);
  - H-E3 predictors collinear with magnitude and partly seed-independent (≈ 8 independent values
    over 24 cells), so p-values are likely anti-conservative;
  - mixed models on the boundary (seed variance ≈ 0), DARTH fits needing an optimiser retry;
  - margins (±10%, ±5 pp, X = 0.05) fixed a priori.
- **Implementation:**
  - DARTH independently re-implemented on hnswlib (termination after the current node; hnswlib
    beam rule; intent-faithful feature order);
  - LightGBM refits functionally but not bitwise reproducible;
  - check timing on deletion states can differ slightly from the reference implementation.
- **Metric:**
  - distance computations only (additive probe in Phase 4b);
  - wall-clock not interpreted;
  - tie-aware recall@10 contract.
- **External validity:** no other datasets, embedding models, index families or production
  workloads; the secondary dataset planned in the design (Phase 9) was not run.

## 10. Figure plan (from existing data only; not created)
| # | Purpose | x-axis | y-axis | Grouping | Source | Role |
|---|---|---|---|---|---|---|
| F1 | Effort heterogeneity at S0 | oracle ef (log) | query share / CDF | train vs test | results/8e605bfc5769_…/oracle_curves.csv (existing oracle_distribution.png) | Supplementary |
| F2 | S0 adaptive policies vs B1 | contract level (S* 0.90 / 0.95 / 0.99; C1 R 0.95) | saving vs B1 (%) with 95% CI | policy (Phase 4b router, DARTH seeds 42/43/44) | phase4b eval tables; exp_c / exp_e S0 eval reports | Primary |
| F3 | Signal stability (Phase 5′) | update magnitude (% of \|S0\|, log2) | pooled Δ\|ρ\| with CI; ±0.05 band | feature × trajectory | results/phase5b_analysis_…/delta_cells_pooled.csv (existing delta_vs_magnitude.png) | Primary |
| F4 | Cost staleness H-E1 | magnitude (log2) | pooled D̄ with CI; ±0.10 band | policy × trajectory (ID / OOD / deletion) | results/exp_e_analysis_20261005T110646Z/analysis.json | Primary |
| F5 | Contract staleness H-E2 | magnitude (log2) | pooled Ē with CI; ±0.05 band | policy × trajectory | same analysis.json | Primary |
| F6 | Cost vs contract decomposition (DARTH) | D̄ | Ē | state (marker = trajectory, size = magnitude) | same analysis.json | Supplementary |
| F7 | H-E3 associations | ρ with 95% CI (forest) | 16 tests | policy × outcome × predictor; Holm mark | same analysis.json | Supplementary |
| F8 | H-E4 among early-stopped queries | magnitude | bias and MACE change from S0 (CI) | trajectory | same analysis.json | Supplementary |
| F9 | Reference recalibration | state | REF high-ef weight / ef pair | seed | results/exp_e_rows_20261005T110522Z/metadata.json | Supplementary |

## 11. Table plan
1. **T1 Setup:** dataset, index, queries, splits, seeds, cost/quality definitions, evolution
   states (with OOD and lazy-deletion notes).
2. **T2 Phase-gate summary:** Phase 4 / 4b / C1 gates and results (from §5).
3. **T3 Phase 5′ H1′:** pooled Δ per feature × state (from docs/phase5b_report.md).
4. **T4 E H-E1:** pooled D̄ [CI] and class per policy × state, plus mean cost and recall.
5. **T5 E H-E2:** pooled Ē [CI], class, and per-seed INCONCLUSIVE notes per policy × state.
6. **T6 (supplementary):** H-E3 16 tests (ρ, CI, raw p, Holm p).
7. **T7 (supplementary):** per-seed H-E1/H-E2, H-E4 per state, and the trend models with fit
   diagnostics.

## 12. Final paper structure
1. Introduction: adaptive search effort, index evolution and policy validity; scoped
   contributions.
2. Background and related work (requires literature verification): HNSW effort control;
   adaptive/early-termination methods (DARTH, Ada-ef, LAET); index updates.
3. Experimental framework: SIFT1M/hnswlib setup, oracle effort, cost/contract definitions,
   pre-registration and phase gates, statistics.
4. Part I — Is there an S0 advantage? Phase 4 / 4b (static router) and C1 (DARTH
   re-implementation); negative results and headroom.
5. Part II — Validity under evolution:
   - 5.1 evolved states and their validation;
   - 5.2 Phase 5′ signal stability;
   - 5.3 Experiment E calibration staleness: D, excess newly failing, reference behaviour,
     structural associations, H-E4, trends.
6. Discussion: the stable-but-drifting picture; cost vs contract decomposition; why adaptive
   policies did not pay off here; what recalibration changes (deletion).
7. Limitations and threats to validity (§9).
8. Reproducibility statement (§15).
9. Conclusion (scoped).

## 13. Abstract specification (bullets only)
- **Problem:** search-effort policies for HNSW are calibrated on a static index; production
  indexes evolve.
- **Gap:** whether S0 calibrations of fixed and runtime-adaptive policies stay valid under index
  evolution, and whether structural change tracks drift. Gap status requires literature
  verification.
- **Method:**
  - pre-registered, multi-seed study on SIFT1M/hnswlib;
  - 30 nested evolved states (1–8% insertion, 2–8% lazy deletion);
  - frozen B1 and DARTH re-implementation vs state-local recalibrated reference;
  - S0-anchored proportional cost drift and excess contract loss with three-valued margin rules.
- **Strongest findings:**
  - no S0 efficiency advantage for adaptive policies here;
  - feature→effort signal stable;
  - both frozen policies within ±10% / ±5 pp in every state;
  - measurable sub-threshold, policy-specific drift (DARTH −2% cost, +3 pp contract loss; B1
    +5% under deletion).
- **Contribution:** a validated protocol and estimands for policy-validity under index evolution;
  scoped negative and precise-null results.
- **Limitations:** one dataset and configuration, bounded churn, lazy deletion, B1-style
  reference, re-implemented DARTH, distance-computation metric.

## 14. Novelty status
- **Established by the literature audit (as documented in this repository):**
  - Ada-ef's published estimator covers only inner product and cosine, not L2;
  - the DARTH paper evaluates SIFT100M/GIST (L2) on static indexes;
  - Ada-ef's repository contains incremental update/deletion experiments for its own method.
  - These are the only literature facts recorded in the repository.
- **Supported empirical gap (this project):** pre-registered measurement of S0-calibration
  validity of fixed and runtime-adaptive effort policies under controlled insertion/deletion, with
  S0-anchored estimands.
- **Not yet independently verified:**
  - whether prior work already measures policy or calibration staleness under index updates
    (including the Ada-ef update experiments and learned early-termination literature);
  - whether the D-type estimand or a state-local recalibration reference exists elsewhere.
  - The broader literature audit that motivated the reframing is **not documented in the
    repository**.
- **Must not be claimed:** "first", "novel", or "no prior work" statements, until a documented
  literature review is completed.

## 15. Reproducibility readiness (documentation gaps only; nothing fixed)
**Present:**
- YAML configs for every run;
- run metadata (config, seeds, compiler, hnswlib commit);
- pinned submodules (hnswlib v0.8.0, googletest, yaml-cpp, LightGBM v4.6.0);
- pinned Python requirements;
- frozen model/policy files and hashes in derived/;
- a SIFT1M download script;
- notes recording every decision;
- design addenda;
- deterministic pipelines;
- unit tests (68 gtests, 64 pytest).

**Gaps:**
1. **No README:** no build instructions, environment setup (.venv, LightGBM static build), or
   per-phase command runbook in one place.
2. **No LICENSE** for the repository.
3. **results/ and data/ are gitignored.** Every reported run directory (Phase 2–E) is absent from
   a public clone; only derived/ artifacts are tracked. Reproduction requires regenerating
   everything, and no summary result bundle is published.
4. **No git history yet:** run metadata records git commit "none"/dirty, so provenance relies on
   file hashes in notes.
5. **The literature audit is undocumented** (§14).
6. **There is no single index** mapping each paper number to its artifact path; it is spread
   across notes and reports.
7. **The open item from Phase 0/1 remains:** clang-tidy cannot analyse files including omp.h
   (libomp-18-dev).
8. **Machine-specific aspects are undocumented:** 16 threads, wall-clock not interpreted, and
   build flags (-march=native) affect the binary but not the deterministic counts.
9. **The secondary dataset (design Phase 9) and Phases 7′–10 were never run;** the design
   document still lists them.

## 16. Final readiness verdict
- **Experimentally complete?** Yes, for the frozen scope. All gated phases ran to a recorded
  decision, the evolution experiments (Phase 5′, E) are complete and validated, and no pending
  experiment is required for the current claims. The secondary dataset and later design phases
  were not run and must be described as out of scope or future work, not as done.
- **Coherent enough for a paper?** Yes. The chain runs: S0 premise not met → reframing → signal
  stability → calibration validity with precise nulls and quantified drift. It is coherent if
  presented as a policy-validity study with scoped negative results, not as an adaptive-routing
  success.
- **Before drafting:**
  - a documented literature verification for related work and novelty (§14);
  - write up the literature audit that motivated the reframing;
  - consolidate a number-to-artifact index;
  - decide the figure set (§10).
- **Before public release:**
  - README with build and runbook;
  - LICENSE;
  - publish or archive the result artifacts behind every reported number;
  - commit history;
  - resolve or document the clang-tidy/omp item;
  - state the scope of unrun design phases.

---

**A. Final research question:** On SIFT1M/hnswlib, do S0-calibrated fixed-effort and
runtime-adaptive search policies remain valid in cost and in recall contract, relative to a
freshly calibrated reference, under bounded index evolution (1–8% insertion, 2–8% lazy
deletion)? And does the cheap-feature→effort signal they rely on remain stable?

**B. Final contribution:** a pre-registered, multi-seed protocol and S0-anchored estimands for
measuring policy validity under index evolution. They show scoped negative results for S0
adaptive efficiency, a stable feature→effort signal, and measurable but sub-threshold,
policy-specific calibration drift. Novelty is unverified.

**C. Three strongest findings:**
1. Both S0-frozen policies (B1, DARTH) stayed within ±10% cost drift and ±5 pp excess contract
   loss in all 30 evolved states, with precise CIs.
2. Measurable sub-threshold drift was policy-specific: DARTH became about 2% relatively cheaper
   while losing S0 passes up to 3.3 pp faster, growing with magnitude; B1 overpaid 5.2% after 8%
   deletion.
3. Neither a frozen cheap-feature router (Phase 4b: −20.9% / −8.9% / +1.1%) nor the DARTH
   re-implementation (C1: about −11%, three seeds) beat calibrated fixed effort at S0 on SIFT1M,
   despite about 47% perfect-information headroom.

**D. Three biggest limitations:**
1. A single dataset and HNSW configuration (SIFT1M, M16/efC200, k = 10).
2. A bounded evolution regime: at most 8% churn, lazy deletion only, narrow regional "OOD".
3. The reference is B1-style (nearly identical to frozen B1 under insertion), and DARTH is a
   re-implementation that failed at S0.

**E. Remaining work:**
- documented literature verification and a write-up of the literature audit;
- a number-to-artifact index;
- figure generation from existing artifacts;
- README, LICENSE, artifact publication and commit history before public release.

**F. Paper readiness: READY** (for drafting; the evidence is complete and audited). Literature
verification must precede any novelty or related-work claims, and the release items in §16 must
precede public release.
