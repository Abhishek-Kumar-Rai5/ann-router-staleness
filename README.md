# Search-Effort Policy Validity Under HNSW Index Evolution

A pre-registered empirical study, on SIFT1M with hnswlib, of whether approximate-nearest-neighbour
(ANN) search-effort policies calibrated once on a static index stay valid, in cost and in
per-query recall contract, as the index evolves through bounded insertion and lazy deletion.

This repository is the frozen research record. It contains the code, configurations, frozen
policies and models, query splits, pre-registration documents, reports and audits. The
experiments are complete. The accompanying paper is in preparation.

**Status at a glance**

| Item | State |
|---|---|
| Experiments | complete and frozen (Phases 1–4, 4b, C1, 5′, Experiment E) |
| Paper | in preparation; not yet available |
| Result/artifact bundle | not yet published (see [Reproducibility](#reproducibility)) |
| License | not yet added (see [License](#license)) |

## Contents
- [1. The problem](#1-the-problem)
- [2. Research question](#2-research-question)
- [3. Experimental system](#3-experimental-system)
- [4. How the study developed](#4-how-the-study-developed)
- [5. Experiment E: frozen-policy validity](#5-experiment-e-frozen-policy-validity)
- [6. Key findings](#6-key-findings)
- [7. Scope and non-claims](#7-scope-and-non-claims)
- [8. Relation to prior work](#8-relation-to-prior-work)
- [9. Repository structure](#9-repository-structure)
- [10. Build, dependencies and tests](#10-build-dependencies-and-tests)
- [Reproducibility](#reproducibility)
- [Provenance and frozen status](#provenance-and-frozen-status)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## 1. The problem

Graph-based ANN indexes such as HNSW expose a search-effort knob: the beam width `ef`. Queries
differ greatly in how much effort they need to retrieve their true nearest neighbours.
Search-effort *policies* exploit or manage this heterogeneity. They come in two families:
- **Fixed-effort policies** use one global `ef`, or a calibrated mix of two `ef` values.
- **Query-adaptive or runtime-adaptive policies** use a per-query `ef` chosen from query features,
  or early termination driven by a learned recall predictor.

Every such policy is **calibrated** on one state of the index: a threshold, a model, an `ef` mix
or a stopping interval is fitted so that a recall target is met at minimum cost. Production
indexes do not stay fixed. Vectors are inserted, and vectors are deleted (in hnswlib, lazily,
by marking them).

**Policy staleness**, as used in this project, is the question of whether a policy calibrated at
the initial index state S0 is still *valid* at a later state s. "Valid" has two separate parts:
1. **Cost validity.** Does the frozen policy still spend about what a policy freshly calibrated
   at s would spend for the same target?
2. **Contract validity.** Do queries that met the per-query recall contract at S0 keep meeting
   it, compared with a freshly calibrated policy?

These two can drift independently, and in this study they moved in opposite directions for one
of the policies. "Policy validity" is therefore broader than whether a calibrated threshold is
out of date.

Both fixed-effort and adaptive policies need a calibration. If that calibration silently decays
as the index changes, the recall a system promises, or the compute it budgets, is no longer
what it delivers.

## 2. Research question

> On SIFT1M / hnswlib-HNSW (L2, M = 16, efConstruction = 200, k = 10), do a fixed-effort policy
> and a runtime-adaptive policy, both calibrated at S0, remain valid in cost and in per-query
> recall-contract terms, relative to a freshly calibrated state-local reference, under bounded
> insertion (1–8%) and lazy deletion (2–8%)? And does the relationship between cheap query
> features and required search effort, on which such policies rely, remain stable?

Two secondary questions:
- whether structural measures of index change are associated with any drift;
- whether the runtime policy's recall predictor stays unbiased.

The question was not the starting point. The project began by asking whether per-query adaptive
effort *saves* cost on a static index, and whether that saving survives index evolution. The
saving was not found at S0 (Section 4), so the project was reframed, with documented, approved
design addenda, to the validity question above. Its development is described in
[Section 4](#4-how-the-study-developed).

## 3. Experimental system

| Component | Setting (source) |
|---|---|
| Dataset | SIFT1M (TEXMEX): 1,000,000 base vectors, 128-d, L2; 10,000 queries (`scripts/download_sift1m.sh`) |
| Index | HNSW via hnswlib **v0.8.0** (commit `3f342966`), used unmodified as a black box; M = 16, efConstruction = 200; single-threaded builds for bit-reproducibility |
| Index seeds | 42, 43, 44 (Phase 4b onwards; Phases 1–4 and C1 use seed 42) |
| Ground truth | exact brute force, independent of hnswlib, recomputed fresh for every index state; verified against the official TEXMEX ground truth (0 of 10,000 rows differ beyond exact distance ties) |
| Recall | **tie-aware Recall@10**: a returned neighbour counts if its distance is within the true 10th-neighbour distance. SIFT1M contains duplicate vectors, so id-based recall would understate correct answers. ID-based recall is also recorded |
| Per-query contract | success ⇔ tie-aware Recall@10 = 1 (10/10) |
| Cost metric | **distance computations per query**, counted by a wrapper around hnswlib's own L2 kernel, upper layers included. Wall-clock is recorded but not used for conclusions |
| Query split | deterministic: 8,000 train / 2,000 test (`splits/sift1m_query_split_seed20261001.csv`, Fisher–Yates on `mt19937_64`, seed 20261001). Calibration uses train only; evaluation uses test only |
| Per-query oracle | for each query, the smallest `ef` on a fixed 121-value grid (10–4096) from which tie-aware Recall@10 = 1 holds at every larger grid value |
| Query features | from a cheap ef = 10 probe: `knn_dist` (probe 10th-neighbour distance), `centroid_dist` (distance to the base centroid), `score_concentration` (d₁/d₁₀); `lid` (Levina–Bickel estimate, ef = 20 probe) as a secondary feature |
| Index evolution | 30 nested evolved states (10 per seed). Insertion of 10k / 20k / 40k / 80k vectors (1 / 2 / 4 / 8%), either in-distribution or regionally concentrated ("OOD"; see limitations), from an 87,788-vector `sift_learn`-derived pool. Lazy deletion (`markDelete`) of 2% and 8% of base vectors |
| Statistics | query-clustered bootstrap (2,000 resamples, seed 20261002; a query moves together with all its seeds, states and policies); three-valued decision rules against pre-registered margins; Holm–Bonferroni correction where specified |

## 4. How the study developed

Each phase had a validation checkpoint or pre-registered gate defined before its results were
seen. Negative outcomes are reported as results, not hidden.

### Phase 1: static baseline
The S0 index, ground truth and fixed-`ef` recall/cost curve were established and verified.
- Ground truth matches the official file up to ties.
- Recall and distance counts are deterministic across repeats and runs.
- **Checkpoint met.** Record: [`docs/notes.md`](docs/notes.md) (Phase 1).

### Phase 2: per-query oracle effort
The `ef` each query actually needs was measured.
- Required effort is highly heterogeneous. On the 8,000 training queries the oracle `ef` has
  median 29, p90 117, p99 327 and maximum 2,794, with 17.3% at the grid floor (ef = 10).
- At the oracle's mean budget (about 1,137 distance computations), the best single fixed `ef`
  reaches mean recall 0.955.
- **Checkpoint met.**

### Phase 3: cheap query features
The question was whether a cheap probe carries signal about required effort. Training-split
Spearman correlations with oracle `ef` were:

| Feature | knn_dist | centroid_dist | score_concentration | lid (secondary) |
|---|---|---|---|---|
| Spearman ρ | +0.459 | −0.445 | +0.363 | +0.558 |

The signal is real but moderate. **Checkpoint met.**

### Phase 4: initial feature router (negative)
A pre-registered three-tier router (a depth-2 decision tree on the probe features) was tested
once on the test split. It was compared with the fixed `ef` that matches its mean recall.
- The router spent **2,480** distance computations per query, against **1,078** for fixed
  `ef` = 53, at the same mean recall of 0.950.
- Two of the three pre-registered criteria failed: the router was not cheaper, and it did not
  save ≥ 10%. **Checkpoint not met.**

A subsequent methodology audit and redesign are documented in
[`docs/methodology_audit.md`](docs/methodology_audit.md),
[`docs/phase4_methodology_validation.md`](docs/phase4_methodology_validation.md) and
[`docs/phase4_redesign.md`](docs/phase4_redesign.md).

### Phase 4b: redesigned probability-based routing (negative)
The redesign was frozen before evaluation:
- per-candidate models of P(query succeeds at effort e | features) over a doubling effort ladder;
- a threshold τ calibrated on training out-of-fold predictions;
- three index seeds;
- a fresh, deduplicated 2,000-query confirmation set drawn from `sift_learn`.

The baseline **B1** is a two-`ef` mix calibrated on training data to the same target. The gate
required, at every target S\*, a saving vs B1 of ≥ 10% with a CI lower bound > 0 and Wilcoxon
p < 0.01, plus quality non-inferiority and per-seed conditions.

| Success target S\* | Saving vs B1 [95% CI] | Gate |
|---|---|---|
| 0.90 | −20.9% [−23.3, −18.5] | FAIL |
| 0.95 | −8.9% [−11.1, −6.8] | FAIL |
| 0.99 | +1.1% [−0.7, +2.9] | FAIL |

The router cost more than B1 at 0.90 and 0.95, and was at parity at 0.99. A perfect-information
policy would have saved about 47% at S\* = 0.95 (1,397 vs 2,651 distance computations). The
cheap features could not exploit that headroom. Report:
[`docs/phase4b_final_report.md`](docs/phase4b_final_report.md); audit:
[`docs/phase4b_post_result_audit.md`](docs/phase4b_post_result_audit.md).

### Experiment C (C1): DARTH-style runtime adaptation (negative at S0)
A runtime-adaptive actuator was tested separately: an **independent, mechanism-faithful
re-implementation** of DARTH (declarative-recall early termination with a LightGBM recall
predictor) on hnswlib's public stop-condition API.
- Ada-ef was considered first and rejected: its published estimator does not support L2.
- No DARTH or FAISS code is copied.
- One documented deviation: the feature order follows the model's training order. See
  [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`docs/exp_c_report.md`](docs/exp_c_report.md).

The comparison was against B1 calibrated to mean Recall@10 = 0.95 (seed 42), on the 2,000 test
queries.

| | DARTH re-implementation | B1 |
|---|---|---|
| Mean tie-aware Recall@10 | 0.9633 | 0.9480 |
| Mean distance computations | 1,172 | 1,057 |
| Saving vs B1 [95% CI] | **−11.0%** [−13.0, −8.9] | — |

- The 10% saving gate failed (**C1 FAIL**). The policy overshoots the recall target and spends
  heavily on a hard tail.
- It replicated on seeds 43 and 44 at −11.5% and −10.8%.
- This is a result about this re-implementation, in this configuration, against this baseline.
  It is not a verdict on DARTH or on adaptive search in general.

### Phase 5′: feature→effort signal stability under evolution
Phases 4b and C1 showed no S0 advantage whose robustness could be tested. The project was
therefore reframed (design-doc Addendum A1) to ask whether the *ingredients* of such policies
survive index evolution.
- The 30 evolved states were built deterministically. All 30 state files reproduce
  byte-for-byte (validation V1–V11 all pass).
- Fresh ground truth, oracle effort and features were computed at every state.

**Result: H1′ supported.** The change in |Spearman ρ| between each feature and oracle effort
stayed within the pre-registered margin of 0.05 in **all 40 pooled** (feature × state) cells and
**all 120 per-seed** cells.
- The tightest cell is `centroid_dist` at regional-OOD 4%: Δ = −0.037, CI lower bound −0.045.
- Individual queries' required effort *does* reshuffle: the per-query effort rank correlation
  with S0 is 0.84 at 8% insertion. The population-level relationship is stable; per-query
  calibration is not guaranteed to be.

Report: [`docs/phase5b_report.md`](docs/phase5b_report.md).

### Experiment E: frozen-policy validity
This is the central experiment, described in the next section.

## 5. Experiment E: frozen-policy validity

Design: [`docs/exp_e_design_freeze_v2.md`](docs/exp_e_design_freeze_v2.md) (frozen 2026-10-05
before any evolved-state run; appended to [`docs/design_doc.md`](docs/design_doc.md) as
Addendum E1). Report: [`docs/exp_e_report.md`](docs/exp_e_report.md).

### Policies (frozen at S0, never retuned or retrained)
- **B1 (fixed effort).** The Phase-4b two-`ef` mix calibrated to mean Recall@10 = 0.95: `ef`
  50/53 with high-`ef` weight 0.538 / 0.596 / 0.637 for seeds 42 / 43 / 44. Each query is
  assigned an `ef` by a deterministic hash, and every query keeps its S0 `ef` at every state.
- **DARTH (runtime adaptive).** The frozen C1 policy for each seed (recall target 0.95, `ef` cap,
  LightGBM model). Files are in `derived/exp_c/` and `derived/exp_e/`. It is studied as a
  calibrated actuator, not as a winning method: it failed C1.

### Reference
**REF(σ, s)** is a B1-style two-`ef` mix recalibrated independently at each seed σ and state s,
on that state's training-query recall curve. By construction REF(σ, S0) = B1(σ) exactly. For
DARTH, REF is a freshly calibrated *simple* policy, not a recalibrated DARTH.

### Estimands and pre-registered decision rules
**H-E1, cost drift.** The cost ratio to the reference is anchored at S0. This keeps a policy's
known S0 inefficiency (C1) out of the staleness estimate:

$$
D(P,\sigma,s) \;=\; \frac{\bar C_P(\sigma,s)\,/\,\bar C_{\mathrm{REF}}(\sigma,s)}{\bar C_P(\sigma,S_0)\,/\,\bar C_{\mathrm{REF}}(\sigma,S_0)} \;-\; 1
$$

Here C̄ is the mean distance computations per query over the 2,000 test queries.
- **Margin:** ±0.10.
- **STABLE** if the 95% CI lies within [−0.10, +0.10];
- **STALE-COSTLIER** or **STALE-CHEAPER** if the CI lies entirely beyond the margin;
- **INCONCLUSIVE** otherwise.

**H-E2, recall-contract drift.** This is the excess rate at which queries that passed the
10/10 contract at S0 newly fail at state s, relative to the reference:

$$
E(P,\sigma,s) \;=\; \mathrm{NF}(P,\sigma,s) - \mathrm{NF}(\mathrm{REF},\sigma,s),\qquad
\mathrm{NF}(P,\sigma,s) = \frac{\#\{q \in K_P(\sigma) : \text{fail}_P(\sigma,s,q)\}}{|K_P(\sigma)|}
$$

K_P(σ) is policy P's S0 pass cohort. The margin is ±0.05 (5 percentage points), with the same
three-valued rule.

**H-E3, structural association (secondary).** 16 Spearman tests (2 policies × 2 outcomes × 4
predictors) over the 24 insertion cells, with bootstrap p and Holm correction. The predictors
are:
- update fraction;
- neighbourhood-overlap decay (1 − top-10 overlap with S0 / 10);
- local-density drift;
- distributional drift.

**H-E4 (descriptive).** DARTH's predicted vs realised recall, among early-stopped queries only.

**Other design rules:**
- Pooled verdicts average the three seeds with equal weight.
- Per-seed verdicts are always reported.
- One query-resample matrix is shared across all seeds, states and policies.
- Secondary magnitude trends use mixed models with a seed random intercept.

## 6. Key findings

Numbers are taken from the frozen reports linked above. Pooled values are means over seeds 42,
43 and 44, with 95% query-bootstrap CIs.

1. **Query difficulty is highly heterogeneous.** Oracle `ef` ranges from the grid floor (10) to
   2,794 on the training queries (median 29, p90 117).
2. **Cheap query features carry real but moderate signal** about required effort (|ρ| 0.36–0.46
   for the three core features; 0.56 for LID).
3. **The initial feature router failed its pre-registered gate.** It cost 2,480 vs 1,078
   distance computations for the matched-recall fixed `ef`.
4. **The redesigned Phase-4b router also failed its ≥ 10% savings gate** at all three success
   targets: −20.9% / −8.9% / +1.1% vs B1 at 0.90 / 0.95 / 0.99.
5. **The DARTH re-implementation failed its ≥ 10% savings gate against B1.** It cost 11.0% more
   (CI −13.0% to −8.9% saving) at higher recall (0.963 vs 0.948), replicated on all three seeds.
6. **The feature→effort relationship was stable** across all tested evolved states: all 40
   pooled and 120 per-seed cells were within the 0.05 margin. Per-query required effort
   nevertheless reshuffled.
7. **No state crossed the ±10% cost-drift margin for either frozen policy (H-E1).** All 20 pooled
   (policy × state) cells and all 60 per-seed cells were STABLE.
   - B1 pooled D̄ ranged from −0.0005 to +0.0040 under insertion, +0.0157 at 2% deletion and
     **+0.0524 [0.051, 0.054] at 8% deletion**. After deletion the recalibrated reference moves
     to a cheaper mix, while frozen B1 keeps its S0 `ef` at slightly higher recall.
   - DARTH pooled D̄ ranged from −0.0021 to −0.0224: slightly *cheaper* relative to the
     reference as the index changes.
8. **Recall-contract drift was measurable for DARTH but stayed inside the ±5 pp margin (H-E2).**
   - B1: Ē from −0.0075 to 0.0000.
   - DARTH: Ē from +0.0022 to **+0.0326 [0.018, 0.046] (in-distribution 8%)**.
   - All pooled cells are STABLE. Three DARTH per-seed cells are **INCONCLUSIVE**, not stale:
     in-distribution 8% (seeds 42 and 44) and deletion 8% (seed 42), with CI upper bounds just
     above 0.05.
   - DARTH's drift grows with magnitude: per doubling of insertion magnitude, the cost drift is
     −0.53 pp [−0.71, −0.35] and the contract drift is +0.66 pp [+0.40, +0.92]. B1 stays flat.
9. **Structural measures were associated with some sub-threshold drift (H-E3, secondary).**
   - 10 of 16 tests were associated after Holm: B1 cost 4/4, B1 contract 2/4, DARTH cost 4/4,
     DARTH contract 0/4.
   - No structural measure was detectably more strongly associated than the plain update
     fraction.
   - The predictors are collinear with magnitude and partly repeated across seeds, so the
     p-values should be read cautiously.
   - No policy became stale, so nothing here shows that structural measures *predict policy
     invalidation*.
10. **DARTH's recall predictor stayed near-unbiased among early-stopped queries (H-E4,
    descriptive).** About 94% of queries stop early. ID-recall bias went from −0.0013 at S0 to
    at most +0.0046, and MACE from 0.0537 to at most 0.0569. The largest change was at 8%
    deletion.

**Summary.** Measurable, policy-specific drift occurred: DARTH became slightly cheaper while
losing S0 contract passes slightly faster, and B1 overpaid after deletion. This drift stayed
below the pre-registered practical invalidation margins throughout the tested regime.

These results do **not** show that adaptive ANN routing is ineffective in general. They show
that, in this configuration, the adaptive policies tested here did not beat a calibrated
fixed-effort baseline at S0, and that their calibrations did not become practically stale
within bounded evolution.

## 7. Scope and non-claims

The conclusions are bounded to:
- SIFT1M (128-d, L2);
- hnswlib HNSW with M = 16 and efConstruction = 200;
- k = 10 and the tie-aware Recall@10 contract;
- the 2,000-query test split;
- three index seeds;
- 1–8% insertion and 2–8% lazy deletion.

Within that scope, this study does **not** establish:
- **General robustness** of adaptive ANN policies, or that any policy is "immune" to index
  evolution. STABLE means "within the pre-registered margins here", not "no drift"; drift was
  measured.
- **Behaviour beyond the tested regime:** above 8% insertion, above 8% deletion, or under
  physical deletion, compaction or graph repair. Deletion here is lazy `markDelete` only.
- **Behaviour on other ANN architectures, libraries or index parameters** (M, efConstruction),
  other k, or other datasets, embedding models or metrics.
- **Independent out-of-distribution generalisation.** The "OOD" states are regionally
  concentrated real SIFT vectors from the same pool as the in-distribution inserts. At 8% the
  OOD set overlaps the ID set by about 91%, so 8% "OOD" is approximately ID by construction.
- **That structural index-change measures predict policy invalidation.** No invalidation
  occurred. The associations concern sub-threshold drift only, and no measure beat update
  fraction.
- **That DARTH is calibrated in general.** H-E4 covers early-stopped queries only and is not
  probabilistic calibration. The DARTH policy is an independent re-implementation, not the
  official system.
- **That adaptive routing is generally non-viable.** Phase 4b documented substantial
  perfect-information headroom. The negative results are scoped to these features, models,
  baselines and gates.
- **Any wall-clock or throughput conclusion.** Wall-clock times were recorded but not
  interpreted. The cost metric is distance computations.
- **A quality-matched efficiency comparison for DARTH.** C1 compared policies at the same
  target, and DARTH realised higher recall.
- **A universal statement about ANN policy staleness.**

## 8. Relation to prior work

Source: [`docs/literature_verification.md`](docs/literature_verification.md). That document is a
targeted verification of the closest work, not an exhaustive systematic review.

Query-adaptive ANN search effort is established. Examples include:
- learned adaptive early termination, LAET (Li et al., SIGMOD 2020,
  [doi:10.1145/3318464.3380600](https://doi.org/10.1145/3318464.3380600));
- DARTH (Chatzakis, Papakonstantinou, Palpanas, PACMMOD 2025,
  [doi:10.1145/3749160](https://doi.org/10.1145/3749160));
- PiP (ECIR 2025, [doi:10.1007/978-3-031-88714-7_39](https://doi.org/10.1007/978-3-031-88714-7_39));
- Ada-ef (Zhang and Miller, PACMMOD 2026,
  [doi:10.1145/3786639](https://doi.org/10.1145/3786639));
- QBAT (PVLDB 19(11), 2026, [doi:10.14778/3836663.3836675](https://doi.org/10.14778/3836663.3836675));
- OMEGA (preprint, [arXiv:2603.06159](https://arxiv.org/abs/2603.06159)).

These methods are developed and evaluated mainly on static indexes.

Staleness under updates has also been observed before:
- Ada-ef's evaluation compares stale, incrementally updated and recomputed versions of its
  estimator after 10% and 50% batch updates.
- OMEGA reports that its learned search models need retraining after index compaction.
- Index degradation under updates is studied separately, for example in FreshDiskANN
  ([arXiv:2105.09613](https://arxiv.org/abs/2105.09613)) and work on unreachable points in HNSW
  ([arXiv:2407.07871](https://arxiv.org/abs/2407.07871)).

This repository's contribution is therefore narrower: a controlled measurement of frozen-policy
validity under bounded index evolution. Its features are:
- a fixed-effort and a learned runtime policy under one protocol;
- comparison against a state-local recalibrated reference;
- cost drift and recall-contract drift estimated separately;
- pre-registered margins and three index seeds.

The strongest wording the literature check supports is cautious:

> We are not aware of a controlled, policy-comparative study that separates cost drift from
> recall-contract drift with pre-registered margins under bounded index evolution, comparing
> frozen policies with state-local recalibration.

The regime also differs from OMEGA's (compaction rebuilds, large update ratios). The STABLE
results here are therefore not presented as contradicting its retraining finding.

## 9. Repository structure

```
.
├── apps/            C++ experiment drivers (one YAML config per run)
├── src/, include/ars/  C++ core library: vector I/O, exact ground truth, hnswlib wrapper,
│                    distance counting, oracle, features, policies, DARTH stop condition
├── tests/           GoogleTest suites (C++)
├── python/          orchestration, router training, statistics and analysis; tests in python/tests/
├── configs/         frozen experiment configurations, by phase (phase1–4, phase4b/, phase5/, exp_c/, exp_e/)
├── derived/         small frozen artifacts: effort tiers and strata, DARTH models and policies
├── splits/          deterministic query split and confirmation-set definitions (+ _invalid/ audit trail)
├── docs/            design document, pre-registrations, reports, audits, literature verification
├── scripts/         dataset download and verification (download_sift1m.sh)
├── third_party/     pinned git submodules: hnswlib, googletest, yaml-cpp, LightGBM
├── CMakeLists.txt   C++ build
├── THIRD_PARTY.md   dependency licences and attribution, including the DARTH re-implementation note
└── CLAUDE.md        working rules and status log used during AI-assisted development sessions
```

| Path | What it holds |
|---|---|
| `apps/` | `ars_static_sweep` (Phase 1), `ars_oracle` (Phase 2), `ars_features` (Phase 3), `ars_eval_policy`, `ars_make_query_subset` (Phase 4b confirmation set), `ars_evolve` / `ars_check_state` (evolved states), `exp_c_darth` (DARTH verify / trace / eval) |
| `python/` | per-phase scripts (`train_router.py`, `phase4b_*.py`, `exp_c_*.py`, `phase5*_*.py`, `exp_e_*.py`) and shared libraries (`router_lib.py`, `router4b_lib.py`, `phase5_lib.py`, `exp_e_lib.py`); pinned dependencies in `python/requirements.txt` |
| `configs/` | every run is driven by a YAML file. `configs/phase4b/confirmation_set_v1_REJECTED.yaml` is kept, by name, as part of the audit trail |
| `derived/` | `sift1m_s0_effort_tiers.json`, `sift1m_s0_query_strata.csv`; `exp_c/darth_lgbm_s0.txt` and `darth_policy_s0.json` (seed 42) with a provenance README; `exp_e/` seed-43/44 DARTH models and policies |
| `splits/` | `sift1m_query_split_seed20261001.csv` (8,000/2,000), confirmation-v2 ids and tagging. `_invalid/` holds a rejected confirmation set (it contained copies of original queries) that must never be used |
| `docs/` | [`design_doc.md`](docs/design_doc.md) (source of truth, with Addenda A1, C1, E1); [`notes.md`](docs/notes.md) (chronological lab notebook: decisions, pre-declarations, stops, approvals); phase reports; audits ([`paper_readiness_audit.md`](docs/paper_readiness_audit.md), [`github_readiness_audit.md`](docs/github_readiness_audit.md), [`literature_verification.md`](docs/literature_verification.md)) |

`data/` and `results/` exist in the working copy but are git-ignored (see
[Reproducibility](#reproducibility)).

## 10. Build, dependencies and tests

### Environment used for the experiments
| Item | Version (recorded in `docs/notes.md` and run metadata) |
|---|---|
| OS / toolchain | Linux; GCC 13.3.0 (Ubuntu 24.04), CMake 3.28.3 (minimum 3.25), C++20, OpenMP 4.5 |
| Build | Release, `-O3 -march=native` (`ARS_NATIVE_ARCH=ON`) |
| CPU | AMD EPYC-Milan |
| Python | 3.12.3, with the pins in [`python/requirements.txt`](python/requirements.txt): numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.6.0, statsmodels 0.15.0, matplotlib 3.11.2, and others |

Submodules are pinned to exact commits:

| Submodule | Version | Commit |
|---|---|---|
| `third_party/hnswlib` | v0.8.0 | `3f3429661187e4c24a490a0f148fc6bc89042b3d` |
| `third_party/googletest` | v1.17.0 | `52eb8108c5bdec04579160ae17225d66034bd723` |
| `third_party/yaml-cpp` | 0.8.0 | `f7320141120f720aecc4c32be25586e7da9eb978` |
| `third_party/LightGBM` | v4.6.0 | `d02a01ac6f51d36c9e62388243bcb75c3b1b1774` |

**Hardware dependence.**
- The build uses `-march=native`, so hnswlib uses the host's SIMD kernels.
- Byte-identical index builds, evolved states and per-query results have been verified on the
  experiment machine only.
- On other hardware, expect results that should agree, and verify them against the recorded
  hashes. Bit-identity is not claimed.

### Building
Clone with submodules, including LightGBM's nested ones:

```bash
git clone --recursive <repository URL>
```

LightGBM must first be built as a static library. This is needed for the DARTH targets and
tests; without it they are skipped and everything else builds. The command is recorded in
`THIRD_PARTY.md` and `CMakeLists.txt`:

```bash
cmake -S third_party/LightGBM -B third_party/LightGBM/build -DBUILD_STATIC_LIB=ON \
      -DBUILD_CLI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/LightGBM/build -j
```

The experiment binaries were built in `build/` as a Release build, with clang-tidy disabled
(`-DARS_ENABLE_CLANG_TIDY=OFF`; see `docs/notes.md`). The available CMake options are:
- `ARS_BUILD_TESTS` (default ON);
- `ARS_ENABLE_CLANG_TIDY` (default ON);
- `ARS_NATIVE_ARCH` (default ON).

Python orchestration scripts invoke binaries as `./build/<name>`, so run them from the
repository root. A step-by-step build and reproduction guide will be added during release
preparation.

### Tests
- **C++:** GoogleTest suites in `tests/`, registered with CTest (`gtest_discover_tests`). The
  DARTH tests are built only when LightGBM is available.
- **Python:** pytest suites in `python/tests/`: router libraries, Phase 5 library, and 31
  synthetic Experiment E tests (unit, end-to-end fixture, determinism, failure handling).
- **Last recorded counts:** 68/68 gtests and 64/64 pytest (`docs/notes.md`, Experiment E
  implementation).

## Reproducibility

**In Git:**
- all C++ and Python source, with test suites;
- every experiment configuration;
- the design document with its pre-registration addenda, the lab notebook, reports and audits;
- frozen DARTH models and policies, frozen effort tiers and strata (`derived/`);
- the deterministic query splits (`splits/`);
- submodule pointers;
- the SIFT1M download script.

**Intentionally not in Git:**

| Excluded | Approx. size | Why |
|---|---|---|
| Raw SIFT1M | 551 MB | third-party dataset; obtained from the official source with `scripts/download_sift1m.sh`, which checks file sizes and headers |
| HNSW index files (S0 and 30 evolved states) | ~20 GB | regenerable deterministically (single-threaded builds; state files verified byte-identical on re-generation) |
| Ground-truth caches | ~2 GB | exact brute force; recomputed per state |
| DARTH training traces | ~5.3 GB | the frozen models in `derived/` are authoritative. A refit is functionally but not bitwise identical (prediction differences ≤ 2.2e-16; `derived/exp_c/README.md`) |
| `results/` (per-run outputs, row-level data, analysis JSON) | ~1.5 GB | to be published as a separate artifact bundle with SHA-256 manifests |
| `results/_invalid/`, `data/confirm/_invalid/` | — | quarantined aborted or rejected runs; never used for any result |

The full inventory and release plan are in
[`docs/github_readiness_audit.md`](docs/github_readiness_audit.md).

**Reproduction levels.**
1. **Analysis-only (planned).** A compact artifact bundle, to be published with the paper, will
   contain:
   - the Experiment E per-query rows (`results/exp_e_rows_20261005T110522Z/`);
   - the Phase 5′ analysis outputs it depends on (`results/phase5b_analysis_20261004T230023Z/`);
   - their manifests.

   Once that bundle is extracted at those paths, the Experiment E analysis entry point in
   `python/exp_e_analysis.py` takes the frozen plan and the rows directory:
   ```bash
   python python/exp_e_analysis.py configs/exp_e/experiment_e.yaml results/exp_e_rows_20261005T110522Z
   ```
   It verifies the frozen policy and model hashes before running, and writes a new
   `results/exp_e_analysis_<timestamp>/analysis.json`. On the experiment machine two runs
   produced byte-identical output (SHA-256 `442fdc78…`). **The bundle is not yet published, and
   no DOI exists yet.**
2. **Full pipeline.** This needs SIFT1M, the build above and substantial compute and disk (the
   evolved states alone are about 20 GB). The entry points are documented in each script's
   docstring, for example:
   - `python python/phase5b_run.py configs/phase5/sift1m.yaml`;
   - `python python/exp_e_run.py {prepare|precheck|run} configs/exp_e/experiment_e.yaml`.

   The complete ordered runbook is a known documentation gap, to be closed before public
   release.

## Provenance and frozen status

- **Pre-registration.** Each phase's gates, margins and analysis rules were written before its
  results existed:
  - `docs/notes.md` pre-declarations;
  - design-doc Addenda A1, C1 and E1;
  - the Experiment E design freeze.

  Deviations, stops and user approvals are logged in `docs/notes.md`.
- **Frozen objects.** Experiment E verified the policy and model hashes at every stage:

  | Seed | Policy | Model |
  |---|---|---|
  | 42 | `bcbe2720…` | `d291a2f1…` |
  | 43 | `6b2275ef…` | `9003ba8b…` |
  | 44 | `df0bd726…` | `21a27a77…` |

  The script, plan and design-freeze hashes are listed in `docs/notes.md` ("Experiment E — FINAL
  FREEZE"). They match the files in this repository's initial commit.
- **Determinism.**
  - Single-threaded index builds.
  - Seeded, stored query splits, checked on every run.
  - Byte-identical regeneration of all 30 evolved states.
  - Repeat DARTH evaluations byte-identical apart from timing columns.
  - The Experiment E analysis run twice with byte-identical output.
- **Commit attribution.** The experiments ran before the repository's first commit, so run
  metadata records `commit: none`. Provenance for the later phases rests on the recorded file
  hashes above.
- **Quarantine.** Invalid or aborted runs are kept apart in `_invalid/` directories, with
  READMEs, and are never used.

## Limitations

In addition to the scope in [Section 7](#7-scope-and-non-claims):
- **Dataset.** One dataset. The secondary higher-dimensional embedding dataset envisaged in the
  original design was **not** run.
- **Evolution.**
  - Insertion is capped at 8% by the available in-distribution pool.
  - States are nested prefixes of one fixed update order. Update-order sensitivity was not
    tested.
- **Reference.** The state-local reference is B1-style. Under insertion it stays at `ef` 50/53,
  so B1's insertion comparisons are close to mechanical. A state-local *recalibrated DARTH* was
  not evaluated.
- **H-E2.** DARTH's S0 pass cohort differs in composition from the reference's. This is a
  disclosed, uncorrected caveat.
- **Statistics.**
  - Seeds are fixed, not resampled.
  - Mixed-model trend fits sit on a boundary (seed variance ≈ 0).
  - The H-E3 p-values ignore predictor dependence.
- **Not performed.** The original design's recalibrated-router comparison, later phases in
  their original form, and graph edge-churn staleness measures were not performed. They should
  not be inferred from the design document.
- **Literature.** The literature check was targeted, not systematic. Some sources were
  verified from abstracts or metadata only.

## Citation

No citation file has been added yet. A `CITATION.cff` and the paper reference will be added when
the paper and artifact archive are available.

If you use this work before then, please cite the repository. Also cite the work it depends on:
- hnswlib (Malkov and Yashunin, HNSW);
- the SIFT1M / TEXMEX corpus (Jégou, Douze and Schmid);
- DARTH, if referring to the re-implemented mechanism.

## License

No license has been added to this repository yet, so no open-source license is currently
granted. A license will be added separately. Third-party components in `third_party/` remain
under their own licences (Apache-2.0, BSD-3-Clause, MIT); see [`THIRD_PARTY.md`](THIRD_PARTY.md).
