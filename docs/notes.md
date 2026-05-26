# Working notes

Decisions and deferred items. Anything that belongs in `docs/design_doc.md`'s
open-items section is staged here until that file is populated.

## Phase 0 — 2026-10-01

### Toolchain decision (to be copied into design_doc.md open-items)
- `g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`, CMake 3.28.3, git 2.43.0,
  OpenMP 4.5 via libgomp, 16 cores.
- **Decision: C++20.** GCC 13 supports it cleanly; project builds with zero
  warnings under `-Wall -Wextra -Wpedantic`.

### Pinned dependencies (git submodules)
- hnswlib `v0.8.0` (3f34296) — header-only, black box.
- googletest `v1.17.0` (52eb8108).

### Open items
- **clang-format / clang-tidy not installed.** Configs (`.clang-format`,
  `.clang-tidy`) are committed and CMake enables clang-tidy automatically once
  it is on PATH (`sudo apt install clang-format clang-tidy`). Proceeding without
  them per user instruction (2026-10-01).
- **`docs/design_doc.md` is incomplete** (as of 2026-10-01 it holds only the
  tail of §18 through §22; §1–§17 are missing). Phase 0 was checked against the
  §21 table row, which is present. Phase 1 needs §6.1, §10, §13 etc.
- **Secondary "modern embedding" dataset not yet named** in the available
  design-doc text, so no download script for it yet. Only SIFT1M is scripted.

### Phase 0 checkpoint (§21): MET
- Clean configure + build from scratch: 0 warnings, 0 errors.
- `ctest`: 4/4 pass (toolchain, OpenMP, hnswlib self-query, markDelete).
- `scripts/download_sift1m.sh`: SIFT1M downloaded to `data/sift/`, all four
  files verified by exact byte size + dim header; SHA-256 recorded in
  `data/sift/SHA256SUMS`; re-run is idempotent and re-verifies sums.

## Phase 1 — 2026-10-01

### Decisions
- **Distance-computation counting:** `ars::CountingL2Space` wraps hnswlib's
  own SIMD L2 function and increments a thread-local counter. hnswlib's
  built-in `metric_distance_computations` was rejected: it is one shared atomic
  (not per-query under threads) and adds the full neighbour-list size even for
  already-visited neighbours that are never evaluated. hnswlib itself is
  unmodified.
- **Ground truth is independent of hnswlib:** own `SquaredL2` (16 accumulator
  lanes, vectorises without -ffast-math). SIFT is integer-valued, so float sums
  are exact; ties are broken by smaller base id. Verified against the official
  `sift_groundtruth.ivecs` by `python/verify_ground_truth.py` (NumPy, float64).
- **Two recall columns:** `recall` (id-based, primary) and `recall_tie_aware`
  (ann-benchmarks distance threshold). SIFT1M contains duplicate vectors, so
  the second guards against id-based recall understating correct answers.
- **Single-threaded index build** (`build_threads: 1`). hnswlib's level RNG is
  consumed in insertion order, so a multi-threaded build is not reproducible
  even with a fixed seed. Test `SingleThreadedBuildIsReproducibleForFixedSeed`.
- **Search is single-threaded** in the sweep (§13 timing rule; also `setEf` is
  index-wide). Recall and distance counts are asserted identical across timing
  repeats — the run aborts otherwise.
- **`-march=native`** on by default (`ARS_NATIVE_ARCH`) so hnswlib uses its AVX
  kernels; effective flags are logged per run.
- **New pinned dependency: yaml-cpp 0.8.0** (submodule) — required by the
  YAML-config rule. Emits an upstream CMake <3.5 deprecation warning; harmless.
- **Python venv** at `.venv/`, deps in `python/requirements.txt`.
- **S0 ground truth and S0 index are cached** in `data/cache/`, keyed by
  file path+size+rows and parameters. The GT cache key includes `S0`; later
  states must compute their own (never reuse).
- **clang-format** applied to all project sources (v18.1.3, Google style).
- **clang-tidy** runs in a separate `build-tidy/` dir; the experiment binary is
  built in `build/` with `-DARS_ENABLE_CLANG_TIDY=OFF` (identical object code,
  and a tidy build cannot stall an experiment). CMake passes
  `--gcc-install-dir` so clang-tidy uses GCC 13's libstdc++: the VM has a
  `/usr/lib/gcc/x86_64-linux-gnu/14` runtime dir without libstdc++-14 headers,
  which clang otherwise picks. Fixed findings: `[[nodiscard]]` x4, C-array x1.
  **Still blocked:** clang cannot parse GCC's `omp.h`; it needs its own
  (`sudo apt install libomp-18-dev`).
- Fixed a Phase 0 metadata bug: build-type flags (`-O3 -DNDEBUG`) were not
  being logged (CMake variable name is upper-case).

### Deferred to later phases (do not act on now)
- **Train/test query split:** not created in Phase 1 — the sweep covers all
  10K SIFT queries and rows are per-query, so the split can be applied later.
  Must be fixed (seeded, written to a file) before Phase 2 oracle labels and
  before choosing the three fixed-ef tier values (§13: from training split only).
- **Phase 5:** an index loaded via `HnswIndex::Load` has its level RNG seeded by
  hnswlib's default (the load constructor does not take a seed), not our config
  seed. Deterministic, but not config-controlled — decide how to handle before
  inserting into a loaded index.
- **Phase 4+:** `HnswIndex::Search` sets the index-wide ef, so per-query routed
  ef must be evaluated single-threaded (or ef-grouped).
- **Phase 2 — oracle definition must handle ties/non-monotonicity.** On the
  first SIFT1M sweep, id-based per-query recall drops between consecutive ef
  values ~500 times in total, but every such drop is a swap between
  equidistant (duplicate) vectors: tie-aware recall is monotone in ef for all
  10K queries except one (a single 0.1 drop at one step). The oracle
  ("smallest ef reaching target recall") should therefore be defined on
  `recall_tie_aware`, or as "smallest ef such that every larger swept ef also
  meets the target". Decide and record this at the start of Phase 2.
- **Git:** repo has no commits yet, so every run's metadata records
  `commit: none, dirty: true`. An initial commit is needed for runs to be
  attributable.

### Phase 1 checkpoint (§21): MET for ground truth and search — 2026-10-01
Canonical run: `results/227a00cec9b6_20261001T193413Z/` (SIFT1M, 1M base, 10K
queries, M=16, efC=200, seed 42, single-threaded build in 554 s, 20 ef values,
3 timing repeats on an otherwise idle machine).
- **Ground truth independently verified:** `gt_verification.json` — vs the
  official TEXMEX `sift_groundtruth.ivecs`, 0 of 10,000 rows differ beyond exact
  distance ties (5,554 rows differ only in the order/choice of equidistant ids;
  top-10 set overlap 0.99944, all of it ties). Stored distances equal a NumPy
  float64 recomputation exactly (max abs error 0.0).
- **Recall approaches 1.0:** recall@10 at ef=1024 is 0.9992 (id) / 0.9999
  (tie-aware). The residual tie-aware misses are 6 queries at 0.9 each — index
  misses (ground truth is verified), as expected for HNSW.
- **Curve shape matches known SIFT1M/hnswlib behaviour:** recall@10 0.71 @
  ef=10, 0.90 @ 32, 0.96 @ 64, 0.99 @ 128, 0.997 @ 256. Distance computations
  and median latency are monotone in ef.
- **Determinism:** recall and distance counts identical across the 3 repeats
  within each run and row for row across two independent runs.
- An earlier run (`..._191636Z`) has invalid latency (concurrent load) — marked
  with `LATENCY_INVALID.txt`; its deterministic columns are identical.
- Per-query spread is wide (5th-percentile recall 0.8 at ef=64 vs mean 0.96),
  which is the heterogeneity the routing premise needs — confirmed properly by
  the Phase 2 oracle distribution check, not here.

**Outstanding before Phase 1 code is "final"** (design doc §19): clang-tidy
cannot analyse the two files that include `omp.h` until `libomp-18-dev` is
installed. All other project sources pass clang-tidy; all are clang-formatted.

## Phase 2 — 2026-10-01 (oracle search effort)

### Pre-phase check
Phase 1 checkpoint re-verified: 26/26 tests, SIFT1M SHA-256 OK, GT verification
PASS, S0 index + GT cached. Refactor of shared S0 setup re-verified to leave
Phase 1 output unchanged (smoke run: recall / tie-aware recall / distance
counts identical on all rows).

### Decisions (methodology)
- **Split** (§7.3, §13), fixed *before* any oracle search, and a function of
  (num_queries=10,000, test_size=2,000, seed=20261001) only:
  Fisher-Yates over ids 0..9999 with `std::mt19937_64(20261001)` and
  rejection-sampled bounded ints (`std::uniform_int_distribution` is
  implementation-defined, so it is not used); permutation positions 0..1999 =
  test. 8,000 train / 2,000 test (top of §10's ~1,000–2,000 range). Stored at
  `splits/sift1m_query_split_seed20261001.csv` (tracked; SHA-256
  e0e3c6faaa56f98367cab0f9baedfb6c8a9ab788543538221d546f0aa8b03802). Every run
  regenerates it and aborts if it differs from the stored file. Byte-identical
  to an independent Python MT19937-64 + Fisher-Yates reimplementation.
- **Recall definition:** tie-aware recall@10 (Phase 1 finding). Nothing in the
  design doc argues for id-based recall; with id-based recall the 1.0 target
  would censor 66 queries purely because of duplicate-vector ties (vs 2).
- **Target:** 0.95 — the value §7.7 gives; chosen from the design doc, not from
  data. NOTE: with k=10, per-query recall is a multiple of 0.1, so ">= 0.95"
  means 10/10. Flagged for approval.
- **Label rule:** oracle ef = smallest grid ef from which recall >= target holds
  at every larger grid ef ("stable reach"). `oracle_ef_first_reach` (first grid
  ef meeting target) is also stored; they differ for 1 of 10,000 queries.
  Censored (not reached at the largest grid ef) labels are left empty, with
  recall at max ef recorded.
- **ef grid:** 10..20 in +1 steps, then x1.05 (rounded), up to 4096: 121 values.
  Fixed by config, not tuned. ef=10 is a true floor (hnswlib uses max(ef, k)).
- **Search:** `HnswIndex::SearchBatch` sets ef once, then searches in parallel
  (16 threads). Verified identical to serial `Search()` (unit test, and 100,000
  rows vs the Phase 1 canonical run at the 10 shared ef values). No timing in
  Phase 2 — only deterministic recall and distance counts.
- **Per-query data kept:** `oracle_curves.csv` = full curve (recall, tie-aware
  recall, distance computations at all 121 ef values) for all 10,000 queries —
  1.21M rows, kept for routing-regret analysis and for re-deriving labels
  under any other target without re-searching.
- **Test-split labels exist** (needed later as the regret reference) but are
  only described, never used to choose anything. Router fitting (Phase 4) must
  filter `split == train`.
- **Not done in Phase 2** (needs a selection rule / approval): the three
  fixed-ef baseline values (§7.6), and difficulty-tertile assignment (§7.4).

### Phase 2 checkpoint (§21): MET — 2026-10-01
Canonical run: `results/8e605bfc5769_20261001T201040Z/` (final binary);
`oracle_report.json` + `oracle_distribution.png` from
`python/analyze_oracle.py <run> --phase1 results/227a00cec9b6_20261001T193413Z`.
All six checks pass: split, complete, rederive (independent Python
re-derivation: 0 mismatches), phase1 (100,000 rows identical), nondegen,
beats_fixed.
- Reproducible: a second full run (`2d04ed0e89a5_20261001T195242Z`, pre-tidy-fix
  binary) gives byte-identical curves and split, and identical labels.
- Train (8,000, 0 censored): oracle ef median 29, p10 10, p90 117, p99 327,
  max 2794; 92 distinct values; 17.3% at the ef=10 floor; mean distance
  computations at oracle 1,137.
- Test (2,000; 2 censored: queries 4659, 8318 — not 10/10 even at ef 4096):
  median 30, p90 111, max 679 — same distribution as train.
- Oracle vs fixed ef (train): fixed ef needs 2794 (24,430 dist. comps, 21.5x)
  to match the oracle's mean recall 1.0; at the oracle's budget (~1,137) the
  best fixed ef (56) gets 0.955. The 21.5x is driven by the single hardest
  train query (test: 7.2x, matching ef 679) — the budget view is the stabler
  summary.
- Sensitivity (descriptive, all queries; nothing chosen from it): tie-aware
  @0.9 -> median 16, p90 59, 0 censored; id-based @1.0 -> 66 censored.

### Open for user approval (Phase 2)
1. Target 0.95 at k=10 is effectively 10/10 per query (design says "for
   example, 0.95"). Keep, or use 0.9 (9/10)?
2. The 2 censored test queries: keep in the fixed test set (they cannot be
   removed — the query set is fixed) and treat oracle effort as censored at the
   max grid ef (4096) for regret, reported separately?
3. Selection rule for the three fixed-ef baselines (§7.6) — from the train
   split's S0 curve only — and the tertile tie rule (§7.4: ~19% of test queries
   share the floor label ef=10, so tertile boundaries hit ties).

## Phase 3 — 2026-10-02 (live routing features)

### Phase numbering note
The user's Phase 3 request covered features + router training/evaluation. In
the design doc (§21) features are Phase 3 and router training/evaluation is
Phase 4 ("must succeed before proceeding"). Following CLAUDE.md (design doc
wins; one phase at a time), this session does Phase 3 (features) plus the
training-only tier/baseline and tertile rules the user asked for, and stops
before router training to ask.

### User decisions recorded (2026-10-02, carried from the Phase 2 review)
- Oracle target stays 0.95 tie-aware recall@10.
- Censored test queries 4659, 8318 stay in the query set; for regret their
  oracle effort is right-censored at ef=4096 and reported separately (4096 is
  NOT their true required effort).
- The 21.5x fixed-vs-oracle figure is not a headline; use test-set and
  per-query analysis.
- Fixed-ef baselines and tertile thresholds come from the training split only,
  rule documented before it is applied to the test split.

### Feature definitions (fixed before computing them)
All distances Euclidean (sqrt of hnswlib squared L2).
- knn_dist = d_k of a probe search (k=10, ef=10 — the cheapest search hnswlib
  allows at k=10, since it uses max(ef, k)).
- centroid_dist = ||q - mean(indexed base vectors)||, mean in double.
- score_concentration = d_1 / d_k of the same probe (1 if d_k = 0).
- lid (ablation only) = Levina-Bickel MLE -1/mean_{i<k} ln(d_i/d_k) over a
  separate deeper probe (k=20, ef=20). Conventions: 0 if any d_i = 0; +inf if
  all equal. Neither occurred on SIFT1M.
- Probe distance counts are stored per query: a router pays the probe before
  its main search. Mean 391 (core probe) / 564 (LID probe) vs mean oracle
  effort 1,137 -> must be counted in router effort in Phase 4.

### Effort-tier / fixed-ef baseline rule (TRAINING SPLIT ONLY; documented
### before being applied to the test split)
Q(p) = smallest grid ef e with fraction of TRAIN queries having oracle_ef <= e
at least p (always an actual grid value).
- ef_low = Q(1/3), ef_med = Q(2/3): equal-mass thirds of train oracle effort.
- ef_high = Q(0.99): covers all but the hardest 1% of train queries; not the
  max (2794), which is set by one outlier.
- These three values are BOTH the fixed-ef baselines (§7.6) AND the router's
  output tiers (§9), so the router chooses per query among exactly the values
  the baselines apply globally. Frozen for all later states (§13).
- Tier label of a query = smallest tier value >= its oracle ef; oracle_ef >
  ef_high (or censored) -> "high", which fails the target even at that tier.

### Difficulty-tertile rule (§7.4; thresholds from TRAIN, applied to TEST)
- T1 = Q(1/3), T2 = Q(2/3) (same values as ef_low, ef_med: "easy" == the low
  tier suffices). easy: oracle_ef <= T1; medium: T1 < oracle_ef <= T2; hard:
  oracle_ef > T2 or censored. Ties at a threshold go to the lower group; group
  sizes are not forced equal.
- ef=10 pile-up: ef=10 is the grid floor, so oracle_ef=10 means "<= 10,
  unresolved". All floor queries are easy. If the pile-up exceeded 1/3, T1
  would equal 10 and easy would be exactly the floor pile — tied queries are
  never split across groups.
- Censored test queries (4659, 8318) are hard, and reported separately.

### Phase 3 checkpoint (§21): MET — 2026-10-02
Canonical run: `results/657980fb7c23_20261002T050617Z/` (`feature_report.json`,
`features_vs_oracle_train.png`), validated by `python/analyze_features.py`.
- No NaN/inf; no edge-case conventions triggered (0 LID=0/inf, 0 d_k=0, 0
  concentration=1). All features non-constant (>= 9,337 distinct values).
- Probe cost equals Phase 1 (k=10, ef=10) rows for all 10,000 queries;
  centroid_dist matches NumPy (max error 4e-13); probe d_k >= exact d_k for
  every query (equal for 17.7%, median overestimate 1.3%).
- TRAIN-only Spearman vs oracle ef (95% bootstrap CI): knn_dist +0.459
  [0.441, 0.478]; centroid_dist -0.445 [-0.463, -0.426]; score_concentration
  +0.363 [0.343, 0.383]; lid (ablation) +0.558 [0.542, 0.573]. All three core
  features meet the pre-set criterion (|rho| >= 0.2, CI excludes 0).
- Reproducible: second run identical (features.csv excl. experiment_id,
  centroid.csv).

### Tiers / baselines / tertiles (rule above; fitted on train, frozen)
`derived/sift1m_s0_effort_tiers.json`, per-query `derived/sift1m_s0_query_strata.csv`.
- Fixed-ef baselines = router tiers: low 18, med 48, high 327.
  Train: recall 0.823 / 0.945 / 0.999; dist comps 532 / 1,007 / 4,593.
- Tertile thresholds T1=18, T2=48. Train tier labels 2,674 / 2,741 / 2,585;
  test difficulty easy 683 / medium 641 / hard 676 (incl. 2 censored).
  72 train / 18 test queries need more than ef_high.
- Train-only upper bound: a PERFECT 3-tier router reaches recall 0.999 at
  2,202 search + 391 probe = 2,593 dist comps vs 4,593 for the cheapest fixed
  ef matching that recall (ef=327): 1.77x headroom with the probe, 2.09x
  without. Phase 4 is feasible but not guaranteed.

### For Phase 4 (do not act before approval)
- Router effort must include the probe cost (mean 391; LID ablation adds a
  564 probe).
- "Best fixed-ef baseline" needs a definition at matched recall (three
  baselines at very different recall levels), plus the per-query and
  per-stratum views the user asked for.

## Phase 4 — 2026-10-02 (initial router) — PRE-REGISTRATION
Written BEFORE any router was fitted and before any test-split evaluation.

### User decisions (2026-10-02)
Probe cost counts in router cost. Tiers frozen at 18/48/327. Target 0.95
tie-aware recall@10. Split 8,000/2,000 seed 20261001. Censored test queries
4659, 8318 kept, reported separately. Difficulty thresholds 18/48 frozen.

### Training data (TRAIN split only)
Inputs: knn_dist, centroid_dist, score_concentration (Phase 3 canonical run).
Target: tier label (low/med/high) from `derived/sift1m_s0_query_strata.csv`.
Evaluation inside training uses only the stored TRAIN oracle curves.

### Candidates (primary router, 3 core features)
- DT2, DT3, DT4: sklearn DecisionTreeClassifier(max_depth=2|3|4,
  min_samples_leaf=100, random_state=20261002). No scaling (trees invariant).
- LR: StandardScaler (fit on training folds only, inside a Pipeline) +
  multinomial LogisticRegression(C=1.0, max_iter=2000).
Simplicity order (fewest parameters first): DT2 < DT3 < LR < DT4.

### Selection rule (TRAIN only — 5-fold stratified CV)
StratifiedKFold(5, shuffle=True, random_state=20261002) on train tier labels.
Out-of-fold predictions -> per-query routed recall and total cost (core probe
+ search distance computations at the predicted tier ef, from train curves).
1. Constraint: OOF failure rate (recall < 0.95) must not exceed that of the
   cheapest fixed grid ef whose train mean recall >= the candidate's OOF mean
   recall. Candidates violating it are excluded.
2. Score: matched-recall cost ratio = cost of the train fixed-ef curve at the
   candidate's OOF mean recall (linear interpolation between grid points)
   / candidate's OOF mean total cost. Highest wins.
3. Ties: candidates within 0.02 of the best score -> the simplest one.
The winner is refit on all 8,000 train queries, serialised with its SHA-256,
and never refit again (it is the frozen static router for Phases 5+).

### Secondary routers (reported, never used to choose the primary)
- Single-feature threshold router (§9): DecisionTreeClassifier(
  max_leaf_nodes=3) on one core feature; the feature is chosen by the same CV
  score on train.
- LID ablation (§9): the selected primary configuration refit with
  knn_dist, centroid_dist, score_concentration, lid. Its cost counts BOTH
  probes (core + LID probe).

### Test evaluation (once, after the router is frozen)
Policies: router, fixed ef 18 / 48 / 327, oracle, matched fixed ef (cheapest
grid ef whose TEST mean recall >= router's TEST mean recall — chosen with
test data, which favours the baseline, i.e. conservative for the router),
plus the two secondary routers.
Per query x policy: recall@10 (tie-aware and id), search / probe / total
distance computations, failure (tie-aware recall < 0.95), difficulty group,
censored flag. Router regret: total and search-only distance computations
minus the oracle's; tier error = predicted - oracle tier (ordinal; > 0
conservative, < 0 aggressive). Censored queries: no regret (oracle effort is
right-censored at 4096); reported separately.
Statistics (§14): paired Wilcoxon signed-rank (zero_method "wilcox"), matched-
pairs rank-biserial effect size, 95% bootstrap CIs over test queries (2,000
resamples, seed 20261002) for means, failure rates and paired differences.

### Phase 4 checkpoint criterion ("router clearly outperforms the best fixed-ef
### baseline", §21) — on TEST, vs the matched-recall fixed ef:
(i)   router mean total cost (incl. probe) lower: paired bootstrap 95% CI of
      the cost difference entirely below 0, and Wilcoxon p < 0.01;
(ii)  relative saving >= 10%;
(iii) failure rate not worse: 95% CI upper bound of (router - matched fixed)
      failure-rate difference <= +0.01.
All three must hold. Dominance vs each of 18 / 48 / 327 is also reported.

### Phase 4 results — 2026-10-02 — CHECKPOINT NOT MET
Training run (train only; 0 test rows loaded): `results/ea8773feafbd_20261002T102325Z/`.
Test evaluation (run once, frozen models, SHA-256 verified):
`results/ea8773feafbd_20261002T102325Z/test_eval/`.

Selection (train 5-fold CV, as pre-registered): every candidate scored 0.33–
0.46 (fixed-ef cost / router cost at matched recall; > 1 would mean the router
wins). DT3 and LR infeasible (OOF failure rate above the matched fixed ef).
DT2 (0.419) and DT4 (0.417) within 0.02 -> DT2 (simplest). Single-feature
router: score_concentration. LID ablation: DT2 + lid.

Frozen DT2 router (uses knn_dist, centroid_dist; score_concentration unused):
  knn_dist <= 210.8 -> low (both centroid branches)
  knn_dist >  210.8 and centroid_dist <= 375.9 -> high; else -> med

Test (2,000 queries; router cost includes the 391 mean probe):
  policy        recall  fail   mean dc  median dc
  router        0.950   0.279  2,480   1,375
  fixed 18      0.824   0.659    527     528
  fixed 48      0.942   0.338  1,003   1,034
  fixed 327     0.999   0.009  4,543   4,796
  fixed 53 (matched recall)  0.950  0.302  1,078  1,114
  oracle        1.000   0.001  1,132     764   (2 censored at 4096)
  single-feature 0.957  0.246  2,767; LID ablation 0.960  0.239  3,478
- Matched recall: fixed ef 53 reaches the router's mean recall at 0.43x the
  router's cost (0.51x even excluding the probe). Router is dominated by the
  matched fixed ef (same recall p=0.68; +1,402 dc, CI [1,312, 1,493],
  Wilcoxon p=1.7e-184, rank-biserial 0.75). Failure-rate diff -0.023, CI
  [-0.047, +0.0005] (not significant).
- Pre-registered criteria: (i) FAIL, (ii) FAIL, (iii) pass -> NOT MET.
  The same conclusion holds at matched failure rate (router 27.9% vs ef 53
  30.2%, difference not significant) — the result is not a framing artefact.
- Tier accuracy 50.4% (chance 33%). Aggressive 27.3% (all fail: 100%),
  conservative 22.3% (0% fail, over-spend). Confusion (true x pred):
  low 394/213/76, med 211/273/157, high 85/250/341.
- Regret (1,998 uncensored): mean +1,368 dc incl. probe, CI [1,289, 1,451];
  median +522. Router met the target on 1,442 queries (mean regret +1,994);
  556 aggressive failures saved 648 dc on average, paid for in recall.
- Difficulty: easy — recall 1.0 but 1,485 dc vs 886 (fixed 48): 289/683
  over-routed. Medium — 32.9% fail (211 routed low) at 2,279 dc vs fixed 48
  at 0% / 1,013 (by construction fixed 48 suffices for easy+medium). Hard —
  51.3% fail at 3,676 dc vs fixed 327 at 2.7% / 5,179.
- Systematic misrouting: leaf "knn <= 210.8, centroid <= 402.8 -> low" (378
  queries): only 45% truly low, 54.8% fail — crowded neighbourhoods that are
  still hard. Leaf "-> high" (574): 41% did not need high, 5,852 mean dc.
  Leaf "-> med" (736): near-uniform true tiers (29/37/34%) — uninformative.
- Censored 4659, 8318: recall 0.9 under EVERY policy including ef 4096
  (one neighbour is never found); router routed them med / low. Both have
  unusually low score_concentration (0.66, 0.72 vs median ~0.88).
- Validation: C++ re-search of every routed query (router, oracle, LID
  ablation; 6,000 rows) equals the curve-lookup evaluation exactly; sklearn-free
  JSON exports reproduce all predictions; 55/55 gtests, 16/16 pytest.
- Interpretation: a PERFECT tier choice would win (train upper bound 1.77x),
  so the loss comes from classification quality: features with Spearman
  ~0.4 give 50% tier accuracy, and the tiers are far apart in cost (327 is
  ~4.5x 48), so each conservative error is expensive and each aggressive
  error is a guaranteed failure. Mixing tiers is also inherently costlier
  than one constant ef at equal mean recall unless routing is accurate.

### Phase 4 — needs user decision (design §21: "must succeed before proceeding")
Not acted on. The test split has now been used once; any revised router
should be selected on train CV only, and a second test evaluation must be
disclosed as such.

## Methodology audit — 2026-10-02
Full report: `docs/methodology_audit.md` (train-only diagnostics:
`python/audit_phase4_train.py` -> `results/audit_phase4_20261002T105613Z/`).
Summary: no implementation bug found; the Phase 4 numbers are correct. But the
checkpoint was unattainable by construction at the router's operating point
(R~0.95): even an omniscient 3-tier router with the additive probe costs 1,284
vs 1,082 for fixed ef (train). Critical: C1 (unattainable gate), C2 (tier
classification with symmetric loss misaligned with cost-at-recall; uncontrolled
operating point). Major: M1 tiers not cost-aware, M2 probe additive vs design
"reused", M3 per-query 10/10 labels vs mean-recall metric, M4 regret not at
matched recall, M5 the Phase 3 "1.77x headroom" note was computed at R=0.999
only (misleading). No experiment changed; awaiting user approval (audit §10).

## Phase 4 methodology validation (second-order audit) — 2026-10-02
Report: `docs/phase4_methodology_validation.md` (train only:
`python/validate_audit_train.py` -> `results/audit_validation_20261002T122651Z/`).
Independent pure-Python hnswlib replica + NumPy GT reproduced 20/20 search
distance counts and 5/5 policy totals. C1/C2 confirmed under both recall
contracts and query bootstrap (fixed - omniscient-3-tier+probe = -203 dc,
CI [-212, -192]). UNRESOLVED: no index-seed replicates (§13). Audit wording
corrections recorded there (V-4, V-6, V-7). No experiment changed.

## Phase 4 redesign — PRE-DECLARATION (2026-10-02)
Written BEFORE computing any per-query-contract feasibility numbers at the
operating points below. Evidence seen so far: mean-contract table (R = 0.952,
0.97, 0.99, 0.999) and one per-query sensitivity point (success 72.3%). No test
data used for any of this.
- Primary contract: per-query success = tie-aware recall@10 >= 0.95 (i.e.
  10/10; the user-locked oracle definition). System target: success RATE >= S*.
- Primary S* = 0.95 (the design's 0.95 convention applied at system level,
  i.e. "95% of queries meet the per-query target", a p95-style service
  target). Secondary operating points S* in {0.90, 0.99}; secondary contract:
  mean recall. Secondary results never decide the gate.
- Primary candidate efforts: data-independent doubling ladder (smallest grid
  ef >= 10 * 2^i, up to the grid max) plus "return the probe's own ef=10
  result" at zero extra cost. No data is used to place candidates.
- Primary cost: fully additive probe (what black-box hnswlib actually
  executes). Sensitivities: shared upper-layer descent; search-only bound.
- Feasibility rule: if the training-split perfect-information bound at S*
  (same candidates, primary accounting) shows < 20% saving vs the fixed ef
  meeting S*, the gate is declared NOT TESTABLE at S* and reported as such;
  S* is NOT moved.

## Phase 4 redesign proposal — 2026-10-02 (NOT approved, NOT implemented)
`docs/phase4_redesign.md`. Train-only evidence:
`results/redesign_evidence_20261002T125113Z/redesign_train.json` (seed 42).
Proposed: per-query success contract, S*=0.95; candidates {probe result, 20, 40,
83, 164, 327, 647, 1343, 2661} (doubling ladder, data-independent); additive
probe primary; per-candidate success models + cheapest-candidate rule with
train-CV tau; B1 = train-calibrated fixed ef (seed 42: 172), B2 = hindsight
fixed; 3 index seeds {42,43,44}; fresh 2,000-vector confirmation set from
sift_learn (seed 20261005). Pre-flight on seed 42: contract-oracle saving
48.1% >= 20% (frozen tiers would give 15.1%: not testable).

## Phase 4b — 2026-10-02 — STOPPED at the ex-ante feasibility rule (before any router fit)
Frozen contract: docs/phase4_redesign.md as amended by docs/phase4_bias_audit.md §10
(approved). Done so far (no router fitted, no evaluation set created/touched):
- Seeds 43, 44 built (configs/phase4b/*, identical to seed 42 except index.seed;
  single-threaded builds; distinct index files). Oracle curves (all 10,000 SIFT
  queries, 121-ef grid): seed 43 results/e795d8b6d6df_20261002T133032Z, seed 44
  results/3e2faa1921f8_20261002T133032Z. Features: seed 43
  results/94945b3661cb_20261002T140528Z, seed 44 results/1c816578a946_20261002T140458Z.
  Seed 42 = Phase 2/3 canonical runs.
- An earlier launch was aborted: a sed bug left index.seed = 42 in the 43/44 configs;
  both runs loaded the seed-42 index and were killed before completion; quarantined in
  results/_invalid/ (README there). Not used.
- Ex-ante feasibility (train only): results/phase4b_feasibility_20261002T140704Z/.
  Per-query (gate) family: OK at S* = 0.90 / 0.95 / 0.99 on all three seeds (ladder
  perfect-info saving +35-36% / +46-47% / +63% vs train B1 mix).
  Mean-recall family: R = 0.97, 0.99 OK; **R = 0.95: STOP_FOR_REVIEW on all seeds**
  (ladder +0.7..1.5% < 10% while full grid +10.5..11.0% >= 10%).
- Open implementation ambiguities listed in the stop report (decision-rule edge cases,
  LID / single-feature secondaries, mean-family mapping, B1 hash seed, strata
  thresholds for seeds 43/44, query-set protocol confirmation).

## Phase 5 REFRAME — design decision log (2026-10-02)
- DECISION (user-approved 2026-10-02): Phase 5 is reframed, not skipped. Instead of
  router degradation (blocked by the §21 Phase 4 condition after the Phase 4b negative
  result), Phase 5′ asks RQ5′: does the feature → oracle-effort relationship
  (knn_dist, centroid_dist, score_concentration; LID as ablation) stay stable as the
  index evolves?
- Recorded as design_doc.md Addendum A1 (appended; original text untouched). Reuses
  Phase 2/3 code on evolved states. Phase 4b artefacts are NOT reopened; no router is
  evaluated.
- Evidence used to set X = 0.05 (read-only, existing S0 data, all 10,000 queries):
  seed-to-seed |rho| spread <= 0.011; paired bootstrap SE of delta 0.0029-0.0048;
  minimum detectable margin at 80% power <= 0.014.
- Parameters marked [APPROVAL] in A1.11 are pending. No implementation until the
  addendum text is approved.

## Addendum A1 approved with amendments (2026-10-02)
- User approved A1 with two amendments, now applied in design_doc.md:
  (1) OOD = regionally concentrated real vectors (pool nearest-neighbour ball around a
  seeded-random anchor; seed 20261010); the coordinate-permutation proposal is withdrawn;
  (2) A1.10 limitations strengthened and marked as mandatory wording for every Phase 5'/6'
  report ("OOD" is a narrow construction; "stable within 1-8% insertion churn").
- Measured (label-free, pool only): regional OOD concentration fades with magnitude
  (distance ratio 0.67/0.71/0.78/0.97; overlap with ID 12/23/45/91% at 1/2/4/8%).
  OPEN, must be decided before real-data states: H2' at 1-4% only, 8% OOD descriptive
  (proposed).
- Order mandated by user: S0 rebuild byte-identity check first (all seeds), then
  synthetic validation.

## Phase 5' — S0 rebuild byte-identity check: PASS (2026-10-02)
A fresh single-threaded build (existing build path, scratch cache) is byte-identical to the
cached S0 for every seed: 42 4282e2e2…, 43 00e01b8a…, 44 3d0b74bb… (sizes equal). Features
computed from the rebuilt indexes are identical to the canonical Phase 3 features.
Rebuild-in-process is viable; snapshot/restore is not needed. Each insertion run will
additionally save and hash its in-process S0 before the first insert.

## Phase 5'(a) synthetic validation — PRE-DECLARATION (before any synthetic data exists)
Purpose (design §11, A1.9a): validate the state-production and measurement pipeline and the
§8 staleness measures where the truth is known by construction. Not used for any headline.

Data (seed 20261011):
- Gaussian mixture, d = 32, K = 20 components, component means ~ N(0, 10² I), unit
  isotropic covariance, float32.
- Base N0 = 50,000; ID insertion pool 100,000; queries 1,000; all from the same mixture.

Index: M = 16, efC = 200, single-threaded, seed 42. Ladder and router are not used.

States (identical constructions to A1.4):
- ID: nested random prefixes (seed 20261008) of 1/2/4/8% of N0 (500/1,000/2,000/4,000).
- OOD: anchor-ball (anchor seed 20261010; increments inserted in the 20261008 order).
- Deletion: markDelete of 2/8% of base (seed 20261009).

PASS criteria (all required):
- C1 Update fraction recorded = exact inserted (or deleted) count / N0.
- C2 Nested monotonicity (exact, every query): |top-10(S0) ∩ top-10(S_m)| is
  non-increasing in m within the ID, OOD and deletion trajectories.
- C3 Distributional drift: D = ||mean(inserted) − mean(base)||₂ / sqrt(tr Cov(base)).
  OOD > ID at every magnitude.
- C4 Ground truth: for every state, NumPy brute force over live vectors matches (all
  queries, up to ties); deleted labels never appear in GT or in any search result.
- C5 Pipeline: oracle labels and features exist for every state and query, with no
  NaN/Inf. The S0 state through the new state path is byte-identical to the legacy S0
  path.
- C6 Known-truth locality: for each OOD state, Spearman(query→anchor distance,
  neighbourhood-overlap decay) < 0 with a 95% bootstrap CI excluding 0.

Descriptive only (no threshold): local-density drift (relative change in knn_dist); ID
vs OOD contrasts; feature–oracle Spearman per state.

## Phase 5'(a) synthetic validation: PASS (2026-10-02)
Report: results/phase5_synthetic_validation_20261002T191139Z/report.json. Runs:
data/phase5/synthetic/states/run_manifest.json. All pre-declared checks C1–C6 pass
(seed 42), and none of the passes is trivial:
- C2: overlap falls (ID 10→9.29, OOD →9.63, deletion →9.26) with 0 monotonicity violations.
- C3: drift OOD 1.21 vs ID 0.01–0.05.
- C4: GT relative error ≤ 3.3e-7; 0 deleted labels returned in 30,000 results per deletion
  state.
- C5: S0 via the state path is identical to the legacy path.
- C6: OOD locality Spearman −0.38 to −0.40 (CI excludes 0); ID ≈ 0.

Caveat: the synthetic OOD stays concentrated (pool 100K vs at most 4K inserts), whereas the
SIFT OOD dilutes (≈ ID at 8%). The synthetic run validates the metrics and the pipeline,
not the SIFT OOD condition.

Code added: ars_evolve, ars_check_state, the state: config section, masked GT and centroid,
Resize / MarkDeleted, phase5_lib / phase5_run / phase5_checks. Tests: 64 gtests, 32 pytest.
Legacy S0 paths re-verified byte-identical (results/_regression_checks/).
NEXT: real-data states need the user's decision on the 8% OOD handling (A1.4 item 4 /
A1.11 item 2).

## Phase 5'(b) — 8% OOD decision recorded (2026-10-02, before any real-data state)
User decision, frozen in design_doc.md A1.4 item 4 / A1.11 item 2: all four magnitudes kept;
H2' (ID vs OOD) tested at 1/2/4% only; 8% OOD generated and used for H1'/H3'/index-change
analyses but descriptive only for ID-vs-OOD, labelled "≈ ID by construction", with the
mandatory sentence: "At 8% insertion magnitude, the available pool forces the regional-OOD
construction to overlap approximately 91% with the matched ID set, so this level does not
provide a clean ID-vs-OOD comparison."
Overlap re-measured with the actual pipeline orders (MT64 Fisher–Yates ID order, seed
20261008; same anchor = pool row 18830; same pool): 11.2 / 23.0 / 46.0 / 91.1% at 1/2/4/8%
(distance ratio 0.666 / 0.709 / 0.784 / 0.973). The A1.4 table's 12/23/45/91% came from the
feasibility probe, which used a NumPy permutation as a stand-in for the ID order. The
construction is unchanged; the "≈ 91%" wording holds.
Pool (A1.9 d), data/phase5/sift/pool_integrity.json: 87,788 vectors, identical to the
feasibility pool; 0 exact duplicates within the pool, with base, with queries, with
confirmation-v2. Label-free distance check (1,000 queries, seed 20261002): median
query→nearest-pool distance 213.7 vs 221.7 for a same-size base subsample (full base 196.3);
min 45.1 vs 31.9. Pool vectors are not unusually close to queries.

## Phase 5'(b) real-data — PRE-DECLARATION (written before any real-data state exists)
Plan config: configs/phase5/sift1m.yaml (A1.4 parameters verbatim). States per seed
(42/43/44): id_{10000,20000,40000,80000}, ood_{same}, del_{20000,80000}; 30 evolved states.
S0 per seed = the canonical runs (seed 42: Phase 2 8e605bfc5769 / Phase 3 657980fb7c23;
seeds 43/44: Phase 4b oracle e795d8b6d6df / 3e2faa1921f8, features 94945b3661cb /
1c816578a946). Read-only; nothing in Phase 4b is modified.

Validation (all must pass before any analysis; any failure = STOP and report):
- V1 provenance: data, pool, order and plan SHA-256 recorded; seeds as in A1.11.
- V2 S0 reproduction (A1.9 b): S0 through the new state path vs canonical, per seed:
  oracle_curves.csv byte-identical; oracle_labels.csv and features.csv identical after
  dropping the experiment_id column.
- V3 in-process rebuilt S0 byte-identical to the cached S0 in every insertion run (A1.9 c).
- V4 state sizes and counts: element count = 1,000,000 + inserted count (insert);
  deleted count = 20,000 / 80,000 (delete); update fraction exact.
- V5 nestedness: each insertion order is a prefix chain (smaller-state inserted vectors =
  prefix of the larger); deleted-label lists are prefixes; ID inserted vectors = pool rows
  of the ID order; OOD sets equal the anchor balls; per query, |top10(S0) ∩ top10(S_m)| is
  non-increasing in m within every trajectory (exact).
- V6 index reproducibility: ars_evolve re-run for every (seed, trajectory) into a separate
  directory; every state file SHA-256 identical.
- V7 ground truth (A1.9 e; stronger than the 200-query sample): NumPy float64 brute force
  over the live vectors for ALL 10,000 queries per state; top-10 sets equal up to float
  ties (rel. tol 1e-5), max relative top-100 distance error < 1e-4; deleted labels never in
  ground truth.
- V8 deleted-vector exclusion: ars_check_state on every state (ef 10/100/1000, all queries):
  0 deleted and 0 out-of-range labels returned; index deleted count matches.
- V9 feature reproducibility: ars_features re-run on all 30 states; features.csv identical
  (minus experiment_id).
- V10 oracle reproducibility: ars_oracle re-run on id_80000 (seed 42), ood_80000 (seed 43),
  del_80000 (seed 44); curves byte-identical, labels identical (minus experiment_id).
- V11 query-set identity: query_id 0..9999 in every output, query file hash identical;
  seed identity and config hash recorded per run; all cells complete, no NaN/Inf features.

Analysis (exactly A1.6/A1.7; nothing chosen after seeing results):
- Effort y = oracle_ef; censored (reached == 0) ranked above all finite values, tied.
- ρ_s(f) = Spearman (average ranks) over all 10,000 queries, per seed and state.
  Δ_s(f) = |ρ_s(f)| − |ρ_0(f)| with the same seed's S0.
- Pooled Δ for (feature, trajectory, magnitude) = mean over the 3 seeds. Bootstrap:
  2,000 resamples of query ids (seed 20261002, numpy default_rng), the same resample
  applied to every seed and state; 95% percentile CI.
- Classification per (feature, evolved state) on the POOLED Δ: STABLE (CI lb ≥ −0.05),
  DEGRADED (CI ub < −0.05), else INCONCLUSIVE. "Every evolved state" = the 10 pooled
  (trajectory, magnitude) states. A1.7 "never averaged away": per-seed Δ, CIs and
  classifications are computed the same way and reported. Any per-seed DEGRADED is listed
  explicitly next to the H1' verdict, and the pooled verdict is reported as conditional on it.
- H1' per core feature: supported if STABLE at all 10; refuted if DEGRADED at any; else
  inconclusive. Overall: supported only if all three core features are supported. LID uses
  the same rule as a secondary.
- H2': Δ_OOD,m − Δ_ID,m, pooled, paired bootstrap CI, two-sided, m ∈ {1,2,4}%. 8%:
  descriptive only, "≈ ID by construction".
- H3': pairwise differences of the pooled Δ among the core features (centroid − knn,
  centroid − concentration, knn − concentration) at the largest magnitude of each
  trajectory (id 8%, ood 8%, del 8%); paired CIs; no threshold.
- Trend: Spearman(log magnitude, pooled Δ) per trajectory and feature (descriptive).
- Descriptive: |ρ| and Δ; relative change Δ/|ρ_0|; per-query feature drift
  (|f_s − f_0| and the median relative change); per-query effort drift
  log2(y_s / y_0) over queries uncensored at both states (censored counts reported);
  difficulty strata = S0 effort of the same seed with the frozen 18/48 thresholds
  (ρ and Δ within each stratum, effort drift per stratum); staleness measures
  (neighbourhood-overlap decay, local-density drift, update fraction, distributional
  drift D as in C3). H4' is not tested here (Phase 7').
- Known labelling quirk: the `state` column in ars_oracle/ars_features outputs is "S0"
  for every run; the state identity comes from run_manifest.json. The combined row-level
  table written by the analysis carries the true state label.

## Phase 5'(b) — V2 deviation found during the run (2026-10-02, before any evolved-state result was looked at)
V2 as pre-declared (features.csv identical minus experiment_id, vs the canonical Phase 3 run
657980fb7c23_20261002T050617Z) FAILS for seed 42, for a format reason only:
- The current ars_features writes two extra columns, knn_dist_lid_probe and
  score_concentration_lid_probe. They were added on 2026-10-02 ~14:45 for the Phase 4b
  LID-ablation router; the S0 features were then re-run per seed (657980fb7c23_…T144828Z,
  94945b3661cb_…T144831Z, 1c816578a946_…T144834Z). The canonical runs named in the
  pre-declaration predate this and have 9 data columns.
- All 9 shared columns (incl. knn_dist, centroid_dist, score_concentration, lid, probe
  counts) are byte-identical to the Phase 3 canonical run. The full file is byte-identical
  to the Phase 4b-era re-run 657980fb7c23_20261002T144828Z.
- Oracle curves and labels: byte-identical to Phase 2 canonical (V2 oracle part PASS).
- Neither extra column is a Phase 5' feature. Not repaired silently: the literal V2 is
  reported as FAIL (format). The proposed resolution is reported to the user and needs
  their decision before any analysis: V2-features = shared columns identical to Phase 3
  canonical AND full file identical to the Phase 4b-era re-run.
State production continues meanwhile (no analysis is run).

## Phase 5'(b) state production COMPLETE; validation (2026-10-03)
Report: results/phase5b_validation_20261003T063010Z/report.json. Manifest:
data/phase5/sift/states/run_manifest.json (+ run_manifest_with_hashes.json).
Every check PASSES except the literal V2 features criterion (all 3 seeds): format only, as
described in "V2 deviation". For all seeds: oracle curves and labels identical to canonical;
the 9 shared feature columns identical; the full file identical to the Phase 4b-era re-run.
V3, V4, V5 (0 overlap-monotonicity violations), V6 (all 30 state hashes reproduce), V7
(NumPy GT, all 10,000 queries, exact), V7b (GT identical across seeds), V8 (0 deleted
labels returned), V9, V10, V11: PASS.
ALL_PASS = False solely because of the literal V2. Analysis NOT run, pending the user's
decision on the V2 resolution.

## Experiment C (Ada-ef actuator at S0) — STOPPED before implementation (2026-10-04)
Inspected: Ada-ef public repo github.com/chaozhang-cs/hnsw-ada-ef @ ed463f9993868f7ecc7c103920644e7f94abb377
(cloned into session scratch only; nothing vendored, built or run), and the paper (arXiv 2512.06636 /
SIGMOD'26, doi 10.1145/3786639).
Blocker (methodological / dataset compatibility): Ada-ef's ef-estimator is built on the
full-distance-list (FDL) distribution, which is derived only for inner product, cosine
similarity and cosine distance. Paper §6.2: "the Euclidean (L2) case remains open due to the
squared terms in its formulation". In the code, init_estimator() accepts only "cd" and "ipd",
and SquaredEuclideanDistanceEstimator is commented out. All published experiments are angular
(GloVe-100, DeepImage-96, MS MARCO, Cohere, LAION, synthetic); there are no SIFT/L2 numbers.
The project's S0 is SIFT1M under L2.
Further findings (not blocking on their own):
- Ada-ef is a patched hnswlib v0.8.0. It adds adaptiveSearchKnn / adaptiveSearchBaseLayerST,
  which collect the first 1+32+31*32 base-layer distances and then reset ef mid-search.
  Build/save/load are unchanged (additive diff), so the S0 file would load. But the actuator
  lives inside the search loop, against the CLAUDE.md black-box rule.
- Needs Eigen 3.4 and Boost.Math (header-only; not installed here).
- Calibration = 200 database-sampled queries + their GT; mean/covariance statistics; a
  score -> (ef, recall) table built by stepping ef from k, 1.5k upward (ef_upper_bound 5000);
  defaults quantile_step 1e-3, 5 bins, exponential decay, target 0.95.
- Governing docs project2_CLAUDE.md / docs/project2_design_doc.md were not found in this repo
  or on disk.
No code written; no DARTH switch; Phase 4b and A1 untouched. Awaiting the user's research decision.

## Experiment C — DARTH compatibility audit (2026-10-04; audit only, no code)
Sources: paper arXiv 2505.19001 (PACMMOD 3(4) art. 242, SIGMOD'26); official code
github.com/MChatzakis/DARTH @ 0d9bafcf31d1d79668bc71139fe93fa5e70b5185 (cloned into session
scratch only).
- Mechanism: inside the HNSW base-layer search, a LightGBM GBDT (100 trees, lr 0.1, seed 42)
  predicts the current recall@k from 11 search-trace features: nstep, ndis, ninserts,
  firstNN, closestNN, furthestNN (k-th), avg/var/median/p25/p75 of the current k-result
  distances. It is called every pi distance computations, pi = mpi + (ipi − mpi)(Rt − Rp);
  the search stops when Rp ≥ Rt. Heuristic ipi = dists_Rt/2, mpi = dists_Rt/10, where
  dists_Rt is the mean number of distances training queries need to reach Rt. Training: 10K
  learning-set queries, an observation logged every 1–2 distance computations, target =
  true recall at that moment.
- L2: native (paper "we use the Euclidean distance (L2)"; SIFT100M, GIST1M are L2).
  No SIFT1M; SIFT100M uses M = 32, efC = 500, efS = 500, k mostly 50. Results are reported
  as speedup vs plain FAISS search → no directly comparable published number for our
  SIFT1M / M16 / efC200 / k10 / hnswlib setup; gate A not available, gate B only.
- Code: the official implementation is inside FAISS's HNSW (different search loop:
  k-heap + efSearch candidate heap, stop after efSearch steps), not hnswlib. Not usable
  as-is under "hnswlib only".
- hnswlib route: v0.8.0 public API searchStopConditionClosest + BaseSearchStopCondition
  exposes per-candidate-pop (should_stop_search), per-distance (should_consider_candidate)
  and result add/remove callbacks → DARTH early termination is implementable as a
  stop-condition class with NO hnswlib modification. Deviations: (i) termination takes
  effect at the next candidate pop, so ≤ 1 node's remaining neighbour distances (≤ 31 at
  M0 = 32) are computed after the decision (result frozen at the decision point); (ii)
  hnswlib beam semantics replace FAISS's efSearch-step rule; the model is trained on our
  own traces, so it is internally consistent.
- Data conflict: sift_learn (the paper's training source) is consumed by the A1 insertion
  pool (87,788) + confirmation v2 (2,000); the remaining 10,212 rows are query copies or
  duplicates. No clean disjoint 10K learning-set sample exists → decision needed.
- Dependencies: LightGBM (C API for inference, not installed; Python lightgbm for training,
  not installed).
No implementation. Awaiting the user's decision.

## Experiment C — STOPPED at a published-mechanism ambiguity (2026-10-04, before any DARTH run)
Done: Addendum C1 written (design_doc.md); feasibility passes (C1.8: ≈ 18.6 M observation rows,
≈ 1 GB, minutes); LightGBM 4.6.0 (pip) + source build (third_party/LightGBM @ v4.6.0
d02a01ac, static lib, not yet wired into CMake); HnswIndex::SearchWithStopCondition added
(additive, public hnswlib API; not yet exercised); include/ars/darth.h interface drafted
(not compiled into any target).
AMBIGUITY (feature order at inference). DARTH reference code:
- training (notebooks_scripts/predictor_training.py) and ALL 588 published all_feats
  models: feature_names = step dists inserts first_nn_dist nn_dist furthest_dist avg_dist
  variance percentile_25 percentile_50 percentile_75;
- runtime (faiss/impl/DeclarativeRecall.cpp, DARTHPredictorHNSW::predict_recall) passes
  {nstep, ndis, inserts, first_nn, nn_dist, avg_dist, furthest_dist, variance, median,
  percentile_25, percentile_75} positionally to LGBM_BoosterPredictForMatSingleRow;
- the paper's Table 1 order (…, furthestNN, avg, var, med, perc25, perc75) matches neither.
Net effect in the reference runtime: avg ↔ furthest are swapped, and the p25 slot gets the
median while the p50 slot gets p25. Either code-faithful (misaligned) or intent-faithful
(aligned) inference is a choice; not made. Awaiting the user's decision.
Also noted for Experiment E (not blocking C): with lazily deleted nodes, the reference skips
the predictor check on a distance to a non-member. Through hnswlib's stop-condition hook a
deleted, non-considered candidate cannot be told apart from a live one, so check timing may
differ slightly on deletion states.

## Experiment C — DARTH policy FROZEN (2026-10-04, before the evaluation run)
- Decisions (user, 2026-10-04): independent re-implementation, no DARTH code copied
  (THIRD_PARTY.md); intent-faithful feature order (design_doc.md C1.9).
- Correctness (step 2): plain stop-condition search vs searchKnn on all 10,000 SIFT queries
  at ef 10/50/53/142 → 0 mismatches in ids, order, distances and distance counts
  (results/9cb179548eb1_20261004T202453Z). Two earlier verify runs FAILED:
  - results/9cb179548eb1_20261004T202241Z: 567–639 order-only mismatches/ef;
  - results/9cb179548eb1_20261004T202416Z: 68–81 id-set mismatches/ef after a first fix.
  Both were caused by my new wrapper's post-processing of equidistant (duplicate-vector)
  results, not by the search: the same ids, distances and counts throughout. Fixed by
  mirroring searchKnn's k-selection (the k entries popped last), then (distance, label)
  order. No search logic changed. gtests: 68/68 (4 new DARTH tests, incl. plain-mode
  equality with lazy deletions).
- Traces (train split only, 8,000 queries): results/02a1c5c22faa_20261004T202634Z;
  17,726,998 observations, 9 s; 0 search mismatches vs searchKnn at ef 142.
- Model: derived/exp_c/darth_lgbm_s0.txt sha256 d291a2f1f8cbd64f9fa2e744dd0cfc8bcc1b4261aba13915c88d1b94430eb041
  (LightGBM 4.6.0, frozen params, 22.5 s). An identical refit differs only in
  last-digit leaf values (multi-threaded summation): max |Δpred| 2.2e-16 on 354,540 rows,
  0 stop-decision changes. C++ runtime vs Python predictions: max |Δ| 2.2e-16 on 17,781
  rows (results/2aaf3176b512_20261004T211200Z) → feature order consistent.
- Policy: derived/exp_c/darth_policy_s0.json sha256 bcbe2720044b51edff1b493b956e413958f8503a6a76bfe86f52317cec8ac75d:
  ef_cap 142, Rt 0.95, k 10, dists_Rt 735.65 → ipi 367, mpi 73.
  Train report: results/exp_c_train_20261004T202826Z.
- Evaluation is run ONCE on the 2,000 test queries; a second identical run serves only as
  the determinism check (non-timing columns must be byte-identical).

## Experiment C (C1) RESULT — FAIL (2026-10-04)
Full report: docs/exp_c_report.md. Analysis: results/exp_c_eval_20261004T211328Z.
- Single frozen evaluation (2,000 test queries, S0 seed 42): DARTH 1,172 mean distance
  computations vs B1 1,057 → saving −11.0% (95% CI −13.0% to −8.9%). Wilcoxon p = 8.5e-5 in
  the B1-favouring direction. Recall 0.963 vs 0.948 (Δ +0.015, CI [0.012, 0.019]).
  G1–G3 fail, G4 passes → C1 FAIL.
- All 16 mechanism / correctness / provenance / determinism checks pass. Predictor: 8.0
  inferences/query, 0.094 ms/query (21% of DARTH wall time); wall 0.436 vs 0.074 ms
  (single-thread).
- Post-evaluation clang-format (whitespace only): rebuilt binary re-verified (0 mismatches;
  eval non-timing output identical, results/82bf1f070903_20261004T211444Z).
- No tuning, no rescue configuration. C1.6 freeze not triggered; Experiment E not started.
  Next research decision is the user's.

## C1 closure — repository / reproducibility (2026-10-04)
- C1 frozen as a negative result: model and policy hashes verified against the report;
  all 9 C1 run directories present; no Phase 2/3/4b/A1 artifact modified; gtests 68/68,
  pytest 33/33. No C1 result regenerated.
- Minimal changes: python/requirements.txt pins lightgbm==4.6.0; derived/exp_c/README.md
  records hashes and reproducibility (artifact authoritative; refit functionally but not
  bitwise reproducible; deterministic settings recorded for future models only);
  THIRD_PARTY.md states that LightGBM is not yet a registered submodule.
- Found: the staged gitlinks for third_party/hnswlib (ca426729…) and third_party/googletest
  (bc548fd2…) differ from the checked-out commits actually used (hnswlib v0.8.0 3f342966…,
  googletest 52eb8108…). The user must re-stage them before the first commit.

## Phase 5'(b) — V2 decision (user, 2026-10-04) and repository fixes
- V2 is APPROVED as pre-existing schema evolution, not a methodological exception. The
  literal features criterion fails only because Phase 4b added two LID-probe output columns
  to ars_features. The 9 shared Phase-3 columns are byte-identical, the full file equals
  the Phase-4b-era S0 re-run, and the oracle part is byte-identical. V2 passes iff all three
  hold (python/phase5b_checks.py; the literal result is still reported). No evolved state,
  feature or oracle output is regenerated.
- Repository: LightGBM registered as a submodule at v4.6.0 (d02a01ac; existing clone
  adopted, nothing re-downloaded or rebuilt). Staged gitlinks corrected to the commits used
  by all experiments: hnswlib 3f342966 (v0.8.0), googletest 52eb8108. python/requirements.txt
  keeps lightgbm==4.6.0. No experimental artifact touched.

## Phase 5'(b) RESULT — H1' SUPPORTED (2026-10-05)
Report: docs/phase5b_report.md. Validation results/phase5b_validation_20261004T225337Z
(ALL_PASS, V2 under the approved rule). Analysis results/phase5b_analysis_20261004T230023Z
(script unchanged since the pre-declaration).
- All 40 pooled cells and all 120 per-seed cells are STABLE (X = 0.05); H1' supported for
  every core feature and for LID. Tightest: centroid_dist at ood 4%, Δ −0.037, CI lb −0.045.
- H2': only centroid_dist differs ID vs OOD (more negative under OOD at 1/2/4%). 8% is
  descriptive (≈ ID by construction).
- H3': centroid_dist is the LEAST stable (opposite to the A1.6 expectation), though within X.
- Descriptive: per-query effort reshuffles (rank-corr with S0 0.84 at 8% ID/OOD; 52% of
  queries need more effort at id 8%) while the population signal is stable.
- Recommendation for E (not started): calibration staleness of S0-calibrated policies (B1
  primary, frozen C1 DARTH secondary) against their own contracts. Needs a new addendum
  and the user's approval.

## Experiment E — S0 DARTH replication, seeds 43/44 (2026-10-05). STOP before evolved states
Option A (user): the frozen C1 rules are re-applied to each seed's own S0 index and training
traces. Mechanical parameterisation (approved): exp_c_train.py / exp_c_eval.py take optional
config keys (model_out, expected_index_sha256_prefix, index_seed, s0_oracle_run,
expected_policy/model_sha256, eval_report_prefix). All defaults equal seed-42 C1. Seed-42
artifacts and configs are byte-identical, and re-running the seed-42 analysis reproduces
c1_report.json exactly (results/exp_c_eval_20261005T074149Z).
| | seed 43 | seed 44 |
|---|---|---|
| S0 index | hnsw_S0_449780e7eaf4f5c0.bin (00e01b8a…) | hnsw_S0_cc3c7fb3ab27c6b5.bin (3d0b74bb…) |
| cap rule → ef | 142 | 135 |
| verify (0 mismatches, 10k queries) | results/df8dfde1f760_20261005T074225Z | results/bf05bc39ad4a_20261005T074344Z |
| trace (train only) | results/e09db978145c_20261005T074234Z, 17,734,937 obs | results/240fb1b418d4_20261005T074354Z, 16,985,782 obs |
| dists_Rt → ipi/mpi | 740.72 → 370/74 | 727.38 → 363/72 |
| model sha256 | 9003ba8b65f789ed3437fd024449646a30fe0f6180bb8b7dc355691d936ee771 | 21a27a771a6de11b2ce6c971be12b022f62b515ba1eea3285a9cea73c572d17d |
| policy sha256 | 6b2275eff8fa732e3d4a3df90f777817f6e0a13a0fef830861fbac0d0449ac4a | df0bd726ba379ccebacc73b977ae1dcefe602f47a37da00736bed9ccc6513740 |
| C++ vs Py predictions | 2.2e-16 (results/96900cb335b5_…) | 2.2e-16 (results/92300faca2cb_…) |
| refit vs model | Δpred ≤ 2.2e-16, 0 stop changes | Δpred ≤ 1.1e-16, 0 stop changes |
| S0 eval / repeat | results/7c10f1776381_20261005T075033Z / 7ebba916c0c6_… | results/c18ba78bb1e3_20261005T075041Z / d1e3499a6e85_… |
| analysis (16/16 checks) | results/exp_e_darth_s0_seed43_eval_20261005T075041Z | results/exp_e_darth_s0_seed44_eval_20261005T075050Z |
S0 sanity (descriptive, test split): mean recall 0.9641 / 0.9625 (seed 42: 0.9633); saving vs
that seed's B1 −11.5% / −10.8% (seed 42: −11.0%); stopped 94.9% / 93.1%. These replicate C1.
LightGBM 4.6.0 (submodule d02a01ac), hnswlib v0.8.0 (3f342966). No evolved-state data accessed.

## Experiment E — implementation in progress (2026-10-05; NOTHING executed)
Design approved conceptually; docs/exp_e_design_freeze_v2.md still "DRAFT FOR APPROVAL", not
appended to design_doc.md. Done: statsmodels 0.15.0 (+ patsy 1.0.3, formulaic 1.2.2) installed and
pinned; python/exp_e_lib.py (pure functions §5–§11); python/exp_e_run.py (prepare / precheck /
run); configs/exp_e/experiment_e.yaml. Not yet written: python/exp_e_analysis.py, unit tests
(python/tests/test_exp_e.py). No E stage has been run (not even prepare/precheck); no evolved-state
outcome exists.

## Experiment E — implementation ready (2026-10-05; NOTHING executed on real data)
- python/exp_e_analysis.py: validate() (§16 hard failures) + pure analyze() (H-E1 D primary with
  R / additive ΔR / per-query relative descriptive and recall context; H-E2 cohorts, transitions,
  NF, E; H-E3 16 tests with bootstrap p and Holm, plus deletion and secondary descriptives;
  H-E4 DARTH id primary / tie-aware secondary with S0 changes; §11 MixedLM, robustness slope,
  deletion difference) + CLI loader (not run).
- python/exp_e_run.py fixes: DARTH rows get cost = distance_computations; verify_frozen()
  (policy and model hashes, B1 specs) at the start of every stage; run requires prepared
  configs; no analysis inside the runner.
- python/tests/test_exp_e.py: 31 synthetic tests (unit + end-to-end fixture + determinism +
  §16 failures). pytest 64/64, gtests 68/68.
- Degenerate H-E3 Spearman (constant input, point or replicate) → hard stop (no rule
  invented). H-E4 descriptive CIs count non-finite replicates rather than hiding them.
- Remaining before execution: user's final approval of the freeze (DRAFT → final, append
  Addendum E1); record the analysis script hash; then prepare → precheck → run → analysis
  (+ determinism re-run).

## Experiment E — FINAL FREEZE (2026-10-05, before any evolved-state run)
Design: docs/exp_e_design_freeze_v2.md (status FROZEN — FINAL), appended to docs/design_doc.md
as Addendum E1. Implementation hashes (sha256) recorded before execution (§17; the user
commits git):
- python/exp_e_lib.py        b32b6b233130246b892f55c9bb61cc7b7bef1fb3854144af7ac11845481f8a95
- python/exp_e_run.py        592b784d275898db7d8369eb14775982296546361db3727f09d402992ad63a1e
- python/exp_e_analysis.py   92f45f02eb893995ca549d9653a9e2c30df5f21655fb11c416f8116e05c5e6e1
- python/tests/test_exp_e.py 4427f8c6a96d47d7c1f5cd6377b659d88fc2e0808c1b184a085bcfa226d4d10f
- configs/exp_e/experiment_e.yaml c602f320b76ddae54b5dce2a17cb1f93000dabba8d0fc9c82e1ab3d2fc93922a
- docs/exp_e_design_freeze_v2.md  43571dfb3d45d638ba264f9310cbf03dadab91f5961c05f4847cb4221c3b5e8e
- python/router4b_lib.py d4a4613d…, python/router_lib.py 1af3e385…, build/exp_c_darth 6cea8d12…

## Experiment E — EXECUTED (2026-10-05). Report: docs/exp_e_report.md
- prepare (33 configs) → precheck ALL PASS (results/exp_e_precheck_20261005T105734Z) → run, 33
  cells, §16 pass (results/exp_e_rows_20261005T110522Z) → analysis twice, byte-identical
  (results/exp_e_analysis_20261005T110646Z, …110753Z). No methodological change after results.
- H-E1 and H-E2: STABLE at every state for B1 and DARTH, both families (STABLE-FAMILY ×4 each).
  DARTH per-seed INCONCLUSIVE for H-E2 at id 8% (s42, s44) and del 8% (s42); no STALE.
- H-E3: 10/16 associated after Holm (B1 D 4/4, B1 E 2/4, DARTH D 4/4, DARTH E 0/4); no structural
  measure beats update fraction (secondary).
- H-E4: bias −0.001 → ≤ +0.005, MACE 0.054 → ≤ 0.057; stopped fraction ≈ 0.94.
- Trends: DARTH D −0.53 pp/doubling, E +0.66 pp/doubling (CIs exclude 0); B1 flat. MixedLM
  convergence warnings reported, not acted on.
