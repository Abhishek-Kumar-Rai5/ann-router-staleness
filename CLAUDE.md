# Project 1 — Router Robustness Under Index Staleness

## Read this first, every session
`docs/design_doc.md` in this repo is the **single source of truth** for this
project — research question, scope, experimental design, metrics, validity
safeguards, and the phased plan. If anything in this file and the design doc
ever conflict, the design doc wins. Read it before doing anything, and
re-read the relevant section before starting each new phase.

## What this project is, in one paragraph
We are testing whether a per-query ANN search-effort router (HNSW/hnswlib),
calibrated once on a static index, stays accurate as that index evolves
through inserts and deletes — and whether a structural measure of index
change predicts routing failure better than raw update count does. This is
a research project, not a product. Every phase produces an artifact that
gets validated before the next phase starts.

## Hard scope boundaries — do not cross without asking first
- Index: **hnswlib only**. Do not swap in FAISS, DiskANN, or a custom HNSW.
- Do not modify or "improve" hnswlib's internal insert/delete/graph logic.
  It is used as a black box.
- Datasets: **SIFT1M (primary), one modern embedding set (secondary),
  synthetic Gaussian data (staleness-metric validation only)**. Do not add a
  third real dataset without explicit approval.
- No CUDA/GPU code unless I explicitly ask for it in a given session. If you
  think GPU would help somewhere, say so and wait — don't implement it.
- No neural-network router. Decision tree / logistic regression /
  single-feature threshold baseline only, per the design doc.
- Do not add new routing features beyond the ones listed in the design
  doc's Router section without asking — the feature set is deliberately
  minimal by design, not by oversight.
- Do not build a full online-learning / continuously-retraining router.
  Only "frozen static router" vs. "recalibrated-at-each-state router" as
  specified.

## Phase discipline — this is the most important rule
Work proceeds **one phase at a time**, in the order given in the design
doc's phased plan (Phase 0 → Phase 10). Before starting a new phase:
1. State which phase you are starting and its objective, in one line.
2. Confirm the previous phase's validation checkpoint (from the design doc)
   was actually met — don't assume, check it (e.g. re-run the test, show
   the number).
3. If a checkpoint was NOT met, stop and flag it. Do not quietly patch
   around a failed checkpoint and move forward.

Do not jump ahead and start implementing a later phase's work "while you're
in there" (e.g. don't write router code while still in Phase 1). If you
notice something from a later phase is relevant, note it in a comment or in
`docs/notes.md` and keep going on the current phase.

## Engineering conventions
- C++17 minimum, C++20 if the toolchain supports it cleanly (check
  `g++ --version` once at Phase 0 and record the decision in
  `docs/design_doc.md`'s open-items, not silently).
- CMake build, hnswlib as a git submodule.
- clang-tidy and clang-format on by default. GoogleTest unit tests enabled
  by default — every new C++ module gets at least basic correctness tests
  before being considered done, especially ground truth and index
  management (these are the two components the whole project depends on
  being correct).
- Python only for: router training, statistics, plotting, experiment
  orchestration. Everything performance- or correctness-critical (index,
  search, ground truth, feature extraction) is C++.
- OpenMP for parallelizing ground truth and the experiment sweep.

## Reproducibility — non-negotiable
- Every experiment run is driven by a YAML config file. No hardcoded
  experiment parameters in source.
- Every run logs: config used, git commit hash, random seeds, build
  flags/compiler version, into a metadata file alongside its results.
- Results are structured, row-level data (one row per query × state ×
  policy), not printed summaries.
- Seed everything. Never silently use a different seed between runs that
  are meant to be comparable.
- The frozen "static router" must be the literal same fitted object reused
  across all index states — never accidentally retrained.
- Ground truth must be recomputed fresh for every index state. Never reuse
  S0 ground truth for a later state.

## Validity rules (see design doc §13 for full list) — check these reflexively
- Same fixed query set across all index states. Never swap or resample it
  between conditions.
- Router train/test query splits never overlap.
- Router training uses only S0 training-split oracle labels — never labels
  from a later state.
- Fixed-ef baselines use identical values across every state, no per-state
  retuning.

## Current status
> Update this section at the end of every session so the next session
> picks up correctly. Keep it short: current phase, what's validated, what's
> next, any open decision still pending.

- **Phase 5' COMPLETE 2026-10-05: H1' SUPPORTED** (docs/phase5b_report.md): feature->effort
  Spearman stable (all cells STABLE, X=0.05) within 1-8% insertion churn / 2-8% lazy deletion;
  per-query effort reshuffles. V2 passed under the approved schema-evolution reading.
- **Experiment C: C1 FAILED 2026-10-04** (docs/exp_c_report.md; frozen negative result).
- **Experiment E EXECUTED 2026-10-05** (design_doc.md Addendum E1; docs/exp_e_report.md): H-E1/H-E2
  STABLE everywhere for frozen B1 and DARTH within 1-8% insertion / 2-8% lazy deletion; H-E3 10/16
  associated (Holm); H-E4 DARTH predictor stays near-unbiased among early-stopped queries. Next step: the user's decision.
- Phase: **4b COMPLETE (negative, audited; docs/phase4b_post_result_audit.md).**
  **Phase 5 REFRAMED (design_doc.md Addendum A1, approved + amended 2026-10-02):**
  RQ5' = stability of the feature -> oracle-effort relationship under index evolution
  (no router). S0 rebuild byte-identity PASS (all seeds); Phase 5'(a) synthetic
  validation PASS (C1–C6, results/phase5_synthetic_validation_20261002T191139Z).
  BLOCKED before real-data states on the user's decision: 8% OOD handling (~ID by
  construction). A1.10 limitation wording is mandatory in all 5'/6' reports. Phase 4b
  artefacts must never be modified.
- Validated so far: Phase 1 GT (`results/227a00cec9b6_20261001T193413Z/`);
  Phase 2 oracle (`results/8e605bfc5769_20261001T201040Z/`); Phase 3 features
  (`results/657980fb7c23_20261002T050617Z/`, all `analyze_features.py` checks
  pass); Phase 4 train `results/ea8773feafbd_20261002T102325Z/` + `test_eval/`;
  55/55 gtests, 16/16 pytest (`python/tests`); smoke configs
  `configs/smoke_{sift,oracle,features}.yaml`.
- Fixed query split: `splits/sift1m_query_split_seed20261001.csv` (seed
  20261001, 8,000 train / 2,000 test). Never regenerate with another seed.
  Router fitting must use `split == train` rows only.
- Frozen from TRAIN only: fixed-ef baselines = router tiers 18/48/327;
  tertile thresholds 18/48 (`derived/sift1m_s0_effort_tiers.json`).
  Censored test queries 4659, 8318: right-censored at 4096, reported apart.
- Other open items: clang-tidy can't check the 3 files including `omp.h`
  (needs `sudo apt install libomp-18-dev`); repo has no commits (user does git
  themselves); secondary dataset not named; level RNG after `HnswIndex::Load`
  not config-seeded (Phase 5); graph edge-churn (§8) deferred to Phase 7.
- Builds: `build/` = experiments (tidy OFF); `build-tidy/` = static analysis.

## When in doubt
Stop and ask, rather than guessing and proceeding. Scope creep — adding
"one more useful feature," "one more dataset," "just quickly trying GPU" —
is the main risk to this project, not technical difficulty. Staying inside
the design doc's boundaries is itself part of what makes this a credible
research artifact.
