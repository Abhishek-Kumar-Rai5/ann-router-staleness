# GitHub / Reproducibility Readiness Audit

Date: 2026-10-05. Status: **READ-ONLY AUDIT.** This file is the only file created. Nothing
was rerun, moved, renamed, deleted or edited. No README, LICENSE or CITATION file was
created. All git operations (commits, pushes, tags) are done by the user; this audit only
recommends.

Evidence was gathered with read-only commands: `git status --porcelain --ignored`,
`git submodule status`, `cat .gitignore`, `du`, `ls`, `find`, `sha256sum`, and reads of
run `metadata.json` files, manifests and docs. Hashes quoted below were either recomputed
during this audit (marked "recomputed") or copied from existing manifests (marked with the
manifest's path).

---

## 0. Headline findings

1. **The repository has no commits.**
   - `git rev-parse HEAD` reports an unknown revision.
   - Only `.gitmodules` and the four submodule pointers are staged. Everything else is
     untracked.
2. **No result can be tied to a git commit.**
   - All **272** `results/*/metadata.json` files record `"git": {"commit": "none", "dirty": true}`.
   - This is the biggest provenance gap.
   - It is partly mitigated for Phase 4b, Phase 5′ and Experiment E, whose manifests record
     SHA-256 hashes of the scripts, plan, design document and binary used.
3. **Every artifact behind a reported number is git-ignored.**
   - `.gitignore` excludes `results/` (1.6 GB) and `data/` (47 GB) wholesale.
   - The frozen models and policies in `derived/` and the query splits in `splits/` are
     *not* ignored. That is good.
4. **No top-level user documentation exists**: no README, LICENSE, CITATION, build/run
   guide, or environment lockfile.
   - Execution commands are scattered across script docstrings, `THIRD_PARTY.md`,
     `CMakeLists.txt` comments and `derived/exp_c/README.md`.
5. **The build uses `-march=native` (`ARS_NATIVE_ARCH=ON`).**
   - The host CPU was an AMD EPYC-Milan.
   - Bit-identical HNSW graphs, distance-computation counts and recall rows are verified
     only on this machine.
   - Reproduction on other CPUs must be described as "expected equivalent, verify against
     hashes", not as "guaranteed bit-identical".
6. **The design doc's planned scope is broader than what was executed.** The secondary
   embedding dataset, the recalibrated router, the Phase 6–10 plan and the graph
   edge-churn metric were **not** performed. The repository and README must not imply
   that they exist (§13).

---

## 1. Current repository inventory

### 1.1 Git state
| Item | State |
|---|---|
| Branch | `master` (no commits; the default branch name used for PRs is `main`) |
| Staged | `.gitmodules`, `third_party/{hnswlib,googletest,yaml-cpp,LightGBM}` (gitlinks) |
| Untracked (`??`) | `.clang-format`, `.clang-tidy`, `.gitignore`, `CLAUDE.md`, `CMakeLists.txt`, `THIRD_PARTY.md`, `apps/`, `configs/`, `derived/`, `docs/`, `include/`, `python/`, `scripts/`, `splits/`, `src/`, `tests/` |
| Ignored (`!!`) | `.pytest_cache/`, `.venv/` (543 MB), `build/` (40 MB), `build-tidy/` (7.3 MB), `data/` (47 GB), `python/__pycache__/`, `python/tests/__pycache__/`, `results/` (1.6 GB) |

`.gitignore` contents (complete):
- build: `build/`, `build-*/`;
- tooling: `.cache/`, `compile_commands.json`;
- Python: `__pycache__/`, `*.pyc`, `.venv/`;
- experiments: `data/`, `results/`.

### 1.2 Categories
| Category | Location | Contents (verified) |
|---|---|---|
| C++ library source | `src/` (13 .cpp), `include/ars/` (13 .h) | config, vecs I/O, ground truth, HNSW wrapper, oracle, features, metrics, policy, query split, run metadata, S0 setup, DARTH |
| C++ experiment drivers | `apps/` (8) | `ars_static_sweep`, `ars_oracle`, `ars_features`, `ars_eval_policy`, `ars_evolve`, `ars_check_state`, `ars_make_query_subset`, `exp_c_darth` |
| Python orchestration/analysis | `python/` (33 .py + `requirements.txt`) | phase 1–4 analysis, router training/eval, Phase 4b, Phase 5 / 5′, Exp C, Exp E |
| Tests | `tests/` (9 gtest files + CMakeLists), `python/tests/` (4 pytest files) | last recorded counts are **68/68 gtests and 64/64 pytest** (`docs/notes.md`). Not rerun in this audit |
| Configs | `configs/` | phase 1–4 YAMLs; `phase4b/` (incl. `confirmation_set_v1_REJECTED.yaml`); `phase5/{sift1m,synthetic}.yaml`; `exp_c/`; `exp_e/` (plan, 4 DARTH S0 seed configs, 33 cell configs); 3 smoke configs |
| Frozen models/policies | `derived/` | `sift1m_s0_effort_tiers.json`, `sift1m_s0_query_strata.csv`; `exp_c/darth_{lgbm_s0.txt,policy_s0.json}` + README; `exp_e/darth_{lgbm,policy}_s0_seed{43,44}.*` |
| Splits / query sets | `splits/` | the canonical split `sift1m_query_split_seed20261001.csv`; confirmation v2 ids and tagging; `_invalid/` (v1, rejected) |
| Documentation | `docs/` (15 .md), `CLAUDE.md`, `THIRD_PARTY.md` | design doc + addenda, notes, phase reports, audits, Exp E freeze, literature verification |
| Scripts | `scripts/download_sift1m.sh` | SIFT1M download + size/dim check + SHA-256 record/verify |
| Submodules | `third_party/` (157 MB) | hnswlib, googletest, yaml-cpp, LightGBM |
| Raw datasets | `data/sift/` (551 MB) | the 4 TEXMEX SIFT1M files + `SHA256SUMS` (ignored) |
| Derived data | `data/confirm/` (2 MB), `data/phase5/` (39 GB), `data/cache/` (2.2 GB), `data/exp_c/` (1.8 GB), `data/exp_e/` (3.4 GB) | confirmation vectors; Phase 5′ pool, state indexes (+ byte-repro copies), synthetic data; ground-truth caches and S0 indexes; DARTH training traces and refits |
| Generated results | `results/` (323 entries, ~1.5 GB excluding `_invalid`) | per-run dirs `<hash>_<UTC>` and named dirs; `_invalid/` (61 MB), `_regression_checks/`, `_determinism_checks/` |
| Manifests/hashes | see §4 | `results/phase4b_final_manifest.json`; `phase4b_train_*/manifest.json`; `data/phase5/sift/states/run_manifest_with_hashes.json`; `results/exp_e_rows_*/metadata.json`; `derived/exp_c/README.md`; policy JSONs; `data/sift/SHA256SUMS` |
| Logs | `data/*.log`, `data/exp_e_analysis{1,2}.err` (empty), `data/phase5/sift/states/*.log`, `results/phase*_logs/`, `results/*.txt` | runtime logs |
| Figures | 10 `.png` under `results/` and `data/` | e.g. `recall_vs_ef.png`, `oracle_distribution.png`, `delta_vs_magnitude.png`. **No paper figures exist yet** |
| Paper source/PDF | none | not started |
| README / LICENSE / CITATION | none | not created (by instruction) |

---

## 2. What must be public

Labels: **KEEP IN REPO** · **RELEASE SEPARATELY** · **DO NOT RELEASE** · **GENERATE BEFORE
RELEASE** · **DOCUMENT ONLY**.

| Category | Paths | Classification | Reason |
|---|---|---|---|
| C++ source, headers, apps | `src/`, `include/`, `apps/` | KEEP IN REPO | produce every number |
| Build system, lint config | `CMakeLists.txt`, `tests/CMakeLists.txt`, `.clang-format`, `.clang-tidy` | KEEP IN REPO | |
| Experiment / analysis Python | `python/*.py` | KEEP IN REPO | all of them, including Phase 4 and 4b scripts that led to negative results; the negative results are reported |
| Python dependency pins | `python/requirements.txt` | KEEP IN REPO + GENERATE a full lock (`pip freeze`: 30 packages vs 12 top-level pins) | transitive versions are currently unpinned |
| Test suites | `tests/`, `python/tests/` | KEEP IN REPO | |
| Configs | `configs/**` incl. `phase4b/confirmation_set_v1_REJECTED.yaml` | KEEP IN REPO | the rejected v1 config is part of the audit trail; its name already marks it |
| Frozen policies / models | `derived/**` | KEEP IN REPO | small text/JSON (≈1.4 MB); authoritative for Exp C/E and the effort tiers |
| Phase 4/4b router models | `results/ea8773feafbd_*/model_*.{joblib,json}`, `results/phase4b_train_311634ee3011_20261002T145649Z/{models,routers}/` | RELEASE SEPARATELY (in the results archive) | inside `results/`. Note: `.joblib` files are pickles that run code on load and depend on the scikit-learn version. The JSON exports (verified to reproduce predictions, per `docs/notes.md`) should be the documented primary form |
| Query split, confirmation ids/tagging | `splits/*.csv` (216 KB) | KEEP IN REPO | needed by every phase |
| `splits/_invalid/` | | KEEP IN REPO **or** DOCUMENT ONLY (user decision) | small; has its own README. It must never be used |
| Design doc, reports, audits, freeze | `docs/*.md` | KEEP IN REPO | the pre-registration trail (design doc addenda, Exp E freeze v2) is a core credibility asset |
| `docs/notes.md` | | KEEP IN REPO | chronological lab notebook. It documents stops, failures and approvals |
| `CLAUDE.md` | | KEEP IN REPO (user decision) or DOCUMENT ONLY | agent instructions, not user documentation. Its "Current status" holds valuable history. Releasing it is transparent but needs a one-line explanation |
| `THIRD_PARTY.md` | | KEEP IN REPO | attribution (DARTH re-implementation, libraries) |
| Submodules | `third_party/*` | KEEP IN REPO (as gitlinks) | pinned commits (§7) |
| Download script | `scripts/download_sift1m.sh` | KEEP IN REPO | |
| SIFT1M SHA-256 sums | `data/sift/SHA256SUMS` (ignored) | GENERATE BEFORE RELEASE: copy into a tracked manifest | currently written locally on first download, so a new user's download records *their* hashes instead of checking ours (§6) |
| Per-query result rows | e.g. `results/exp_e_rows_*/rows.csv.gz` (3.7 MB), `results/phase5b_analysis_*/rows_query_state_seed.csv.gz`, `results/exp_c_eval_*/per_query_darth_vs_b1.csv`, `results/phase4b_eval_*/per_query_rows.csv` | RELEASE SEPARATELY | the evidence behind every table. Modest size |
| Analysis outputs (JSON / CSV) | `results/*analysis*`, `*report*.json`, `tables.md` | RELEASE SEPARATELY (plus hashes in a tracked manifest) | |
| Canonical per-run dirs (oracle, features, GT sweep) | e.g. `results/8e605bfc5769_*` (30 MB each), `results/227a00cec9b6_*` (14 MB) | RELEASE SEPARATELY | the inputs to later phases |
| Ground-truth caches | `data/cache/gt_*.{ivecs,fvecs}` (2.2 GB including indexes) | DOCUMENT ONLY (hash manifest); optional archive | regenerable deterministically by exact brute force |
| HNSW index binaries | `data/cache/hnsw_S0_*.bin`, `data/phase5/sift/states/**/*.bin` (20 GB) | DOCUMENT ONLY (hash manifest); optional archive | regenerable (single-threaded builds; V6 byte-identical repro). Too large for git |
| State reproduction copies | `data/phase5/sift/states_repro/` (20 GB) | DO NOT RELEASE | duplicates kept only for the V6 byte-identity check. The check's outcome is recorded in the validation report |
| Phase 5′ insertion pool + inserted vectors | `data/phase5/sift/pool.fvecs` (44 MB), `*_inserted.fvecs`, `pool_rows.csv`, `pool_integrity.json` | RELEASE SEPARATELY (pool + integrity file) or DOCUMENT ONLY (regenerated by `phase5b_run.py`, checked against the stored file) | derived from SIFT data; same redistribution caveat as §6 |
| Synthetic Phase 5′(a) data | `data/phase5/synthetic/` (179 MB) | DOCUMENT ONLY (regenerable from config) + hash manifest | |
| DARTH training traces | `data/exp_c/trace_train.f64` (1.8 GB), `data/exp_e/*/trace_train.f64` (1.8 + 1.7 GB) | DOCUMENT ONLY (hash in policy JSON); optional archive | the frozen model is authoritative. A refit is functionally but not bitwise equal (`derived/exp_c/README.md`) |
| Refit models | `data/exp_c/darth_lgbm_s0_refit.txt`, `data/exp_e/*/darth_lgbm_s0_refit.txt` | RELEASE SEPARATELY (small) or DOCUMENT ONLY | evidence for the non-bitwise-refit statement |
| Invalid runs | `results/_invalid/`, `data/confirm/_invalid/` | DO NOT RELEASE (contents); DOCUMENT ONLY (existence + `README.txt` text) | must never be used. Mentioning that they exist keeps the record honest |
| Dry-run / determinism / regression checks | `results/_dryrun_*`, `results/_determinism_checks/`, `results/_regression_checks/` | RELEASE SEPARATELY (determinism and regression checks support claims); dry-runs DO NOT RELEASE | |
| Superseded / non-canonical run dirs | the many `<hash>_<UTC>` dirs not referenced by any manifest or report | user decision. **Recommended: include in the full archive with an index marking canonical vs. superseded** | deleting them would hide history; listing them without a label invites misuse |
| Logs | `data/*.log`, `results/phase4b_logs/`, `results/*_run.txt` | RELEASE SEPARATELY (small) | |
| Build trees, venv, caches | `build/`, `build-tidy/`, `.venv/`, `__pycache__/`, `.pytest_cache/` | DO NOT RELEASE | environment-specific |
| Figures | none for the paper yet | GENERATE BEFORE RELEASE | paper figures should be produced by a tracked script from released artifacts |
| Paper source / PDF | none | GENERATE BEFORE RELEASE | |
| README, LICENSE, CITATION.cff, artifact index, runbook | none | GENERATE BEFORE RELEASE | |

---

## 3. .gitignore audit

| Ignored pattern | What it hides that matters | Recommendation |
|---|---|---|
| `results/` | **All** canonical result dirs, row-level data, analysis JSON, Phase 4/4b models, manifests (`phase4b_final_manifest.json`), validation reports | **B + C.** Package canonical results separately and track a small manifest (path + SHA-256 + size + role) in git. Optionally commit a curated `results/` subset of a few MB (the Exp E rows and analysis, Phase 5′ analysis). If so, change the ignore rule to an allow-list. That change is a user decision and is not made here |
| `data/` | SIFT1M (third-party), `SHA256SUMS`, the confirmation vectors, Phase 5′ pool and states, GT caches, DARTH traces, state manifests (`data/phase5/sift/states/run_manifest_with_hashes.json` holds the per-state **index hashes** and run-dir mapping) | **SIFT1M: D** (exclude; download + verify). **`SHA256SUMS` and the state manifests: C** (copy into a tracked manifest). **Indexes, GT, traces: C** (+ optional B). **Confirmation vectors and pool: C** (+ B, small) |
| model directories | there is no model-specific ignore. Frozen models live in `derived/`, which is **not** ignored. Phase 4/4b models live under `results/` and are therefore ignored | `derived/` → **A**. Phase 4/4b models → **B** with **C** |
| generated artifacts | `build/`, `build-*/`, `compile_commands.json` | **D**, correct |
| logs | logs live in `data/` and `results/`, so they are ignored implicitly | **B** (in archives) |
| caches | `.cache/`, `__pycache__/`, `*.pyc`, `.venv/` | **D**, correct. `.pytest_cache/` is ignored only via the global status (it is shown as `!!`); consider an explicit entry |

**Risk:** `git add .` today would commit no results at all. Without a separate artifact
package, a cloned repository could not verify a single reported number.

---

## 4. Experiment → artifact traceability

"Git tracked?" refers to the intended state after the first commit. **Today nothing is
tracked, because there are no commits.** "Ignored" means it would be excluded by
`.gitignore` even after commit.

| Experiment | Main result | Source artifact | Current location | Git tracked? | Release requirement |
|---|---|---|---|---|---|
| Phase 1 | Exact GT verified against official SIFT1M GT; recall–ef curve | `gt_verification.json`, `results.csv`, `sweep_summary.csv`, `metadata.json` | `results/227a00cec9b6_20261001T193413Z/`; GT `data/cache/gt_S0_01564caeeb173bc1.*`; index `data/cache/hnsw_S0_c86ffb2a63e7325b.bin` (sha256 `4282e2e2…dc8cc5`, recomputed) | Ignored | archive run dir; hash manifest for GT + index |
| Phase 2 | Per-query oracle effort labels (S0, seed 42) | `oracle_labels.csv`, `oracle_curves.csv`, `oracle_report.json`, `split.csv` | `results/8e605bfc5769_20261001T201040Z/` | Ignored | archive |
| Phase 3 | Query features; `analyze_features.py` checks pass | `features.csv`, `feature_report.json` | `results/657980fb7c23_20261002T050617Z/` | Ignored | archive |
| Phase 4 | Router checkpoint **NOT MET** (`docs/notes.md` §"Phase 4 results") | model files, `oof_predictions.csv`, `test_eval/eval_report.json`; audits `results/audit_phase4_20261002T105613Z/`, `results/audit_validation_20261002T122651Z/`, `results/redesign_evidence_20261002T125113Z/` | `results/ea8773feafbd_20261002T102325Z/` (+ audit dirs); tiers `derived/sift1m_s0_effort_tiers.json` (sha256 `0909c3aa…7230`, recomputed) | Ignored (results); tracked (derived) | archive; tiers in repo |
| Phase 4b | Negative: router fails the gate at S\* ∈ {0.90, 0.95, 0.99} (`docs/phase4b_final_report.md`) | `results/phase4b_final_manifest.json` (router manifest sha256 `2f5f1ab6…`, 1,142 files verified; confirmation-set and GT hashes); training dir; eval dirs `per_query_rows.csv`, `eval_report.json`, `tables.md` | `results/phase4b_train_311634ee3011_20261002T145649Z/`, `results/phase4b_eval_confirmation_20261002T175418Z/`, `results/phase4b_eval_old_test_second_look_20261002T175740Z/`; confirmation vectors `data/confirm/sift_learn_confirm_v2_seed20261007_n2000.fvecs` (sha256 `69070a20…d91d`, recomputed = manifest) | Ignored (results, data); tracked (configs, splits) | archive results + manifest; **must not be modified** |
| C1 / DARTH | **C1 FAIL**: +11.0% dc vs B1 (95% CI 8.9–13.0%) at higher recall (`docs/exp_c_report.md`) | `c1_report.json`, `per_query_darth_vs_b1.csv`; `train_report.json`; model `derived/exp_c/darth_lgbm_s0.txt` (`d291a2f1…`), policy `derived/exp_c/darth_policy_s0.json` (`bcbe2720…`) | `results/exp_c_eval_20261004T211328Z/`, `results/exp_c_train_20261004T202826Z/`; verify runs incl. `results/9cb179548eb1_20261004T202453Z`; trace `data/exp_c/trace_train.f64` | Model/policy tracked; results/trace ignored | archive results; trace hash only |
| Phase 5′ | **H1′ SUPPORTED**: feature→effort Spearman stable in every cell (X = 0.05) (`docs/phase5b_report.md`) | validation `report.json`, `overlap_top10.npz`; analysis `analysis.json` (`0371e266…5850`, recomputed), `rows_query_state_seed.csv.gz` (`631f0c97…c128`, recomputed), `delta_cells_*.csv`, `rho_by_seed_state.csv`; state manifest with per-state index hashes | `results/phase5b_validation_20261004T225337Z/`, `results/phase5b_analysis_20261004T230023Z/`; `data/phase5/sift/states/run_manifest_with_hashes.json` and per-state oracle/features run dirs it lists; synthetic `results/phase5_synthetic_validation_20261002T191139Z/` | Ignored | archive results + state manifest; index hashes only |
| Experiment E | H-E1/H-E2 STABLE everywhere for B1 and DARTH; H-E3 10/16 associated (Holm); H-E4 near-unbiased (`docs/exp_e_report.md`) | see below | see below | see below | see below |

### 4.1 Experiment E, explicitly
| Item | Artifact | Location | Hash (source) | Tracked? |
|---|---|---|---|---|
| Frozen policies | seed 42 `darth_policy_s0.json`; seeds 43/44 `darth_policy_s0_seed{43,44}.json` | `derived/exp_c/`, `derived/exp_e/` | `bcbe2720…`, `6b2275ef…`, `df0bd726…` (`results/exp_e_rows_20261005T110522Z/metadata.json` `policy_sha256`) | yes (derived) |
| Frozen models | `darth_lgbm_s0.txt`, `darth_lgbm_s0_seed{43,44}.txt` | `derived/exp_c/`, `derived/exp_e/` | `d291a2f1…`, `9003ba8b…`, `21a27a77…` (policy JSONs / earlier session records) | yes |
| B1 / REF mixes | computed inside the run; recorded per (seed, state) | `metadata.json` → `mixes` | covered by the metadata file | no (results) |
| State definitions | Phase 5′ states (id/ood 10k–80k, del 20k/80k; seeds 42–44) | `configs/phase5/sift1m.yaml`; `data/phase5/sift/states/run_manifest_with_hashes.json`; `order_{id,ood,del}.csv` | index sha256 per state in that manifest | config yes; manifest/orders **no (ignored)** |
| Configuration | plan + 33 cell configs + 4 DARTH S0 seed configs | `configs/exp_e/` | plan `c602f320…` (`metadata.json`) | yes |
| Design freeze | `docs/exp_e_design_freeze_v2.md` | `docs/` | `43571dfb…` (`metadata.json` `design_doc_sha256`) | yes |
| Analysis code | `exp_e_lib.py`, `exp_e_run.py`, `exp_e_analysis.py`, `router4b_lib.py`, `router_lib.py` | `python/` | `b32b6b23…`, `592b784d…`, `92f45f02…`, `d4a4613d…`, `1af3e385…` (`metadata.json` `scripts_sha256`) | yes |
| DARTH binary | `build/exp_c_darth` | `build/` | `6cea8d12…` (`metadata.json`). Machine-specific (`-march=native`) | no (build) |
| Per-query rows | `rows.csv.gz` | `results/exp_e_rows_20261005T110522Z/` | `bb4c6e25…a22f` (recomputed) | no |
| DARTH per-cell runs | 33 run dirs | listed in `metadata.json` `darth_runs` | — | no |
| Precheck | `precheck.json` | `results/exp_e_precheck_20261005T105734Z/` | — | no |
| Analysis outputs | `analysis.json` (two runs, identical) | `results/exp_e_analysis_20261005T110646Z/`, `…110753Z/` | `442fdc78…6559` (recomputed) | no |
| Stage-1 S0 DARTH seed evals | eval dirs | `results/exp_e_darth_s0_seed{43,44}_eval_*` | — | no |

**Gap:** no single tracked index file maps every reported number to its artifact and
hash. The facts exist, spread across reports, manifests and `metadata.json`.

---

## 5. Reproducibility level per phase

The criteria were:
- data availability;
- seeds;
- configuration;
- dependency versions;
- **commit hash** (none anywhere);
- model and artifact hashes;
- documented commands;
- expected outputs;
- access to large files.

| Phase | Level | Why |
|---|---|---|
| Phase 1 | **PARTIAL** | **Present:** config and seed (level seed 42, single-threaded build); public data via the script; outputs recorded. **Missing:** commit is "none"; no tracked hashes for GT or index; no documented run command (the binary's usage is `ars_static_sweep <config.yaml>`, but the build and invocation sequence is undocumented); results are not public |
| Phase 2 | **PARTIAL** | **Present:** fixed split (seed 20261001) in repo. **Missing:** same gaps as Phase 1; the order of oracle → `analyze_oracle.py` is undocumented |
| Phase 3 | **PARTIAL** | Same as Phase 2. **Additionally missing:** the `analyze_features.py` arguments (`--oracle --phase1 --gt --base`) need the specific run dirs, which are not mapped anywhere tracked |
| Phase 4 | **PARTIAL** | **Present:** training is deterministic with config; models exported to JSON; tiers frozen in `derived/`. **Missing:** a negative result whose evidence is only in ignored `results/`; no commit |
| Phase 4b | **PARTIAL (strongest of the early phases)** | **Present:** a final manifest hashing the router artifacts (1,142 files), the confirmation set, GT and configs; determinism check exists. **Missing:** commit is "none"; seed-43/44 indexes and GT are not hash-listed in a tracked file; the multi-step eval pipeline (`phase4b_evaluate.py <mode>`, `phase4b_report.py`) has no runbook |
| C1 / DARTH | **PARTIAL** | **Present:** frozen model and policy hashed and tracked; the evaluation is deterministic (byte-identical repeat); `derived/exp_c/README.md` gives regeneration commands; LightGBM pinned. **Limits:** model regeneration is only functionally equivalent (≤ 2.2e-16); the training trace (1.8 GB) is ignored; the binary is built with `-march=native`; commit is "none" |
| Phase 5′ | **PARTIAL** | **Present:** a plan-driven orchestrator; a manifest with per-state index hashes; V6 byte-identical state repro. **Missing:** 20 GB of states are not public and must be regenerated (hours of compute); the manifest is in ignored `data/`; commit is "none" |
| Experiment E | **PARTIAL (closest to FULL)** | **Present:** plan, design and script hashes, policy and binary hashes, submodule pins, rows and analysis hashes; deterministic analysis (two identical runs); documented usage `exp_e_run.py {prepare|precheck|run}`. **Missing:** it depends on Phase 5′ states and Phase 2/3/4b S0 artifacts that are not public; the `exp_e_analysis.py <plan> <rows_dir>` invocation has no docstring; commit is "none". **Analysis-only reproduction** from released `rows.csv.gz` could become FULL once the rows are released |

**No phase is FULL today.** None is INSUFFICIENT, because configs, seeds, code and (for later
phases) content hashes do exist. The limiting factors are the same everywhere:
- the lack of commits;
- the lack of public artifacts;
- the lack of a runbook.

---

## 6. External datasets

| Dataset | Used? | Source | Redistribution | In repo? | How to obtain | Expected files | Hashes recorded |
|---|---|---|---|---|---|---|---|
| SIFT1M (TEXMEX, INRIA; Jégou et al.) | **Yes**, the only real dataset | `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` (in `scripts/download_sift1m.sh`) | no explicit licence was found in the repo. **Do not redistribute**; point users to the official source and cite the TEXMEX corpus / Jégou, Douze, Schmid (TPAMI 2011). Licence status should be checked by the user before any redistribution of derived vectors | No (ignored, correct) | `scripts/download_sift1m.sh` | `data/sift/sift_base.fvecs` (516,000,000 B, d = 128), `sift_query.fvecs` (5,160,000 B), `sift_learn.fvecs` (51,600,000 B), `sift_groundtruth.ivecs` (4,040,000 B, d = 100) | `data/sift/SHA256SUMS` (**ignored**): base `21f66e29…1816`, learn `331bc82b…72ea`, query `f7fc9be1…fdc`, groundtruth `2b71de0a…4f65f4f`. These must be published in a tracked file; the script then verifies them instead of recording new ones |
| Confirmation set v2 (2,000 `sift_learn` rows, seed 20261007) | Yes (Phase 4b) | derived from `sift_learn.fvecs` | derived SIFT data; same caveat | ids/tagging in `splits/` (tracked); vectors ignored | regenerated from `sift_learn` + ids. **The command is a documentation gap** (`ars_make_query_subset <config>` with `configs/phase4b/confirmation_set_v2.yaml` is the likely path, but it is not documented as such) | `data/confirm/sift_learn_confirm_v2_seed20261007_n2000.fvecs` | `69070a20…d91d` (`results/phase4b_final_manifest.json`, matches recomputed) |
| Phase 5′ insertion pool | Yes | derived from SIFT data per design doc Addendum A1.4 | derived | no | `phase5b_run.py` rebuilds and checks it | `data/phase5/sift/pool.fvecs`, `pool_rows.csv`, `pool_integrity.json` | in `pool_integrity.json` and the state manifest (both ignored) |
| Synthetic Gaussian | Yes (Phase 5′(a) metric validation only) | generated | n/a | no | `phase5_run.py configs/phase5/synthetic.yaml` | `data/phase5/synthetic/` | in the run outputs |
| **Modern embedding dataset (secondary)** | **NOT USED.** Never named, downloaded or run | — | — | — | — | — | — |

Nothing was downloaded during this audit.

---

## 7. Submodules and dependencies

| Dependency | Pinned version / commit | Recorded correctly? | Notes |
|---|---|---|---|
| hnswlib | `3f3429661187e4c24a490a0f148fc6bc89042b3d` (`git describe` prints `v0.2-374-g3f34296`; `THIRD_PARTY.md` calls it v0.8.0) | Staged gitlink + `.gitmodules`; also in every run's `metadata.json` `build.hnswlib_commit` | the describe string looks like v0.2 because the tag is not on the fetched history. README should state "v0.8.0 = commit 3f342966" to avoid confusion |
| googletest | `52eb8108c5bdec04579160ae17225d66034bd723` | Staged | tests only |
| yaml-cpp | `f7320141120f720aecc4c32be25586e7da9eb978` (0.8.0) | Staged | not listed by name in CLAUDE.md, but in `THIRD_PARTY.md` |
| LightGBM | `d02a01ac6f51d36c9e62388243bcb75c3b1b1774` (v4.6.0) | Staged; Exp E metadata records it | must be built separately as a static library (`THIRD_PARTY.md`, `CMakeLists.txt:116–119`). Its own nested submodules must be initialised (`--recursive`). Not yet stated in a user-facing doc |
| Python | 3.12.3 (`.venv`) | not recorded in a tracked file | |
| Python packages | `python/requirements.txt`: matplotlib 3.11.2, numpy 2.5.3, pandas 3.0.6, PyYAML 6.0.3, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0, pytest 9.1.1, lightgbm 4.6.0, statsmodels 0.15.0, patsy 1.0.3, formulaic 1.2.2 | the installed versions match all 12 pins (verified) | 30 packages installed; **transitive deps are unpinned**. Generate a full lock file before release |
| C++ toolchain | GCC 13.3.0 (Ubuntu 24.04), CMake 3.28.3, C++20 (`cxx_standard 202002`), OpenMP 4.5 (201511), Release, `-march=native -O3` | compiler/flags recorded in each `metadata.json`; CMake version not recorded | `ARS_NATIVE_ARCH=ON` ties results to the host ISA. **Document the host CPU (AMD EPYC-Milan) and the bit-identity scope** |
| clang-tidy | `ARS_ENABLE_CLANG_TIDY` (OFF in `build/`) | — | open item: three OpenMP files can't be checked without `libomp-18-dev` |
| OS | Linux (Ubuntu 24.04 toolchain) | not recorded centrally | |

Additional environment documentation needed:
- an `ENVIRONMENT` / Installation section (OS, compiler, CMake, Python, OpenMP);
- `git clone --recursive` (or `git submodule update --init --recursive`);
- the LightGBM static-build step;
- the venv setup;
- thread counts;
- `-march=native` caveats;
- the disk requirement (about 47 GB of `data/` were produced; a full rerun needs at least the
  20 GB of Phase 5′ states plus caches);
- runtime estimates.

---

## 8. Experiment runbook: what can and cannot be established

"Established" means the repository itself states the invocation (a script docstring,
`Usage:` line or doc). "GAP" means another researcher could not reconstruct it without
help. No command is invented here.

| Step | Established from repo | Documentation gaps |
|---|---|---|
| 1. Environment | LightGBM build command (`THIRD_PARTY.md`); `python/requirements.txt`; CMake options `ARS_BUILD_TESTS`, `ARS_ENABLE_CLANG_TIDY`, `ARS_NATIVE_ARCH` | **GAP:** CMake configure/build command for the project (only `-DARS_ENABLE_CLANG_TIDY=OFF` is mentioned in `docs/notes.md`); venv creation; submodule init; how to run tests (`ctest` and `pytest python/tests` appear in notes only as results) |
| Data | `scripts/download_sift1m.sh [DATA_DIR]` | **GAP:** confirmation-set generation; expected disk layout |
| 2. Phase 1 | `ars_static_sweep <config.yaml>` (apps docstring) with `configs/phase1_sift1m.yaml`; `python/verify_ground_truth.py --ours --official --base --queries [--out]`; `python/plot_recall_ef.py results/<id>` | **GAP:** the binary path (`build/…`); order of steps; expected outputs listed in one place |
| 3. Phase 2 | `ars_oracle <config.yaml>` with `configs/phase2_oracle_sift1m.yaml`; `python/analyze_oracle.py <oracle_run_dir> [--phase1 <run_dir>]` | **GAP:** sequence; which run dir is canonical |
| 4. Phase 3 | `ars_features <config.yaml>` with `configs/phase3_features_sift1m.yaml`; `python/analyze_features.py <run_dir> --oracle --phase1 --gt --base` | **GAP:** sequence; argument values |
| 5. Phase 4 | `python/train_router.py configs/phase4_router_sift1m.yaml`; `python/evaluate_router.py <cfg> <train_run> [--replot]`; `python/define_effort_tiers.py --oracle <run> --out derived`; audits (`audit_phase4_train.py`, `validate_audit_train.py`, `redesign_evidence_train.py`) | **GAP:** sequence; the redesign context |
| 5. Phase 4b | `ars_make_query_subset <config>`; `ars_oracle` / `ars_features` with `configs/phase4b/*`; `phase4b_feasibility.py <runs…>`; `phase4b_train.py configs/phase4b/router_train.yaml`; `phase4b_confirmation_checks.py <stage> <cfg> …`; `phase4b_verify_confirm_gt.py <oracle_run_dir>`; `phase4b_evaluate.py configs/phase4b/evaluate.yaml <mode>`; `phase4b_report.py <eval_dir> <evaluate.yaml> <mode> > tables.md` | **GAP:** valid `<mode>` values in a runbook; the seed-43/44 S0 build sequence; the full stage order |
| 6. C1 | `exp_c_darth {verify|trace|eval|predict_check} <config>` (usage at `apps/exp_c_darth.cpp:350`); `exp_c_train.py <cfg> <trace_run>`; `exp_c_eval.py <cfg> <eval_run> <repeat_eval_run>`; regeneration steps in `derived/exp_c/README.md` | **GAP:** the ordering of verify → trace → train → eval → repeat; an explicit warning that refitting overwrites `derived/` (stated only in the derived README) |
| 7. Phase 5′ | `phase5_run.py configs/phase5/synthetic.yaml`; `phase5_checks.py configs/phase5/synthetic.yaml`; `phase5b_run.py configs/phase5/sift1m.yaml` (resumable); `phase5b_checks.py configs/phase5/sift1m.yaml`; `phase5b_analysis.py configs/phase5/sift1m.yaml <validation_dir>` | **GAP:** runtime/disk needs; that `build/ars_evolve` is invoked by relative path (so it must be run from the repo root) |
| 8. Experiment E | `exp_e_run.py {prepare|precheck|run} configs/exp_e/experiment_e.yaml` (invokes `./build/exp_c_darth eval`); DARTH seed 43/44 S0 configs `configs/exp_e/darth_s0_seed{43,44}{,_repeat}.yaml` | **GAP:** Stage-1 seed-43/44 training sequence; `exp_e_analysis.py` takes `<plan> <rows_dir>` (from code at `python/exp_e_analysis.py:420`, no usage docstring) |
| 9. Report generation | reports are hand-written Markdown from the analysis outputs | **GAP:** no script regenerates report tables or figures. Paper figures must come from a tracked script |

---

## 9. Paper ↔ repository traceability (required structure)

Not created here. The recommended tracked file is `docs/artifact_map.md` or
`artifacts/MANIFEST.csv` (user's choice). One row per paper element:

| Column | Content |
|---|---|
| `paper_ref` | Figure / Table / Claim ID (e.g. `Fig3`, `Tab2`, `Claim-HE1-B1`) |
| `statement` | the exact number or claim as printed |
| `experiment` | Phase 1 / 2 / 3 / 4 / 4b / C1 / 5′ / E |
| `pre-registration` | design doc addendum / freeze section (e.g. `exp_e_design_freeze_v2.md §6.1`) |
| `artifact` | logical name (e.g. "Exp E analysis") |
| `file` | path inside the release archive (e.g. `results/exp_e_analysis_20261005T110646Z/analysis.json`) plus the JSON key or CSV filter |
| `sha256` | content hash of that file |
| `archive` | which release asset / DOI holds it |
| `producer` | script + config (+ script hash where recorded) |
| `reproduce` | the exact command, taken from the runbook (§8) once it is written |
| `level` | analysis-only / full-pipeline reproducibility |

Every figure script should read only released files and print the hashes it consumed.

---

## 10. Recommended GitHub repository structure

This adds files only. The existing tree stays as it is.

```
README.md                    # new (§12)
LICENSE                      # new (licence choice is the user's; check compatibility with Apache-2.0/MIT/BSD deps)
CITATION.cff                 # new
THIRD_PARTY.md               # existing
CMakeLists.txt, .clang-*     # existing
apps/ src/ include/ tests/   # existing
python/ (+ requirements.txt, + a full lock file)
configs/                     # existing
derived/                     # existing (frozen models/policies, in git)
splits/                      # existing
scripts/                     # existing + future figure/verification scripts
docs/                        # existing reports/audits/design
docs/REPRODUCE.md            # new runbook (§8)
docs/artifact_map.md         # new paper↔artifact map (§9)
artifacts/                   # new, tracked, small: SHA-256 manifests only
  datasets.sha256            #   SIFT1M + confirmation vectors
  results_manifest.csv       #   canonical result files, hash, size, role, archive
  states_manifest.json       #   copy of the Phase 5′ index hashes / run mapping
figures/                     # new, generated by scripts/ from released artifacts
paper/                       # new, paper source (PDF via release, or here after acceptance)
results/, data/              # remain ignored; populated from archive or regeneration
```

Not recommended:
- moving existing `results/` dirs or renaming run IDs, since every report cites the
  current paths;
- relocating `derived/`.

---

## 11. Release strategy

The storage choice has not been made yet. This section only recommends.

| Artifact category | Size | Recommended | Why |
|---|---|---|---|
| Code, configs, tests, docs, splits, `derived/`, hash manifests | ~4 MB | **A. normal git** | small text; needed to read and run |
| Paper PDF, figures | small | **A** (figures) + **C** (PDF per tag) | |
| Canonical results bundle (all report-cited run dirs, rows, analysis JSON, manifests, Phase 4/4b models, logs, determinism/regression checks) | ≈ 0.3–1.5 GB (the full `results/` minus `_invalid` and dry-runs is ~1.5 GB; the canonical subset is much smaller) | **D. Zenodo (DOI)**, mirrored as **C. GitHub Release** asset(s) if each file is < 2 GiB | a citable, immutable DOI that outlives the repo; the Release gives convenience |
| Experiment E + Phase 5′ analysis-only bundle (`rows.csv.gz`, `analysis.json`, Phase 5′ analysis dir) | ~30 MB | **C + D** (optionally **A** under an allow-list) | enables FULL analysis-only reproduction of the headline claims |
| HNSW index binaries (S0 per seed + 30 state indexes) | ~20 GB | **E. exclude**, hash manifest only (**D** optional if the user wants instant verification) | deterministic regeneration with V6 byte-identity; size; host-ISA caveat |
| GT caches | ~2 GB | **E**, hash manifest | exact brute force; regenerable |
| DARTH training traces | ~5.3 GB | **E**, hash manifest (**D** optional) | frozen models are authoritative; traces are only for refitting |
| `states_repro/` | 20 GB | **E** | duplicate |
| SIFT1M raw | 551 MB | **E**, download + tracked hashes | third-party data; redistribution not established |
| Confirmation vectors, Phase 5′ pool | 2 MB / 44 MB | **D** (with the results bundle), subject to the SIFT redistribution check, else **E** + regeneration | derived SIFT data |
| `_invalid/`, dry-runs, build trees, venv | — | **E** (document `_invalid` existence) | |
| **Git LFS (B)** | — | **Not recommended** | GitHub LFS quotas and bandwidth costs; no DOI; Zenodo fits immutable research artifacts better |

---

## 12. Required README sections (content specification only)

1. **Project overview.** One paragraph. Hnswlib, SIFT1M, L2, k = 10.
2. **Research question.** As executed, not as originally planned:
   - Phase 5′ RQ5′: feature→effort stability;
   - Experiment E: S0-calibration validity of a fixed-effort and a DARTH-style policy under
     bounded evolution;
   - the original router question and how it was reframed after Phase 4/4b.
3. **Main findings**, including the negative ones:
   - Phase 4/4b router negative;
   - C1 FAIL;
   - H1′ supported;
   - E results;
   - the bounded regime (1–8% insertion, 2–8% lazy deletion);
   - the A1.10 limitation wording.
4. **Repository structure**: the tree from §10 with a one-line role per directory.
5. **Installation**:
   - OS and toolchain;
   - recursive clone;
   - LightGBM build;
   - CMake configure/build (to be established);
   - Python venv + lock;
   - tests;
   - the `-march=native` note.
6. **Datasets**:
   - SIFT1M download script, licence/citation note, expected hashes;
   - derived sets;
   - an explicit statement that **no secondary embedding dataset was used**.
7. **Reproduction**:
   - two tiers: (a) analysis-only from released rows; (b) the full pipeline;
   - disk and time requirements;
   - a link to `docs/REPRODUCE.md`.
8. **Experiment map**: the §4 table in brief, with the pre-registration doc for each.
9. **Artifact verification**:
   - how to download the archive;
   - `sha256sum -c` against `artifacts/`;
   - the bit-identity scope.
10. **Paper**: status, link, `docs/artifact_map.md`.
11. **Limitations**:
    - single real dataset;
    - bounded churn;
    - lazy deletion only;
    - three index seeds;
    - single host;
    - the DARTH re-implementation is not a reproduction;
    - planned experiments not performed;
    - the literature audit was not systematic.
12. **Citation**: CITATION.cff; also cite DARTH, hnswlib and SIFT1M.
13. **License**: the project licence plus `THIRD_PARTY.md`.

Also recommended:
- a "Provenance & integrity" note (pre-registration, `_invalid` handling, no commits during
  the experiments, script hashes);
- a statement that development was AI-assisted, if the user chooses to disclose it.

---

## 13. Publication status (record)

- **The experiments are complete.** Phases 1–4, 4b, C1, 5′(a/b) and Experiment E were
  executed. Phase 4/4b and C1 are frozen negative results.
- **The paper is not yet final.** It has not been drafted, and no paper figures exist.
- **The repository is not public or release-ready.**
  - There are no commits.
  - The artifacts are not packaged.
  - There is no README, LICENSE or CITATION.
- **Planned but not performed. These must not be implied to exist:**
  - the secondary modern-embedding dataset (design doc §235/§247, Phase 9);
  - the recalibrated router and the original static-vs-recalibrated comparison;
  - design-doc Phases 6–10 in their original form (the project was reframed by Addendum A1
    and the later C1 and E1 addenda);
  - graph edge-churn staleness (§8, deferred);
  - update-order sensitivity;
  - non-lazy (compacting) deletion;
  - larger churn levels;
  - other ANN libraries.
- **The literature audit (`docs/literature_verification.md`) was not an exhaustive
  systematic review.** Some sources were checked from abstract or metadata only, ProS was
  not read, and one source was blocked.
- **Novelty claims remain qualified.** Gap verdict B: only "we are not aware of …"
  wording; Ada-ef §7.5 and OMEGA are prior staleness evidence and must be cited.

---

## 14. Final release checklist

### BEFORE PAPER DRAFT
- [ ] User makes the initial commit(s) of the current tree, so code exists at a commit.
- [ ] Verify that the committed `python/exp_e_*.py`, `router*_lib.py`, the Exp E plan and the
      design freeze match the hashes in `results/exp_e_rows_20261005T110522Z/metadata.json`.
      Record the commit ↔ hash correspondence in `docs/notes.md` or the artifact map. Run
      metadata itself stays untouched.
- [ ] Do the same check for the Phase 5′ and Phase 4b manifests where scripts or configs are
      hashed.
- [ ] Draft `docs/artifact_map.md` skeleton (§9) listing every number the paper will cite.
- [ ] Write figure scripts (tracked) that read only canonical artifacts and print the
      hashes they consume.
- [ ] Decide the treatment of non-canonical run dirs (archive with labels vs. exclude).

### BEFORE PUBLIC RELEASE
- [ ] Choose a LICENSE and check compatibility with Apache-2.0 (hnswlib), MIT (yaml-cpp,
      LightGBM) and BSD-3 (googletest).
- [ ] Write README.md (§12), `docs/REPRODUCE.md` (close every §8 GAP with commands actually
      tested on a clean clone), and CITATION.cff.
- [ ] Publish the SIFT1M SHA-256 sums in a tracked file. Decide whether
      `download_sift1m.sh` should verify against them instead of recording them (a code
      change, so it needs approval).
- [ ] Generate a full Python lock (`pip freeze`) and record the Python, CMake, GCC, OS and
      CPU versions.
- [ ] Document the `-march=native` / bit-identity scope. Optionally test a portable
      (`ARS_NATIVE_ARCH=OFF`) build. That would be a new verification run, so it needs
      approval.
- [ ] Create `artifacts/` hash manifests (datasets, canonical results, state/index hashes,
      traces).
- [ ] Package the canonical results bundle and the analysis-only bundle. Verify the hashes
      after packaging.
- [ ] Do a clean-clone dry-run (another directory or machine): build, tests, analysis-only
      reproduction of Exp E and Phase 5′ numbers from the bundle.
- [ ] Check the SIFT1M terms before releasing any derived vectors (confirmation set, pool).
- [ ] Review docs for local absolute paths, personal data and the user's email. Decide
      whether `CLAUDE.md` ships.
- [ ] Clarify the hnswlib version string (v0.8.0 = 3f342966) everywhere it is presented.
- [ ] Ensure README and paper limitations list the non-performed experiments (§13).
- [ ] Optional: resolve clang-tidy coverage of the OpenMP files.

### BEFORE FINAL GITHUB TAG
- [ ] The paper is final, and every `artifact_map` row has a file, a hash and a command.
- [ ] Upload the archive to Zenodo (DOI reserved) and mirror the GitHub Release assets.
- [ ] Put the DOI and paper citation into README and CITATION.cff.
- [ ] Re-run tests (gtest + pytest) at the tag commit and record the counts.
- [ ] Re-verify all artifact hashes against the released assets.
- [ ] User creates the annotated tag (e.g. `v1.0-paper`) and pushes it. The user performs all
      pushes.
- [ ] Optional: enable Zenodo–GitHub integration so the tag gets a DOI snapshot of the code.

---

## 15. Final verdict

**A. Current repository readiness: NOT READY** for public release.
- The research content is in good shape:
  - pre-registration docs;
  - frozen hashed models and policies;
  - a hashed Exp E provenance chain;
  - deterministic analyses.
- The packaging is absent: no commits, no README, LICENSE, CITATION or runbook, and every
  result artifact is git-ignored.

**B. Biggest reproducibility gap.** No result is linked to a code version:
- all 272 run metadata files record `commit: none, dirty: true`;
- the full pipeline depends on about 20 GB of regenerated Phase 5′ states built with a
  `-march=native` binary on one host.

Script-level SHA-256 hashes (Exp E, Phase 5′, 4b) are the partial bridge. They must be
reconciled with the first commit.

**C. Biggest GitHub packaging gap.** `.gitignore` excludes all of `results/` and `data/`,
with no tracked manifest. Together with the missing README and runbook, a cloned
repository could not locate, verify or regenerate a single reported number.

**D. Artifacts that must be released:**
- all code, configs, tests, docs, `splits/` and `derived/` (frozen models and policies);
- hash manifests for SIFT1M, the confirmation set, state indexes, GT caches and traces;
- the canonical results bundle:
  - Phase 1–4 canonical runs;
  - Phase 4b train/eval dirs + `phase4b_final_manifest.json`;
  - C1 train/eval/verify runs;
  - Phase 5′ validation + analysis + the state manifest;
  - Exp E rows, metadata, precheck, analysis and per-cell DARTH runs;
  - the determinism and regression checks.

**E. Artifacts that should NOT be released:**
- SIFT1M raw files (third-party);
- `results/_invalid/` and `data/confirm/_invalid/` contents (document their existence
  only);
- dry-runs;
- `states_repro/`;
- build trees, `.venv` and caches;
- by default, the ~20 GB of index binaries, the GT caches and the DARTH traces (hash
  manifests instead).

**F. Recommended strategy.** Three layers:
- **GitHub (plain git):** code + docs + configs + `derived/` + `artifacts/` hash manifests +
  figures + paper source.
- **Zenodo DOI archive (mirrored as a GitHub Release):** the canonical results bundle, plus
  a small analysis-only bundle that reproduces the Exp E and Phase 5′ tables from rows.
- **Regenerate, verified by hash:** everything else (indexes, GT, traces).

No Git LFS.

**G. Exact next task after this audit:**
1. The user makes the initial commit of the current tree. It must be made by the user;
   I will not run it.
2. Then, with approval, a read-only **provenance reconciliation** step: confirm that the
   committed scripts, plan and design freeze match the SHA-256 hashes recorded in the
   Exp E, Phase 5′ and Phase 4b metadata, and draft the `docs/artifact_map.md` skeleton.

Figures and paper drafting can follow in parallel, but should read only from artifacts
listed in that map.

STOP.
