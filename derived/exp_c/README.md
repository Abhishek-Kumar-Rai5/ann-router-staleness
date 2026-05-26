# Experiment C (C1) frozen artifacts — negative result, do not modify

| File | sha256 | Role |
|---|---|---|
| darth_lgbm_s0.txt | d291a2f1f8cbd64f9fa2e744dd0cfc8bcc1b4261aba13915c88d1b94430eb041 | LightGBM 4.6.0 recall predictor (text model) |
| darth_policy_s0.json | bcbe2720044b51edff1b493b956e413958f8503a6a76bfe86f52317cec8ac75d | complete frozen DARTH policy (model hash, feature order, ef cap 142, Rt 0.95, ipi 367, mpi 73, data/index hashes) |

Design: docs/design_doc.md Addendum C1 (C1.9). Result (C1 FAIL): docs/exp_c_report.md.

## Reproducibility
- **The artifact is authoritative.** The C1 result was produced with exactly
  `darth_lgbm_s0.txt`. Re-running the evaluation with it is deterministic: a repeat run is
  byte-identical apart from timing columns.
- **The model file is not bit-reproducible by refitting.** It was trained with the frozen
  settings (LGBMRegressor(objective="regression", n_estimators=100, random_state=42),
  otherwise default, i.e. multi-threaded).
  - An identical refit on the identical trace (sha256 in the policy file) gives a
    byte-different file: last-digit differences in leaf values from multi-threaded
    floating-point summation.
  - Predictions differ by ≤ 2.2e-16 (354,540 trace rows), and no 0.95 stop decision
    changes.
  - So a regenerated model is functionally, but not bitwise, equivalent.
- **Deterministic settings are recorded for future models only.** For new models,
  LightGBM's `deterministic=True, force_row_wise=True, num_threads=1` give bit-reproducible
  refits. These are **not** the C1 settings and must not be used to regenerate or replace
  the C1 model: that would change the frozen configuration.
- **Regenerating (functionally equivalent):**
  1. build LightGBM v4.6.0 (see CMakeLists.txt);
  2. `./build/exp_c_darth trace configs/exp_c/darth_s0.yaml`;
  3. `python python/exp_c_train.py configs/exp_c/darth_s0.yaml <trace_run>`. This
     overwrites these files, so do it in a copy of the repository, never here.
