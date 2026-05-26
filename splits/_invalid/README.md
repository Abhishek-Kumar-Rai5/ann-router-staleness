# Rejected confirmation set v1 (historical record — never evaluated)
Files: confirmation_sift_learn_seed20261005_n2000_{ids,tagging}.csv (vectors in
data/confirm/_invalid/). Config: configs/phase4b/confirmation_set_v1_REJECTED.yaml.
Drawn 2026-10-02 from all 100,000 sift_learn rows (seed 20261005). REJECTED by the
frozen integrity check: 190 exact duplicates of original SIFT queries (150 train,
40 test) and 1 internal duplicate pair; root cause: sift_query is contained in
sift_learn. No ground truth, oracle curves, features or evaluation were ever computed
on it. See results/phase4b_confirmation_STOP.md and docs/phase4b_evaluation_state.md.
