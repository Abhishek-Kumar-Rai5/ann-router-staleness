# Phase 5′ report — feature→oracle-effort signal stability under index evolution

**Verdict: H1′ SUPPORTED.**
- All three core features (knn_dist, centroid_dist, score_concentration), and LID as the
  secondary feature, are **STABLE** at every one of the 10 evolved states. This uses the
  pre-declared rule (design_doc.md A1.6/A1.7; docs/notes.md pre-declaration): pooled Δ, X =
  0.05.
- Per seed, all 120 (feature × state × seed) cells are also STABLE; no seed-specific DEGRADED
  result exists.
- The relationship is **stable within 1–8% insertion churn (and 2–8% lazy deletion)**. This is
  not a general claim about index evolution (A1.10).

Run: results/phase5b_validation_20261004T225337Z (V1–V11),
results/phase5b_analysis_20261004T230023Z (analysis). Characterisation/precondition
experiment only: no router or adaptive policy is evaluated.

## 1. Validation V1–V11: ALL PASS
- **V1:** provenance (data, pool, orders, plan hashes; seeds 20261008/20261009/20261010).
- **V2:** S0 reproduction, under the user-approved reading of pre-existing schema evolution
  (2026-10-04; docs/notes.md).
  - Oracle curves and labels are byte-identical to the canonical runs.
  - The 9 shared Phase-3 feature columns are identical.
  - The full file equals the Phase-4b-era S0 re-run.
  - The literal criterion fails only through the two LID-probe columns added in Phase 4b,
    and is still reported in the validation report.
- **V3:** the rebuilt S0 is byte-identical in all 6 insertion runs.
- **V4:** sizes and counts are exact.
- **V5:** nestedness holds; 0 overlap-monotonicity violations.
- **V6:** all 30 state files reproduce byte-for-byte.
- **V7:** NumPy float64 brute-force ground truth on all 10,000 queries per state; exact up to
  float ties, 0 deleted labels. V7b: ground truth is identical across seeds.
- **V8:** 0 deleted or out-of-range labels returned.
- **V9:** features reproduce on all 30 states.
- **V10:** the oracle reproduces on id_80000 (seed 42), ood_80000 (seed 43) and del_80000
  (seed 44), including freshly recomputed ground truth.
- **V11:** query, seed and config identity hold, with complete, finite outputs.

## 2. Primary result: pooled Δ = |ρ_s| − |ρ_0|
Mean over seeds 42/43/44; 95% query-clustered bootstrap CI (2,000 resamples, seed 20261002);
censored efforts ranked highest. S0 baseline |ρ_0|: knn 0.464, centroid 0.449, concentration
0.376, LID 0.568.

| State | knn_dist | centroid_dist | score_concentration | lid (secondary) |
|---|---|---|---|---|
| id 1% | +0.002 [−0.002, +0.005] | −0.002 [−0.006, +0.001] | −0.000 [−0.005, +0.005] | −0.001 [−0.005, +0.003] |
| id 2% | +0.002 [−0.003, +0.008] | −0.006 [−0.012, −0.001] | +0.000 [−0.007, +0.008] | −0.001 [−0.006, +0.005] |
| id 4% | +0.005 [−0.002, +0.012] | −0.006 [−0.013, +0.001] | +0.003 [−0.007, +0.012] | +0.000 [−0.008, +0.008] |
| id 8% | +0.005 [−0.004, +0.014] | −0.009 [−0.018, +0.001] | +0.011 [−0.001, +0.024] | +0.005 [−0.005, +0.015] |
| ood 1% | −0.001 [−0.005, +0.002] | −0.011 [−0.015, −0.006] | −0.001 [−0.006, +0.004] | −0.003 [−0.007, +0.001] |
| ood 2% | −0.002 [−0.007, +0.003] | −0.017 [−0.023, −0.011] | +0.003 [−0.005, +0.010] | +0.000 [−0.006, +0.006] |
| ood 4% | +0.002 [−0.004, +0.008] | **−0.037 [−0.045, −0.029]** | +0.006 [−0.003, +0.016] | +0.002 [−0.006, +0.010] |
| ood 8%¹ | +0.003 [−0.006, +0.012] | −0.013 [−0.022, −0.004] | +0.005 [−0.008, +0.018] | +0.001 [−0.010, +0.011] |
| del 2% | +0.000 [−0.003, +0.003] | −0.000 [−0.003, +0.003] | −0.001 [−0.005, +0.003] | +0.001 [−0.003, +0.004] |
| del 8% | −0.003 [−0.010, +0.003] | −0.004 [−0.010, +0.002] | −0.001 [−0.009, +0.008] | +0.003 [−0.004, +0.010] |

¹ At 8% insertion magnitude, the available pool forces the regional-OOD construction to
overlap approximately 91% with the matched ID set, so this level does not provide a clean
ID-vs-OOD comparison.

**A1.7 classification:** STABLE in all 40 pooled cells (CI lower bound ≥ −0.05). The tightest
margin is centroid_dist at ood 4%: CI lower bound −0.045, i.e. 0.005 inside the margin.

## 3. Per-seed results
All 120 per-seed cells are STABLE (delta_cells_per_seed.csv). There are no seed-specific
DEGRADED or INCONCLUSIVE cells. Seed-to-seed |ρ| differences at S0 are ≤ 0.011, as in A1.6.

## 4. Hypotheses
- **H1′ (primary):** supported for knn_dist, centroid_dist and score_concentration, so H1′
  overall is supported. LID (secondary) is also supported.
- **H2′ (ID vs regional OOD at 1/2/4%; secondary, no threshold):**
  - centroid_dist degrades more under regional OOD than under ID at every inferential level:
    Δ_OOD − Δ_ID = −0.008 [−0.014, −0.002] (1%), −0.011 [−0.018, −0.004] (2%) and
    −0.031 [−0.040, −0.023] (4%);
  - knn_dist, score_concentration and LID show no ID/OOD difference (all CIs include 0);
  - **8% (descriptive only, ≈ ID by construction):** all four CIs include 0. Per the
    mandatory wording above, this is not evidence about a genuine OOD condition.
- **H3′ (pairwise feature differences at the largest magnitude; secondary):**
  - at id 8% and ood 8%, centroid_dist changes more negatively than knn_dist (−0.014 and
    −0.016, CIs exclude 0) and than score_concentration (−0.020 and −0.018, CIs exclude 0);
  - at del 8%, there are no differences;
  - so the A1.6 *expectation* (global centroid_dist more stable than the local features)
    is **not borne out**: the global feature is the least stable, although within the X
    margin.
- **Trend (descriptive, 4 points):** centroid_dist's Δ becomes more negative with magnitude
  under ID (Spearman −0.8). knn_dist and concentration drift slightly upward.

## 5. Effect sizes beyond the correlation (descriptive; pooled over seeds)
- **Per-query effort is reshuffled even though the feature→effort correlation is stable.**
  The Spearman correlation of each query's required effort with its own S0 effort is:
  - in-distribution: 0.97 / 0.95 / 0.91 / **0.84** at 1/2/4/8%;
  - regional OOD: 0.97 / 0.95 / 0.92 / 0.84;
  - deletion: 0.98 / 0.93 at 2/8%.
- At id 8%, 52% of queries need more effort and 23% less, but the mean log₂ effort ratio
  is only +0.005. The distribution is nearly unchanged while individual queries move.
- Deletion lowers effort: at 8%, 66% of queries need less and 8% more; mean log₂ ratio −0.06.
- **Neighbourhood overlap with the S0 top-10** (mean / share of queries with any change):
  - id 8%: 9.06 / 61%;
  - ood 8%: 9.10 / 57%;
  - del 8%: 9.20 / 57%.
- **Features themselves barely move:** knn_dist rank-correlation with S0 ≥ 0.996, and the
  median relative change is 0. score_concentration and LID move more (rank-correlation 0.78
  and 0.87 at id 8%).
- Distributional drift D: ID ≤ 0.011; OOD 0.66 / 0.63 / 0.56 / 0.07 (1/2/4/8%); deletion
  ≈ 0.
- **Difficulty strata** (S0 effort, frozen 18/48 thresholds; descriptive): within-stratum Δ
  is mostly ≥ 0. Restricting each stratum's range at S0, followed by later effort
  reshuffling, plausibly inflates within-stratum ρ (regression toward the mean). Not
  interpreted further.

## 6. What this establishes
At 1–8% insertion churn (in-distribution or regionally concentrated) and 2–8% lazy deletion
on SIFT1M/hnswlib, the *population-level* rank association between the three cheap
features (and LID) and the oracle-required effort stays within 0.05 of its S0 value.

## 7. What it does NOT establish
- **That a router or adaptive policy built on these features stays accurate or saves
  cost.** Signal stability is necessary, not sufficient. Phase 4b and C1 found no S0 cost
  advantage over B1 to preserve in the first place.
- **That per-query calibration is stable.** The opposite is visible descriptively: up to
  about half the queries change required effort, and per-query effort rank-correlation
  falls to 0.84 at 8%.
- **That a genuine OOD distribution behaves similarly.** "OOD" is a narrow construction:
  regionally concentrated real SIFT vectors (a nearest-neighbour ball around one seeded
  anchor in the same sift_learn pool that supplies the ID inserts), not an independently
  sourced or semantically different distribution. Its concentration weakens with
  magnitude: distance ratio 0.67 at 1%, 0.97 at 8%, where OOD ≈ ID.
- **Anything beyond 8% churn,** physical deletion or graph repair (deletion is lazy only),
  or another dataset (SIFT1M only).
- **Anything about H4′** (structural measures vs update count), which is reserved for
  Phase 7′.

## 8. Implications for Experiment E (recommendation only; nothing started)
- **The question has moved.** Combined with Phase 4b (static router: no saving vs B1) and C1
  (DARTH re-implementation: 11% more cost than B1 at S0), Phase 5′ shows the *population*
  signal is stable, while *per-query* required effort reshuffles substantially. So "does the
  feature→effort signal decay?" is answered: not within this churn. The open, evidence-backed
  question is **calibration staleness**:
  - does a policy calibrated at S0 still meet its quality contract, and at what cost, when
    queries' required efforts reshuffle?
  - and does structural change (overlap decay, density drift) predict which queries and
    states go wrong?
- **Recommended E framing (needs a new addendum and your approval):** calibration staleness
  of S0-calibrated policies measured against their own contracts. Neither requires an S0
  advantage over B1, which neither has.
  - **Primary:** the frozen B1 fixed-effort mixes (the static calibration everything has
    been compared with).
  - **Secondary, labelled as a C1-failed actuator:** the frozen C1 DARTH policy, as a
    runtime-calibrated contrast. It holds its contract (0.963 ≥ 0.95) at S0.
  - **Outcomes, per state:** realised mean recall vs target, per-query success, cost, and
    whether overlap decay / density drift predict per-query recall loss (this links to H4′).
  - All 30 states, features and oracle curves already exist. B1's per-state behaviour can be
    read from the existing oracle curves without new searches.
- **Not recommended:** searching for a third actuator before E (actuator-shopping), or
  presenting E as testing whether adaptive policies "lose their advantage" (there is none
  to lose here).

## 9. Limitations (A1.10, mandatory wording)
1. "OOD" is a narrow construction: regionally concentrated real SIFT vectors from the same
   sift_learn pool as the ID inserts, not an independently sourced or semantically
   different distribution. Its concentration weakens with magnitude: distance ratio 0.67 at
   1%, 0.97 at 8%, where OOD ≈ ID.
2. The finding is "stable within 1–8% insertion churn (and 2–8% lazy deletion)", not a
   general claim about index evolution.
3. Deletion is lazy only (markDelete); there is no physical removal or graph repair.
4. One dataset (SIFT1M) at this stage.
5. Signal stability is necessary, not sufficient; this says nothing directly about router
   cost savings.
6. At 8% insertion magnitude, the available pool forces the regional-OOD construction to
   overlap approximately 91% with the matched ID set, so this level does not provide a clean
   ID-vs-OOD comparison.

Further limits:
- Three index seeds and one fixed query set (10,000 SIFT queries).
- The OOD anchor is a single seeded choice.
- H2′/H3′ are secondary and have no thresholds.
- The strata analysis is descriptive and affected by range restriction.

## Artifacts
| Artifact | Location |
|---|---|
| Validation | results/phase5b_validation_20261004T225337Z/report.json (+ overlap_top10.npz) |
| Analysis | results/phase5b_analysis_20261004T230023Z/: analysis.json, delta_cells_pooled.csv, delta_cells_per_seed.csv, rho_by_seed_state.csv, per_state_descriptive.csv, strata_*.csv, rows_query_state_seed.csv.gz (row-level: one row per seed × state × query), delta_vs_magnitude.png |
| State manifest | data/phase5/sift/states/run_manifest.json (+ run_manifest_with_hashes.json) |
