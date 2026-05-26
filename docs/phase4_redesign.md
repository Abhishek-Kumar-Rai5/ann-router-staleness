# Phase 4 Redesign — Locked Experimental Contract (proposal for approval)

Date: 2026-10-02. Status: **PROPOSAL. Nothing here has been implemented.**

This document builds on `docs/design_doc.md`, `docs/notes.md`,
`docs/methodology_audit.md`, `docs/phase4_methodology_validation.md` and
`CLAUDE.md`.

Ground rules for this step:
- No retraining, no new HNSW index, no test-split access, no change to
  canonical Phase 1–4 results.
- The primary choices were written into `docs/notes.md` ("Phase 4 redesign —
  PRE-DECLARATION") **before** the feasibility numbers below were computed.
- Training-split evidence comes from:
  - `python/redesign_evidence_train.py` →
    `results/redesign_evidence_20261002T125113Z/redesign_train.json`
    (seed 42 index, 8,000 training queries, no model fitted);
  - the earlier audit and validation outputs.
- Test-split results were used for nothing in this document.

**Disclosure.** The decision to redesign was triggered by the Phase 4 test
outcome. No *parameter* below was chosen using test data. However, the
*direction* of the feasibility evidence was predictable from the training-only
audit: per-query contracts and stricter targets give routing more headroom.
That is why every choice is justified from the research question and the
design text, and why the gate (§16) is deliberately strict.

---

## 1. Original research question

Design §1/§4: "Do per-query ANN routing decisions, calibrated on one index
state, remain accurate as the index evolves — and does a structural measure of
index change predict routing failure better than a simple update count does?"

Phase 4 is the prerequisite (§21: "a router with no S0 advantage cannot
meaningfully be said to 'degrade'"). The question Phase 4 must answer:

> **Can query-level information be used to allocate ANN search effort per query
> so that a system achieves a specified retrieval-quality target with less
> computation than a fixed search effort?**

## 2. What Phase 4 actually tested

Whether a 3-class tier classifier, trained with symmetric 0-1 loss on
oracle-tier labels (tiers 18/48/327 from training quantiles of oracle ef), and
charged a full extra probe search, beats the fixed-ef curve at **the mean recall
the classifier happened to reach** (≈ 0.95). Result: no, at 2.3× the cost
(test) / 2.4× (train OOF).

## 3. Problems with the original formulation

From the audit (C1, C2, M1–M5) and the validation (V-1 to V-7):

1. **The gate was unpassable by construction.** At the router's operating point,
   even perfect information restricted to 18/48/327 plus the probe loses to
   fixed ef: −203 dc, CI [−212, −192] under the mean contract; 1,248 vs 1,130
   under the per-query contract.
2. **The training objective didn't match the evaluation objective.** Labels
   meant "cost of 10/10 per query"; the evaluation used mean recall. The 0-1
   loss ignored that "high" costs about 5× "med".
3. **No declared operating point.** The quality level was a side effect of the
   classifier's errors.
4. **Tiers placed for class balance, not cost.** 48 → 327 is about 4.5× in cost.
5. **Ambiguous baseline.** "Matched recall" against a fixed ef selected on the
   evaluation set.
6. **Regret not at matched quality.** Failures produced negative regret.
7. **One index seed**, while §13 requires 2–3 seeds for core conditions.
8. **The test split has been seen once.**

The implementation itself is correct (independently verified, 20/20
distance counts).

## 4. Corrected research objective

For a declared quality contract Q, compare **the cheapest computation that
achieves Q** under:
- (a) one fixed efSearch for all queries;
- (b) a per-query choice among a fixed candidate set, made from query-time
  features.

Both are pre-committed policies, calibrated on training data only and evaluated
on the same held-out queries under the same cost accounting.

Phase 4's job: show (or fail to show) that (b) needs **materially less
computation** than (a) **without worse quality**.

## 5. Recall contract

| Contract | Meaning | Alignment with the question |
|---|---|---|
| Mean recall@10 ≥ R* | Average quality over queries | Hides the tail: a fixed ef meets mean 0.95 while ~29% of queries miss 10/10. Lets any policy trade hard-query failures for easy-query gains. Not what §2.3 motivates ("hard queries need more effort *to reach the same accuracy*"). |
| **Per-query success rate ≥ S\*** | Fraction of queries whose own recall meets the per-query target | Matches the design's motivation (§2.3) and the oracle definition (§5.3, §7.7: *per query* minimum effort for a target). Gives one well-defined binary outcome per query (paired tests, regret split). Is the usual service-level form (p95-style). |

**Decision (proposed): primary contract = per-query success rate.**
- A query "succeeds" iff tie-aware recall@10 ≥ 0.95, which is 10/10 (the
  user-locked definition).
- The system meets the contract iff the success rate ≥ S*.
- Mean recall is a pre-declared **secondary** contract, reported, never gating.

**Why not chosen for a positive result.** The design text fixes the oracle per
query and motivates routing by per-query difficulty, both written before any
experiment. A known weakness: the per-query contract ignores how badly a
failure fails (9/10 and 1/10 count the same). Mitigations:
- mean recall is reported for every policy;
- a mean-recall non-inferiority guard is reported (secondary).

## 6. Operating-point selection rule

**Rule (pre-declared):** primary **S\* = 0.95**, the design's 0.95 convention
applied at system level ("95% of queries meet their per-query target").
Secondary operating points: S* ∈ {0.90, 0.99}, reported, never gating. S* is not
tuned on any data.

**Feasibility pre-flight (pre-declared).** On each index seed's training split,
compute the perfect-information ("contract oracle", §8) cost at S* with the
primary candidates and accounting. If its saving versus the fixed ef meeting S*
is **< 20%**, the gate is declared **NOT TESTABLE** at S* and reported as such.
S* is never moved to rescue the gate.

Training evidence (seed 42; primary candidates; per-query contract):

| S* | Fixed ef meeting S* (train) | Its cost | Perfect info, additive probe | Saving | Descent-shared | Search-only |
|---|---|---|---|---|---|---|
| 0.80 | 71 | 1,353 | 1,010 | +25.4% | +30.8% | +54.3% |
| 0.90 | 117 | 2,006 | 1,254 | +37.5% | +41.7% | +56.9% |
| **0.95** | **172** | **2,734** | **1,419** | **+48.1%** | +51.4% | +62.4% |
| 0.98 | 256 | 3,770 | 1,583 | +58.0% | +60.5% | +68.4% |
| 0.99 | 327 | 4,593 | 1,644 | +64.2% | +66.3% | +72.7% |

At S* = 0.95 the pre-flight passes (48.1% ≥ 20%) on seed 42, so the gate is
**passable in principle**. The bound assumes perfect information. The previous
router's ~50% tier accuracy suggests a real router realises only a fraction of
it, so passing is not assured.

For comparison:
- Under the **mean** contract, the same candidates give only +1.5% at R = 0.95,
  +11.7% at 0.97 and +34.4% at 0.99.
- The **frozen tiers** under the primary contract give +15.1% at S* = 0.95,
  which would be NOT TESTABLE.

## 7. Probe-cost model

What black-box hnswlib actually executes: one `searchKnn(k=10, ef=10)` probe,
then, unless the probe's own result is returned, one routed
`searchKnn(k=10, ef=c)`.

- **Primary (physically exact): fully additive.**
  - Cost = probe dc + routed-search dc, both counted by `CountingL2Space`.
  - Returning the probe's own ef=10 result costs **only the probe**. That is a
    real, deployable option with no extra work (§10).
- **Sensitivity 1: shared descent.** Subtract this query's upper-layer descent
  (measured: mean 118 dc, p10–p90 89–149, 31% of the probe). The descent is
  ef-independent and repeated identically. Realising it requires a public-API
  path (`searchBaseLayerST` with the descent endpoint) that re-implements the
  descent outside hnswlib. **Not assumed in the primary result.** Using it in a
  primary result needs a separate scope ruling.
- **Sensitivity 2: search-only bound.** The probe is free; this is an
  optimistic upper bound on routing benefit only.
- **Not allowed:** resuming the probe's base-layer search state. That needs a
  custom search loop, outside the black-box boundary.
- Fixed-ef policies compute no features and pay no probe.
- The LID ablation is charged its extra (k=20, ef=20) probe *instead of* the
  core probe: its top-10 provides the core features, which fixes audit m2. The
  option of returning a probe result then uses the ef=20 probe result.

## 8. Oracle definition

Three distinct references, never conflated:

1. **Grid oracle** (unchanged Phase 2 artefact): per query, the smallest ef on
   the 121-value grid from which the per-query target holds at every larger
   grid ef. Used for difficulty groups (frozen thresholds 18/48) and as a
   descriptive reference.
2. **Candidate oracle** (training label and per-query regret reference): per
   query, the cheapest candidate c in the candidate set (§10, with "probe
   result" as the cheapest) from which the per-query target holds at every
   larger candidate (stable reach within the candidate set). If no candidate
   satisfies it, the query is *unachievable*: it counts as a failure for every
   policy and is reported separately. The 2 known censored test queries are
   examples.
3. **Contract oracle at S\*** (system-level lower bound): the perfect-
   information policy over the same candidates and the same accounting that
   reaches success rate ≥ S* at minimum mean cost. Each achievable query either
   succeeds at its candidate-oracle cost or is served at the cheapest
   candidate; the cheapest-to-satisfy ⌈S*·n⌉ queries succeed. This is the bound
   used in §6.

## 9. Router target and output

| Option | What is learned | Alignment |
|---|---|---|
| A. Tier classification | P(tier = t \| x) with 0-1 loss | Misaligned: symmetric loss, no quality control (audit C2). |
| B. Regress required ef | E[log oracle ef \| x] | Ordinal information kept, but needs an ad-hoc margin; cost asymmetry not modelled. |
| C. Success probability per candidate | P(success at c \| x) for every candidate c | Directly models the contract outcome. |
| **D. Cheapest candidate predicted to satisfy the target** | The decision rule on top of C | Directly implements the research question's allocation. |

**Decision (proposed): C + D.**
- Label: the candidate-oracle index y_i (§8). Success at candidate c ⇔ y_i ≤ c.
- **Model:** for each candidate threshold c, a binary model of P(y ≤ c | x).
  Family as in design §9: logistic regression or a small decision tree
  (depth ≤ 4); no neural models. Monotonicity in c is enforced by a cumulative
  maximum over c.
- **Decision:** choose the cheapest candidate c with P(y ≤ c | x) ≥ τ.
- **Calibration of τ:** on training out-of-fold (OOF) predictions only, τ = the
  smallest value whose OOF success rate ≥ S*. The result: the router targets
  the contract, and its operating point is declared, not emergent.
- **Model selection (training only):** candidates {LR, DT2, DT3, DT4}, 5-fold
  stratified CV (seed 20261002). Each candidate gets its own τ. Pick the lowest
  OOF mean total cost among those whose OOF success ≥ S*; ties within 1% go to
  the simplest (DT2 < DT3 < LR < DT4).
- **Output** remains a discrete effort from a small candidate set, consistent
  with design §9. Features stay exactly the three Phase 3 features; LID is
  only the ablation.

## 10. Candidate effort construction

**Should 18/48/327 stay frozen as router outputs? No.** They were built as
balanced classification classes (ef quantiles), not as effort options. Under
the corrected contract they make the gate untestable (+15.1% < 20%). The
48 → 327 jump makes every conservative error cost about 4,300 dc.

**Rule (pre-declared, data-independent; no training or test data used to place
values):**
- Candidates = {**probe result** (ef 10, zero extra search)} ∪ {smallest grid ef
  ≥ 10·2^i, for i = 1, 2, …, as long as such a grid value exists}.
- This gives **{probe, 20, 40, 83, 164, 327, 647, 1343, 2661}**.
- Reasons for a doubling ladder:
  - uniform relative resolution: adjacent candidates differ by about 2× in ef
    and roughly 1.7–2× in cost, against 4.5× before;
  - no class-balance or data-fitting step that could be tuned;
  - it covers the full range needed for S* up to 0.99 (training: the
    contract-oracle success target is reachable at every S* tested).
- Training-quantile placement was rejected: it re-introduces a data-dependent
  tuning step with no cost justification.

**Separately, the three fixed-ef baselines of design §7.6** (used across all
states in later phases) should be the contract-aligned fixed efforts
**{ef_fix(0.90), ef_fix(0.95), ef_fix(0.99)}** from each seed's training split
(seed 42: 117 / 172 / 327). This replaces the descriptive 18/48/327, which no
longer correspond to any declared quality level. **Needs your approval:** it
reverses the Phase 3 freeze. The tertile thresholds 18/48 stay frozen for
subgroup analysis.

## 11. Regret definition

Per evaluation query i, with the candidate oracle (§8) as reference. Probe and
search costs are kept separate:

| Quantity | Definition | Reported as |
|---|---|---|
| Success s_i | Router's tie-aware recall@10 ≥ 0.95 | Success rate |
| **Avoidable failure** F_i | s_i = 0 and the candidate oracle is achievable | Rate (quality regret) |
| Unavoidable failure U_i | No candidate achieves the target (incl. censored) | Count, reported separately |
| **Overspend** O_i | For s_i = 1: router search dc − search dc at the candidate oracle (≥ 0 except for non-monotone curves, which are reported) | Mean over successes (efficiency regret) |
| Underspend on failures | Candidate-oracle search dc − router search dc for F_i = 1 | Reported, **never netted** against overspend |
| Probe overhead P_i | Probe dc (0 for fixed policies) | Mean |
| **Total cost** C_i | Probe + routed search dc | Mean, median, distribution |
| **Contract regret** | Mean C(router) − mean C(contract oracle at the router's realised success rate) | System-level cost gap to perfect information |

The old regret (total − grid-oracle dc, failures included) is retained only as
a descriptive, clearly labelled legacy metric. For Phases 6–8 (H1), robustness
is measured as the change across states in **avoidable failure rate** and
**overspend**, separately.

## 12. Fixed-effort baseline definition

All comparisons use the same evaluation queries, the same per-query contract,
the same `CountingL2Space` accounting (fixed policies pay no probe) and the
same metrics.

- **B1 (primary): pre-committed fixed effort.**
  - ef_fix(S*) = the smallest grid ef whose **training-split** success rate
    ≥ S*, frozen with the router before evaluation (seed 42: ef 172).
  - Evaluated on the held-out set exactly like the router.
  - "Router better than fixed effort" means **lower mean total cost than B1 with
    non-inferior success rate on the same queries** (§16).
- **B2 (robustness): hindsight-optimal fixed effort.**
  - On the evaluation set, the cheapest non-adaptive policy reaching the
    router's **realised** success rate.
  - A single grid ef, or a randomised mix of the two bracketing grid ef values
    (linear interpolation in success rate; realisable in expectation).
  - It is tuned on the evaluation set, so it **favours the baseline**. The
    router must also beat it, which rules out a pass caused by B1's
    calibration luck.
- No other "matched recall" construction is used for the gate.

## 13. Multi-seed protocol

**Decision (proposed): the corrected gate requires 3 independent HNSW index
seeds.**
- Seeds {42, 43, 44}, each built single-threaded (reproducible).
- §13 of the design requires 2–3 seeds for core conditions, and the validation
  showed index-seed variance is the one unmeasured source of uncertainty.

**Per seed:**
- its own index;
- its own oracle curves and candidate-oracle labels for all 10,000 SIFT queries;
- its own features (the probe depends on the index);
- its own router (trained on that seed's training split);
- its own ef_fix(S*).

Ground truth depends only on the data and is shared.

**Introduced:** at the start of Phase 4b, before any router fitting.
Approximate cost per new seed: index build ~9 min, oracle curves ~13 min,
features seconds.

**Aggregation:**
- primary statistics pool all seeds, with a bootstrap stratified by seed
  (resample queries within each seed);
- per-seed results are reported alongside;
- the gate also requires every individual seed to point in the right direction
  (§16).

## 14. Train / validation / test protocol

- **Training:** the existing 8,000-query training split (seed 20261001),
  unchanged.
- **Validation:** 5-fold stratified CV inside the training split, used for
  model selection and τ calibration only. No separate validation split.
- **Test.** The existing 2,000-query test split has been seen once, and its
  outcome triggered this redesign. A second look would be outcome-adaptive.
  **Proposed: a fresh held-out "confirmation set".**
  - 2,000 vectors drawn once from `data/sift/sift_learn.fvecs` (part of the
    SIFT1M distribution, disjoint from the base set and the query set).
  - Seeded sample (seed **20261005**, Fisher–Yates as for the split).
  - Ground truth against the base set (fresh, as for any query set), oracle
    curves and features per seed.
  - Created **only after** every seed's router and ef_fix are frozen and
    hashed.
  - Used **exactly once** for the gate.
- **The old 2,000-query test split:** a disclosed, secondary "second look",
  reported next to the confirmation result and never gating.
- **Caveats (approval needed):**
  - This changes the design's "single fixed query set" (§7.2) for the gate.
  - `sift_learn` may differ slightly in difficulty from `sift_query`. The
    paired router-vs-fixed comparison is within the same set; absolute success
    levels may shift.
  - Which set serves as the evaluation set for Phases 5+ must also be decided
    (§17, decision 9).

## 15. Statistical analysis

- **One primary hypothesis** (the gate, §16): no multiplicity correction.
  Everything else is pre-declared secondary or descriptive.
- **Unit:** per query, paired across policies.
- **Cost:**
  - relative saving = 1 − mean C(router) / mean C(B1);
  - seed-stratified paired bootstrap, 2,000 resamples, seed 20261002, 95%
    percentile CI;
  - Wilcoxon signed-rank on paired per-query total cost, with matched-pairs
    rank-biserial effect size.
- **Quality:**
  - success-rate difference (router − B1) with paired bootstrap CI;
  - exact McNemar test on paired success indicators;
  - mean recall difference (secondary).
- **Regret decomposition (§11):** avoidable failure rate, mean overspend, probe
  overhead, contract regret, each with bootstrap CIs.
- **Reported, non-gating:**
  - per-seed results;
  - difficulty groups (frozen 18/48);
  - S* = 0.90 and 0.99;
  - mean-recall contract;
  - probe-accounting sensitivities;
  - LID ablation and single-feature router;
  - the old-test-split second look.

## 16. Corrected Phase 4 checkpoint

Primary configuration: per-query contract, S* = 0.95, candidates {probe, 20,
40, 83, 164, 327, 647, 1343, 2661}, additive probe, B1 baseline, confirmation
set, seeds {42, 43, 44}.

**Pre-flight (training only, before evaluation).** For every seed, the contract
oracle's saving versus that seed's ef_fix(S*) is ≥ 20%. If any seed fails,
**NOT TESTABLE**: stop and report; S* is not moved.

**PASS iff all of:**
1. **Cost:** pooled relative saving versus B1 ≥ **10%**, *and* its 95% CI lower
   bound > 0, *and* Wilcoxon p < 0.01.
2. **Quality non-inferiority:** pooled 95% CI lower bound of (router − B1)
   success rate ≥ **−0.01**, *and* the router's pooled success rate ≥ S* − 0.02.
3. **Robustness to baseline choice:** pooled saving versus B2 has a 95% CI lower
   bound > 0.
4. **Seed consistency:** on **every** seed, the point-estimate saving versus B1
   is > 0 and the point-estimate success difference is ≥ −0.01.

**FAIL** if the pre-flight passes but any of 1–4 fails. That is then a genuine
negative result under a passable gate.

Why this is not artificially easy:
- The router pays the full probe (~391 dc, about 14% of B1's ~2,734 at seed
  42), so its search must save about 24% just to break even.
- It must beat a hindsight-tuned baseline (B2).
- It must hold on all three seeds.
- It is evaluated on never-seen queries.
- The perfect-information bound (48%) assumes perfect knowledge. The previous
  router realised about 50% tier accuracy from the same features.

## 17. Exact experiment to run next (Phase 4b), after approval

1. **Seeds.** Build seed-43 and seed-44 indexes (single-threaded). Run the
   Phase 2 oracle (all 10,000 SIFT queries, same grid) and Phase 3 features
   for each, with the existing apps and new configs. Re-use the cached ground
   truth.
2. **Pre-flight (train only).** Contract-oracle feasibility per seed at S* =
   0.95 (primary candidates, additive probe). Stop if NOT TESTABLE.
3. **Router fitting (train only, per seed).** Candidate-oracle labels;
   per-candidate success models for {LR, DT2, DT3, DT4}; 5-fold CV; τ
   calibration; selection rule (§9).
4. **Freeze.** Router objects (SHA-256), τ, ef_fix(0.90/0.95/0.99) per seed,
   and this contract. Written to `docs/notes.md` before step 5.
5. **Confirmation set.** Draw 2,000 vectors from `sift_learn` (seed 20261005).
   Compute ground truth, oracle curves (full grid) and features per seed. Hash
   all artefacts.
6. **Single evaluation.** Pre-registered analysis (§15) on the confirmation set
   → gate decision (§16). Then the disclosed secondary second look on the old
   test split. Independent validation: C++ re-search of every routed query plus
   the Python replica on a sample.
7. **Stop and report.** No Phase 5 until the gate result is reviewed.

New code needed (not written): a candidate-oracle and per-candidate success
model trainer; a confirmation-set extraction step; evaluation under the new
contract. Existing C++ apps are reused through configs.
