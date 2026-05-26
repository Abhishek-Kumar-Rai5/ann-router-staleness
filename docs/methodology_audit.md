# Methodology Audit — Phases 1–4 (focus: the negative Phase 4 result)

Date: 2026-10-02. Scope: the whole pipeline to date, measured against
`docs/design_doc.md`, `docs/notes.md` and `CLAUDE.md`.

**Audit constraints honoured.** Nothing in the experiment was changed: no router
change, no retraining, no new test evaluation, no design-doc edit, and the
canonical Phase 4 results are untouched. Every new number below comes from the
**training split only**, via `python/audit_phase4_train.py`. That script reads
the train curves, train features, the frozen tiers, and the out-of-fold (OOF)
predictions saved by the Phase 4 training run; it fits nothing and asserts that
no test row is read. Output: `results/audit_phase4_20261002T105613Z/audit_train.json`.
Test-split figures are quoted only from the existing canonical report
(`results/ea8773feafbd_20261002T102325Z/test_eval/eval_report.json`).

Notation:
- **R** is mean tie-aware recall@10.
- **dc** means distance computations per query.
- **Omniscient** means perfect per-query knowledge, computed as a Lagrangian
  lower bound over the given option set (the lower convex hull of achievable
  (R, cost) pairs). It bounds *any* router limited to those options from below.

---

## 1. Executive conclusion

**The Phase 4 numbers are correct, but the Phase 4 checkpoint was unattainable
by construction.** The negative result is real for this exact configuration.
It is not evidence about the research question.

1. **No implementation bug was found that affects any result.** Several things
   were cross-checked independently and agree:
   - ground truth against the official file;
   - search versus stored curves, on 6,000 C++ re-searches;
   - distance accounting across every phase;
   - recall definitions;
   - leakage guards;
   - the selection rule, recomputed;
   - feature availability.
   The measured numbers are correct for the experiment as defined.
2. **With the frozen tiers and an additive probe, no router can win at the
   point where the trained router lands.** That point is R ≈ 0.95. On train, a
   perfect-knowledge router using the tiers {18, 48, 327} *and* paying the
   probe costs **1,284 dc**, against **1,082 dc** for fixed ef at the same R.
   The checkpoint was unpassable before any model was fitted.
3. **Routing itself has substantial headroom.** A perfect-knowledge router free
   to use any ef costs 670 dc at R = 0.952, 38% below fixed ef. Its headroom
   grows with the recall target: 59% at R = 0.99, 76% at R = 0.999. Three design
   choices remove that headroom:
   - the tier set (quantiles of oracle ef, not cost-aware);
   - charging the probe as a full extra search (it equals an ef=10 search
     exactly, and its result is discarded);
   - a symmetric-loss classification target whose errors push the router to an
     uncontrolled operating point (R ≈ 0.95), where routing gains least.
4. **The largest single loss is the training target and decision rule.**
   Between "perfect knowledge, 3 tiers, with probe" (1,284) and the trained DT2
   (2,583), +1,299 dc per query is lost to classification under a cost-blind
   0-1 loss. Of that, 597 dc per query comes from errors that send queries *up*
   to the ef = 327 tier, at about +4,200–4,500 dc each.
5. **The ~50% tier accuracy is exactly what the Phase 3 correlations predict.**
   A Gaussian-copula estimate at ρ = 0.46–0.56 gives 48–52% tertile accuracy;
   the observed out-of-fold accuracy is 49–54%. The features are informative
   but weak, and that is a genuine finding.
6. **My own Phase 3 feasibility check was misleading.** I reported "1.77×
   headroom with the probe → Phase 4 feasible". That figure was computed at
   R = 0.999, the operating point of *perfect* tier labels. The trained router
   cannot choose its operating point; its errors place it at 0.95, where the
   headroom is negative. This should have been computed across operating points.

**Verdict.** Accept the Phase 4 result as a correct measurement of this
configuration: frozen DT2, tiers 18/48/327, additive probe, unconstrained
operating point. Do **not** accept it as evidence that query-aware routing
cannot beat fixed ef on SIFT1M, nor as evidence that the features are useless.
The design, not the data, decided the outcome.

---

## 2. What the methodology intended to measure

- **Research question (design §1, §4).** Do per-query routing decisions,
  calibrated on S0, stay accurate as the index evolves, and does structural
  change predict routing failure better than update count?
- **Role of Phase 4 (§21).** A gate. "Router clearly outperforms the best
  fixed-ef baseline on S0 — must succeed before proceeding — a router with no
  S0 advantage cannot meaningfully be said to 'degrade'." Phase 4 therefore
  has to establish that routing saves effort at a recall target. Its purpose is
  *reduce search effort at a target recall* (§2.3–2.5, §5.3), not *predict a
  label*.
- **Oracle (§5.3, §7.7).** Minimum efSearch reaching a target recall ("for
  example, 0.95"), per query, per state.
- **Router (§9).** Three cheap features, "reused from a shallow initial graph
  probe"; a discrete tier output; a small decision tree or logistic regression.
- **Baselines (§7.6).** "Three constant efSearch values, chosen from the S0
  recall-versus-effort curve".
- **Regret (§7.14, §12).** "The gap between the effort the router chose and the
  effort the oracle would have chosen **at matched recall**."

## 3. What the implementation actually measures

| Element | Implementation |
|---|---|
| Oracle label | Smallest grid ef from which tie-aware recall@10 ≥ 0.95 holds at that ef and every larger grid ef. With k = 10 this means **10/10 per query**. |
| Tiers | Training-quantile tiers of oracle ef at 1/3, 2/3 and 0.99 give **18 / 48 / 327**. Tier label = smallest tier ≥ oracle ef. |
| Training target | 3-class **classification of the tier label** with symmetric 0-1 loss (DT majority leaves, unweighted multinomial LR). |
| Model selection | 5-fold CV score = fixed-ef cost ÷ router cost at the router's *own* OOF mean recall, after a failure-rate feasibility filter. |
| Router cost | Probe (a full ef=10, k=10 search, mean 391 dc, result discarded) **+** search at the predicted tier ef. |
| Comparison | Router versus the fixed-ef curve at the router's emergent mean recall (0.950 on test), versus fixed 18/48/327, and versus the oracle. |
| Regret | Router total dc minus the oracle's dc, for every uncensored query, including failed ones, where it is negative. |

**Key shift.** The *training* objective became "predict which of three bins
contains the effort needed for 10/10", while the *evaluation* objective is
"mean cost at matched **mean** recall". These are different problems. The
router's operating point is a side effect of its classification errors, not a
declared target.

## 4. Verified-correct components

| Component | Evidence | Severity |
|---|---|---|
| Ground truth | 0 / 10,000 rows differ from the official TEXMEX file beyond exact ties; float64 recomputation error 0.0 (Phase 1). | None |
| Distance accounting | One counter (`CountingL2Space`, thread-local, wraps hnswlib's own L2) counts every distance evaluation in `searchKnn`, upper layers included. Phase 1 serial = Phase 2 batch (100,000 rows identical). Probe dc = the ef=10 curve dc for all 10,000 queries; LID probe = the ef=20 curve dc (audit G). Router total = probe + search, counted once (unit test; test rows). Fixed-ef and oracle pay no probe. No double counting within any policy. | None |
| Recall | Tie-aware recall@10 with the same function (`TieAwareRecallAtK`) in Phase 1, the oracle, the features probe, the router and the fixed baselines. All Phase 4 policy recalls come from the Phase 2 curves, which equal a fresh C++ re-search exactly for router, oracle and LID ablation (6,000 rows). | None |
| Oracle construction | Stable-reach and first-reach rules differ for 1 / 10,000 queries; labels re-derived independently in Python with 0 mismatches; byte-reproducible. | None |
| Censored queries | 4659 and 8318 are 9/10 at every ef up to 4096 under every policy. Excluded from regret, labelled "hard", reported separately. | None (see m3) |
| Grid resolution | Steps of 5% or less above ef 20, +1 below, so labels overestimate by at most 5%. Below the tier resolution (18/48/327); cannot matter. | None |
| Split and leakage | Split depends only on (n, size, seed), verified against a separate Python implementation. Features use no labels. Centroid comes from base vectors. Tiers and tertiles come from train only. Training loaded 0 test rows. The test evaluation ran once and is SHA-verified. | None |
| Selection rule executed as written | Recomputed: DT3 and LR infeasible (OOF failure 0.248 > 0.238; 0.223 > 0.210); DT2 0.4187 and DT4 0.4169 within 0.02, so DT2. | None (see m1) |
| Features available at query time | Probe search, the query vector, and a centroid of the *indexed* vectors. No ground truth or label. | None |
| Feature directions | Trees and LR learn sign and threshold; no hand-coded direction. Observed: larger knn_dist goes with harder, larger centroid_dist with easier, flatter scores with harder. All physically plausible. | None |
| Matched-recall interpolation | The train fixed curve is monotone in both recall and cost. Interpolation (1,075) and the cheapest grid ef (53 → 1,078) agree within 0.3%. | None |
| Same population | All policies are evaluated on the same 2,000 test queries, censored included, and paired by query_id. | None |

## 5. Potential methodological problems

Each issue lists evidence, why it matters, whether it explains the 2.3×, and
how to test it without touching the test split.

### C1 — The checkpoint is unattainable at the router's operating point (Critical)
- **Evidence (train, audit C).**

  | R (mean recall) | Fixed ef | Omniscient any-ef | Omniscient 3 tiers | Omniscient 3 tiers + probe |
  |---|---|---|---|---|
  | 0.952 | 1,082 | 670 | 893 | **1,284** |
  | 0.97 | 1,352 | 749 | 1,182 | 1,573 |
  | 0.99 | 2,176 | 903 | 1,778 | 2,169 |
  | 0.999 | 4,565 | 1,076 | 2,191 | 2,582 |

  Free ef choice with the probe, but no tier restriction: 1,061 at 0.952
  (+2% headroom), 1,140 at 0.97 (+16%), 1,294 at 0.99 (+41%), 1,467 at 0.999
  (+68%).
- **Why it matters.** At R ≈ 0.95, where the trained router lands, even perfect
  knowledge cannot beat fixed ef with these tiers and this probe accounting.
  The checkpoint, as operationalised, tested the configuration, not routing.
- **Explains the 2.3×?** Partly. It sets the floor: the best possible router
  here is already 1.19× worse than fixed. The remaining gap is C2.
- **Test without contaminating test.** Already done on train (audit C). Extend
  to more R values and option sets on train only.

### C2 — The training target and decision rule don't match the objective (Critical)
- **Evidence (train OOF DT2, audit D).** Error prices relative to the true tier:

  | Error (true → predicted) | Count | Effect per query |
  |---|---|---|
  | low → high | 341 | **+4,544 dc** |
  | med → high | 767 | **+4,204 dc** |
  | low → med | 799 | +441 dc |
  | med → low | 955 | −451 dc, recall −0.163, 100% fail |
  | high → low | 402 | −3,975 dc, recall −0.253, 100% fail |
  | high → med | 809 | −3,924 dc, recall −0.151, 100% fail |

  Spread over all 8,000 queries: conservative errors add **641 dc** per query
  (597 of it from errors *to* "high"); aggressive errors save 650 dc but lose
  **0.047 mean recall**.

  The perfect-knowledge 3-tier allocation at R = 0.952 sends only **114 of
  8,000** queries to "high". DT2 sends about 2,482 there (OOF).
- **Why it matters.**
  - The 0-1 loss treats a ~4,300 dc overspend the same as a 441 dc one.
  - Majority-vote leaves ignore that "high" costs ~5× "med".
  - Nothing in training sets the operating point. Aggressive errors pull R
    down to 0.95 by themselves, exactly where routing gains least (C1).
- **Explains the 2.3×?** Yes, the largest component. The decomposition at
  R = 0.952 (train): fixed 1,082 → omniscient any-ef 670 → restricted to tiers
  893 (+223) → + probe 1,284 (+391) → trained DT2 2,583 (**+1,299**).
- **Test without contaminating test.** Train CV only: keep the same model
  family, derive calibrated tier probabilities (or per-tier success
  probabilities), and choose the tier that minimises expected cost subject to a
  pre-declared R*. See §12.

### M1 — Tier construction is not cost-aware (Major)
- **Evidence (audit E).**

  | Tier | Train oracle ef, median (p10–p90) | Overspend when correctly classified |
  |---|---|---|
  | high (327) | 87 (53–210) | 2,979 dc per query (5,233 at 327 vs 2,255 at own oracle ef) |
  | med (48) | 30 | 229 dc |
  | low (18) | 10 | 70 dc |

  Omniscient any-ef versus omniscient 3-tier: 670 vs 893 at R = 0.952; 1,076
  vs 2,191 at R = 0.999.
- **Why it matters.** Tiers at equal-*count* quantiles of ef balance classes,
  which suits a classifier. They do not minimise cost. The 0.99-quantile top
  tier makes "high" expensive even when correct and catastrophic when wrong.
  The ~10× cost ratio between adjacent top tiers (48 → 327) amplifies every
  conservative error.
- **Explains the 2.3×?** Yes, through C1 (+223 dc even with perfect
  knowledge) and C2 (the price of "→ high" errors).
- **Fairness of "tiers = baselines".** As a comparison this is fair: the
  router chooses among the same values the baselines use. But fixed 18 / 48 /
  327 are weak points; only the full-curve matched comparison is meaningful,
  and that was reported. Wording deviation: §7.6 says baselines are "chosen
  from the S0 recall-versus-effort curve", while they were taken from oracle
  quantiles at your instruction (m5).
- **Test without contaminating test.** Omniscient allocation on train for
  candidate tier sets chosen by a train-only cost-aware rule (§12).

### M2 — The probe is charged as a full extra search, with no reuse (Major)
- **Evidence.**
  - Design §9: "reused from a shallow initial graph probe."
  - Implementation: the probe is a separate `searchKnn(k=10, ef=10)`, dc
    identical to the ef=10 search for all 8,000 train queries. Its result
    (mean recall 0.71) is discarded, then the routed search re-runs from the
    entry point.
  - Mean probe cost 391 dc is 74% of an ef=18 search, 36% of the matched fixed
    cost, and about 95% of the omniscient headroom at R = 0.952 (1,061 vs 1,082).
- **Why it matters.** The design envisaged *reusing* probe work. Under
  black-box hnswlib, search state can't be resumed, but the probe's *result*
  is a valid ef=10 answer. The additive count is an upper bound and search-only
  is a lower bound; the true deployable cost depends on a design decision.
- **Explains the 2.3×?** No, not alone: DT2 search-only is still 2,192 vs
  1,082 on train (2.0×), and 0.51× on test. It is decisive for whether *any*
  3-tier router can win at R ≈ 0.95 (C1).
- **Test without contaminating test.** Report both accounting bounds on train.
  Optionally evaluate on train an option set that includes "accept the probe
  result" (ef 10, zero extra cost).

### M3 — Per-query labels versus a mean-recall evaluation metric (Major)
- **Evidence.** The labels mean 10/10 per query (0.95 at k = 10). The matched
  comparison is on mean recall. On train, fixed ef 53 reaches mean 0.95 with
  71.0% of queries at 10/10, 17.5% at 9/10 and 11.5% lower.
  Oracle (10/10 for everyone) costs 1,137 dc; an omniscient router at mean
  0.952 costs 670.
- **Why it matters.** Labels ask for far more (perfect recall) than the
  evaluation rewards (mean 0.95, where 9/10 is "almost free" for fixed ef).
  The design (§7.7) gives 0.95 as an example and does not say whether it is
  per query or a mean.
- **Explains the 2.3×?** Not by itself. At matched *failure rate* the verdict
  is the same: fixed ef 56 costs 1,130 vs DT2 2,583 on train (test: 27.9% vs
  30.2%, not significant). But it is why the router's operating point is
  uncontrolled (C2).
- **Test without contaminating test.** Declare one objective (mean recall R*,
  or per-query success S*) and evaluate every policy at it on train CV.

### M4 — Regret is not "at matched recall" (Major for later phases)
- **Evidence.** The design (§7.14, §12) defines regret at matched recall. The
  implementation computes total dc minus oracle dc for every uncensored query.
  For the 556 test failures it is negative (−648 dc average): effort "saved"
  by missing the target.
- **Why it matters.** Regret is the H1 robustness measure for Phases 6–8. Its
  current mean mixes overspend with recall loss and can *improve* when the
  router becomes more aggressive under staleness.
- **Explains the 2.3×?** No.
- **Test.** Redefine on paper: effort regret only on queries where the router
  met the target, failures reported as a separate rate (or a log-ef regret).
  No new evaluation is needed to decide.

### M5 — The Phase 3 feasibility statement was evaluated at the wrong operating point (Major, process)
- **Evidence.** `docs/notes.md` Phase 3 says "1.77× headroom … Phase 4 is
  feasible". That compared perfect tier labels (R = 0.999) with fixed ef at
  0.999. The audit shows the trained router operates at R = 0.95, where the
  same bound is negative.
- **Why it matters.** It supported proceeding to Phase 4 without examining
  feasibility across operating points or probe accounting.
- **Explains the 2.3×?** No. It explains why the result was surprising.

## 6. Potential implementation problems

Nothing found that changes any Phase 1–4 number. Minor items:

| ID | Issue | Evidence | Effect | Severity |
|---|---|---|---|---|
| m1 | Feasibility filter is a hard threshold on a noisy quantity | It excluded the two best-scoring candidates (DT3 0.457, LR 0.456) by 1.0–1.3 pp failure-rate margins. | Selection differs from the best score, but every candidate scores ≤ 0.46 and all lose. | Minor |
| m2 | LID ablation is charged both probes | The LID probe is a k=20, ef=20 search whose top-10 already contains everything the core features need. Mean probe 564 + 391. | Overcharges the ablation by ~391 dc. Its search-only ratio (0.48) still loses. | Minor |
| m3 | Oracle "all queries" summary includes the 2 censored queries at their ef=4096 cost | `summary.oracle` mean 1,131.7 includes 19,977 + 18,970 dc. An uncensored summary exists. | About +19 dc on the oracle mean; conflicts with "do not pretend 4096 is their required effort". | Minor |
| m4 | Matched fixed ef chosen from test means | ef 53 picked as the cheapest grid ef with test mean recall ≥ the router's. | Favours the baseline slightly; interpolation gives 1,075 vs 1,078. | Minor |
| m5 | Baseline derivation wording | §7.6 "chosen from the S0 recall-vs-effort curve" vs oracle-distribution quantiles (your instruction). | None on the matched comparison. | Minor |
| m6 | Significance presentation | p-values down to 1e-184 reflect n = 2,000 and very large effects. No multiple-comparison correction, which is irrelevant at these magnitudes. | Interpretation only. Effect sizes and CIs are the informative quantities. | Minor |

## 7. Severity summary

| ID | Issue | Severity | Explains the 2.3×? |
|---|---|---|---|
| C1 | Checkpoint unattainable at the router's operating point (tiers + additive probe) | **Critical** | Sets the floor (1.19× even with perfect knowledge) |
| C2 | Target and decision rule misaligned with cost-at-recall; operating point uncontrolled | **Critical** | Largest component (+1,299 dc) |
| M1 | Tiers not cost-aware (ef quantiles; 0.99-quantile top tier) | **Major** | Yes, via C1 and C2 |
| M2 | Probe charged as a full extra search; design says "reused" | **Major** | Not alone (search-only still 2.0×); decisive for C1 |
| M3 | Per-query 10/10 labels vs mean-recall evaluation | **Major** | No (verdict robust), but drives the operating point |
| M4 | Regret not at matched recall | **Major** (Phases 6–8) | No |
| M5 | Phase 3 feasibility judged at R = 0.999 only | **Major** (process) | No |
| m1–m6 | See §6 | Minor | No |
| Pipeline correctness (§4) | | None | No |

## 8. Phase 4 conclusions that remain valid

1. The frozen DT2 router, with tiers 18/48/327 and an additive probe, is
   dominated by fixed ef on S0. Mean recall at matched cost: test 0.43×, train
   OOF 0.42×. Matched failure rate gives the same verdict. Robust and
   replicated.
2. The three core features carry real but limited signal: tier accuracy is
   about 50%, consistent with Spearman ρ ≈ 0.36–0.46 (expected 44–52%).
3. Error asymmetry is structural: every aggressive error fails the target and
   no conservative error does; misroutes to the top tier dominate cost.
4. Misrouting is systematic. In crowded-but-hard neighbourhoods (small
   knn_dist, low tier predicted) 55% fail; the "med" leaf is uninformative.
5. The pipeline (ground truth, oracle, features, curves, evaluation, leakage
   controls) is correct and reproducible.

## 9. Conclusions that should NOT yet be trusted

1. "Per-query routing cannot beat fixed ef on SIFT1M S0." Not shown:
   perfect-knowledge headroom is 38–76% without the probe and 2–68% with it,
   depending on R.
2. "The Phase 3 features are insufficient for routing." Not shown: the
   decision rule is cost-blind, so feature limits and decision-rule
   misalignment are confounded.
3. "Phase 4 failed because the router is a poor classifier." Only partly: even
   a perfect classifier could not pass at R ≈ 0.95 under this accounting.
4. The Phase 3 note "1.77× headroom → feasible". Valid only at R = 0.999.
5. Mean regret figures as "routing regret at matched recall" (M4).
6. Any reading of the Phase 4 failure as an answer to H1–H5.

## 10. Recommended corrections, by priority

All need your approval; none has been made.

1. **Declare the objective and operating point (resolves M3, C1).** Choose
   either a mean-recall target R* or a per-query success rate S*, fixed before
   any new fitting from train-only reasoning. The audit table suggests routing
   is viable around R* ≥ 0.97–0.99 and not at 0.95. Compare all policies at
   that point.
2. **Decide probe accounting (M2).** Options:
   - (a) keep it additive (strict, as now);
   - (b) report both additive and search-only bounds;
   - (c) add "accept the probe result" (ef = 10, zero extra cost) as an
     option. This changes the frozen tier set.
3. **Align the training target and decision rule with the objective (C2).**
   Keep the design's model family (DT/LR) and features, and change only what is
   predicted and how the tier is chosen. Candidates, conceptually:
   - **(a) Tier classification** (current): cheapest, but cost-blind.
   - **(b) Regression of log oracle ef plus a safety margin**: uses ordinal
     information, but the margin re-introduces an operating-point choice.
   - **(c) P(success | x, ef) per tier**, choosing the cheapest tier with
     P ≥ τ, where τ sets S*: aligned with a per-query objective.
   - **(d) E[recall | x, ef] per tier plus Lagrangian allocation to hit R***:
     aligned with the mean-recall objective.

   (c) and (d) are justified, (a) is not, and (b) is acceptable but weaker.
   All stay inside the design's "decision tree / logistic regression" scope.
4. **Revisit the tiers (M1).** Your Phase 3 decision froze them. A train-only,
   cost-aware rule (for example, three tiers minimising the omniscient cost at
   R* on train) would be needed. Any change must happen before Phase 5, since
   §13 requires identical baselines across states from then on.
5. **Redefine regret per design §12 (M4)** before Phases 6–8.
6. **Minor fixes:** LID ablation probe accounting (m2), oracle censored-row
   labelling (m3), and a tolerance or ranking instead of a hard feasibility
   filter (m1).
7. **Test-set protocol.** The test split has been used once. Options:
   - (a) a single, pre-registered, disclosed second evaluation of the revised
     router on the same 2,000 queries;
   - (b) a fresh held-out query set, for example from `sift_learn`. That is the
     same dataset, but it changes the "fixed query set" design and needs
     approval.

   In both cases the revised router must be selected on train CV only.

## 11. Should the negative Phase 4 result be accepted?

**Yes, as a correct measurement of the configuration that was run.** It is
reproducible, independently validated, and consistent with the Phase 3
correlations. It can be reported as: *"A 3-tier router trained on tier labels
with symmetric loss, with ef-quantile tiers and an additive probe, does not
beat fixed efSearch at matched recall on SIFT1M."*

**No, as a test of the Phase 4 research gate.** The gate asks whether routing
can beat fixed ef at a recall target. The configuration made that impossible
at the operating point it landed on, independent of model quality (C1). The
correct scientific status is "gate not yet properly tested", not "gate
failed".

## 12. Exact next experiment required

**Phase 4b feasibility study: training split only, no test-split access.**
Each step needs your approval of the items in §10 first.

1. **Operating-point and accounting table (train, no model).** Extend the audit
   table to R ∈ {0.95, 0.97, 0.98, 0.99, 0.995, 0.999} (and S* equivalents) for
   the option sets {current tiers; current tiers + probe-result option; full
   grid}, under both probe accountings. Output: the region where routing has at
   least 10% headroom with probe accounting. *Pre-declare R* (or S*) from it.*
2. **Objective-aligned decision rule (train 5-fold CV, nested).** For the
   existing family (DT2–DT4, LR) on the same three features, fit per-tier
   success probabilities (or calibrated tier posteriors). Choose tiers by
   expected-cost minimisation at R*, with any threshold or multiplier tuned
   inside the training folds only. Score = OOF cost versus the fixed-ef curve
   at the achieved R. LID stays an ablation.
3. **Tier set (only if you approve reopening it).** A train-only cost-aware
   tier choice, compared under step 2.
4. **Gate before any test use.** Proceed to a single disclosed test evaluation
   only if CV shows at least 10% savings at R* with a bootstrap CI excluding 0.
   Otherwise record a robust negative S0 result and bring the design question
   (§21) back to you.

---

### Appendix: provenance of audit numbers
- Train-only diagnostics: `python/audit_phase4_train.py` →
  `results/audit_phase4_20261002T105613Z/audit_train.json` (sections A–G).
- Test numbers quoted (not recomputed):
  `results/ea8773feafbd_20261002T102325Z/test_eval/eval_report.json`.
- Expected-accuracy estimate: Gaussian copula, 200,000 draws, seed 20261003
  (audit F). Observed OOF accuracy: DT2 0.491, DT3 0.510, LR 0.540, DT4 0.520;
  two-step errors 9.3% (expected 8–10%).
