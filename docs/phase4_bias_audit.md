# Phase 4 Redesign — Bias and Researcher-Degrees-of-Freedom Audit

Date: 2026-10-02. **Read-only audit.**
- No new evidence was generated, nothing was implemented, and no index, router
  or result was created or changed.
- The test split and the proposed confirmation set were not accessed.
- `docs/phase4_redesign.md` and `docs/notes.md` are unchanged.

Inputs: `docs/design_doc.md`, `docs/phase4_redesign.md`, `docs/notes.md`,
`CLAUDE.md`, `docs/methodology_audit.md`, `docs/phase4_methodology_validation.md`,
the Phase 1–4 code, and the existing training-only redesign evidence
(`results/redesign_evidence_20261002T125113Z/redesign_train.json`).

**Research question used as the yardstick.** Can query-level information be
used to allocate ANN search effort per query so that a specified
retrieval-quality target is achieved with less computation than a fixed
search effort?

**Timeline of information** (needed to judge "decided before seeing outcomes"):

| Event | What was known |
|---|---|
| E0 — Phase 4 pre-registration | Nothing about router performance |
| E1 — Phase 4 test result | Router 2.3× costlier than fixed ef at matched mean recall |
| E2 — Training-only audit and validation | Perfect information beats fixed ef only at high mean-recall targets. Headroom grows with stricter targets. The old tiers make the gate unpassable at R ≈ 0.95. One per-query point (success 72%): perfect-info any-ef 578 vs fixed 1,130. |
| E3 — Redesign pre-declaration (`docs/notes.md`) | Written after E1 and E2, **before** E4 |
| E4 — Redesign feasibility table | Per-query contract headroom at S* = 0.80–0.99; frozen-tier and mean-contract comparisons |

So every redesign choice was made after E1 and E2. The question for each is
whether it corrects a defect that exists independently of E1, or moves the
experiment toward conditions that E2 had already shown to favour routing.

---

## 1. Executive conclusion

**The redesign is not uniformly router-favouring, but it is still partly
OUTCOME-DEPENDENT and is not ready to freeze as written.**

- **Several changes correct genuine defects** that would be defects whatever
  the router's result. They are justified independently and can stay:
  - the router was restricted to 3 effort values while the fixed baseline
    effectively had the full 121-value grid;
  - training used a symmetric 0-1 loss while evaluation was cost-at-quality;
  - the router's operating point was uncontrolled;
  - regret went negative for failures;
  - "matched recall" was ambiguous.
- **Three choices are outcome-dependent in a router-favouring direction:**
  1. **Switching the primary contract from mean recall to per-query success.**
     Independently supported by the design (§2.3, §12, §14), but adopted after
     E2 showed it is where routing has room. It changes the opportunity
     materially: perfect-information saving at the 0.95 level is +1.5% under
     the mean contract versus +48.1% under the per-query contract.
  2. **Selecting S\* = 0.95 as the single primary level.** Its justification
     ("the design's 0.95 convention applied at system level") reuses a
     per-query recall threshold as a success-rate target, which is numerology.
     E2 had already shown that stricter targets give more headroom.
  3. **The 20% "not testable" feasibility floor.** Its value is unjustified.
     It converts would-be failures into "not testable", which is
     router-protective. It was also used circularly as an argument for
     reopening the tiers (+15.1% < 20%).
- **Three further problems are not about direction but would distort the result:**
  4. **The proposed bootstrap (resampling queries independently within each
     seed)** treats the same 2,000 queries as independent across 3 seeds. That
     gives anti-conservative CIs, which favour whichever side is ahead.
  5. **The B1 baseline overshoots S\*** by up to one grid step (smallest grid
     ef ≥ S*), while the router's τ is continuous and can hit S* exactly. This
     is a small router-favouring asymmetry.
  6. **The "router success ≥ S\* − 0.02" criterion applies only to the
     router.** That is a baseline-favouring asymmetry under distribution shift.
- **The fresh `sift_learn` confirmation set** introduces no outcome-based
  selection: nothing has ever been computed on it. But it changes the design's
  fixed query set (§7.2), and its distribution relative to `sift_query` is
  unknown. A shift would most plausibly penalise the router, whose
  feature-to-difficulty mapping must transfer.

**Bottom line.** The redesign corrects real defects, but in its current form it
still leaves room to pick conditions that favour the router. A contract that is
independent of router performance is possible (§10). It requires:
- reporting both contracts;
- no single selected S*;
- no "not testable" escape;
- a query-clustered bootstrap;
- symmetric calibration of router and baseline.

Those changes are **flagged for approval, not applied.**

## 2. Decision-by-decision audit table

Columns:
- **Pre** = decided before outcome evidence (E1/E2/E4);
- **Indep.** = independently justified by the research question or design;
- **Class** = this audit's classification;
- **Material?** = whether a reasonable alternative would materially change
  the conclusion.

| # | Decision | Pre | Indep. | Class | Mechanism | Material? |
|---|---|---|---|---|---|---|
| 1 | Primary contract: per-query success rate instead of mean recall | No (after E1, E2) | **Yes, partly.** §2.3 per-query motivation; §5.3/§7.7 per-query oracle; §12 "fraction of failed queries"; §14 averages hide failures. **Against:** §21 and §7.14 use "matched recall" language. | **OUTCOME-DEPENDENT (router-favouring in effect)** | A fixed ef must be set for the tail to meet a per-query rate; the mean contract lets fixed ef sacrifice hard queries cheaply. E2 showed this. | **Yes:** +1.5% vs +48.1% perfect-info saving at 0.95 |
| 2 | Primary S* = 0.95; 0.90 / 0.99 secondary | Value before E4, after E2 | **Weak.** The design's 0.95 is a per-query recall *threshold* (§7.7), not a *success rate*; reusing the number is not a justification. | **OUTCOME-DEPENDENT** | E2 implied that headroom rises with S*. Choosing one level among {0.90, 0.95, 0.99} is a researcher degree of freedom; 0.95 is mid-range (E4: +37.5 / +48.1 / +64.2%). | Moderately: all three exceed any floor, but the realised router margin will differ |
| 3 | 20% feasibility floor → "NOT TESTABLE" | Value before E4, after E2 | **No.** Not derived from the question or from systems practice; no rationale recorded. | **OUTCOME-DEPENDENT; router-protective** | It discards operating points where routing cannot win, relabelling a negative as "untestable". It was also used to argue against the old tiers (15.1% < 20%). | Yes for the frozen-tier argument; not binding for the ladder (37–64%) |
| 4 | Candidate ladder {probe, 20, 40, 83, 164, 327, 647, 1343, 2661} | Before E4 | **Yes.** Base 10 = k (the hnswlib minimum); factor 2 conventional; values snapped up to the Phase 2 grid; top limited by the grid. No data used. | **APPROXIMATELY NEUTRAL** (corrects a baseline-favouring defect) | The old design gave the router 3 options while the fixed comparator used 121 grid values. Equalising resolution removes that asymmetry. A finer set lowers quantisation loss (more router opportunity) but adds per-candidate model variance (less). Net effect unknown. | Possibly (√2 spacing or full grid); effect sign unknown |
| 5 | Reopening tiers 18/48/327 as router outputs | After E1, E2 | **Yes, via the resolution asymmetry (#4) and the class-balance-not-cost construction (audit M1).** **Not** via the 20% floor. | **Defect correction**, but the redesign's stated reason is circular | The tiers were classifier classes, not effort options. Defective whatever the router's result. | Yes (+15.1% vs +48.1% bound), but the old setting was the defective one |
| 6 | Replace the descriptive fixed baselines 18/48/327 by ef_fix(0.90/0.95/0.99) | After E2 | Yes: baselines tied to declared quality levels (§7.6 "chosen from the S0 curve") | **BASELINE-FAVOURING (stronger baselines)**, i.e. conservative | Old 18/48/327 included weak points (18) that no declared target needs. | No (descriptive in later phases) |
| 7 | Router objective: per-candidate P(success) + cheapest candidate with P ≥ τ | After E1, E2 | **Yes.** It directly encodes "meet a quality target with least computation"; it fixes audit C2 (cost-blind loss, emergent operating point). | **APPROXIMATELY NEUTRAL** as a formulation; remaining parameters are researcher degrees of freedom (see #7a) | The change targets a structural defect, not DT2's specific errors. | Unknown until frozen |
| 7a | Model family {LR, DT2–4}, hyperparameters, monotone fix, τ rule, selection tie rule | Partly (family from design §9) | Family yes; hyperparameters, tie rule, monotonicity method and τ rule are unspecified or ad hoc | **RESEARCHER DEGREES OF FREEDOM** (direction unknown) | Many small free choices. Each must be frozen before fitting. | Individually small; jointly unknown |
| 8 | Fully additive probe (primary); shared descent and probe-free as sensitivities | Before E4 | Yes: physically what black-box hnswlib executes | **BASELINE-FAVOURING (conservative)** | The ~118 dc identical descent is charged twice. Sensitivities are labelled. | Sensitivities would raise router savings by ~3–14 points (E4); kept non-primary, correctly |
| 8a | "Return the probe's own ef=10 result" option | Before E4 | Yes: physically real; no extra work | **APPROXIMATELY NEUTRAL** | Fixed ef 10 is equally available to the baseline (same cost as the probe). It must be applied consistently: the contract oracle and every router variant get the same option. | Small |
| 9 | B1 = smallest grid ef with train success ≥ S* | After E2 | Yes: a deployable, pre-committed baseline calibrated on the same training data | **ROUTER-FAVOURING (small)** | B1 overshoots S* by up to one grid step (seed 42 at 0.95: train success 0.9534 at ef 172), while the router's continuous τ hits S* exactly. That gives up to ~one grid step of cost (≤ ~5%) to the router. | Could matter near the 10% threshold |
| 10 | B2 = hindsight-optimal fixed at the router's realised success (gating) | After E1 | Yes: it guards against "saving cost by sliding down within the −1-point tolerance" | **BASELINE-FAVOURING (by design)** | Uses evaluation-set outcomes for the baseline only. | Makes the gate stricter; justified |
| 11 | Three oracle references (grid / candidate / contract) | After E2 | Yes: clean separation; none chosen to flatter routing | **APPROXIMATELY NEUTRAL** | The candidate and contract oracles depend on the ladder, so they inherit #4. | No |
| 12 | Regret split (avoidable failure / overspend on successes / probe / legacy) | After E2 | Yes: separates quality loss from computation loss (fixes audit M4) | **APPROXIMATELY NEUTRAL, with a hiding risk** | "Overspend on successes" is conditional on success. A router that fails more hard queries removes their large overspend from the average. | Only if read alone; the gate uses unconditional metrics |
| 13 | Gate: ≥ 10% saving, CI lb > 0, Wilcoxon p < 0.01 | **Yes (E0, the original Phase 4 pre-registration)** | 10% is a conventional "material" threshold; CI and test standard | **APPROXIMATELY NEUTRAL** | Set before any router result. | — |
| 14 | Gate: success non-inferiority, CI lb of (router − B1) ≥ −0.01 | **Yes (E0)** | Margin conventional; ~2 SE at n = 2,000 | **APPROXIMATELY NEUTRAL** | Symmetric (paired difference). | — |
| 15 | Gate: router success ≥ S* − 0.02 | After E1 (new) | Arbitrary tolerance; **applied only to the router** | **BASELINE-FAVOURING** | Under a distribution shift both policies drop below S*, but only the router is penalised. Redundant with #14 when B1 also misses. | Possibly, on a shifted query set |
| 16 | Gate: positive saving vs B2 | After E1 (new) | Yes (see #10) | **BASELINE-FAVOURING (by design)** | — | Stricter |
| 17 | Gate: every seed in the right direction | After E1 (new) | Yes: §13 core conditions across seeds; a conjunction, so no multiplicity inflation | **BASELINE-FAVOURING (mildly conservative)** | A conjunction raises false negatives, not false positives. | Only if the true effect is small |
| 18 | Seeds {42, 43, 44} | After E2 | Yes (§13). Values arbitrary but outcome-free (43 and 44 never built). **Seed 42 is the development seed:** all redesign evidence came from it. | **APPROXIMATELY NEUTRAL**, with a note on seed 42 | — | No |
| 19 | Bootstrap "stratified by seed, resample queries within each seed" | After E2 | **Statistically incorrect** for this design: the same 2,000 queries appear under every seed | **Anti-conservative (favours whichever side leads)** | Treating 3 × 2,000 correlated observations as independent understates variance. A pooled Wilcoxon would also violate independence. | Could turn a borderline result into a pass (or a clear fail into a "significant" fail) |
| 20 | Fresh 2,000-vector confirmation set from `sift_learn` (seed 20261005) | After E1 | **Partly.** A clean unseen set is a valid remedy for a once-seen test split. But it changes §7.2 (one fixed query set), and its representativeness relative to `sift_query` is unknown. | **UNRESOLVED (direction unknown; plausibly router-penalising)** | No outcome on `sift_learn` has ever been computed, so no direct selection bias. Router and B1 are both calibrated on `sift_query`; a shift hits the router's learned feature-to-difficulty map harder than B1's single ef. | Possibly |
| 21 | Old test split as a disclosed, non-gating second look | After E1 | Yes | **APPROXIMATELY NEUTRAL** (transparency) | — | — |
| 22 | Feature set: the three Phase 3 features; LID ablation | **Yes (design §9, Phase 3)** | Yes | **APPROXIMATELY NEUTRAL** | Unchanged from before E1. | — |

## 3. Every potential researcher degree of freedom

These are still open in `docs/phase4_redesign.md`. Each must be fixed in writing
before any Phase 4b computation.

1. **Contract family:** per-query success vs mean recall, and which decides the
   gate (#1).
2. **Operating point(s):** the S* value, or a set of values, and how a set is
   combined (#2).
3. **The feasibility rule:** whether it exists, its threshold, and its
   consequence (#3).
4. **Candidate set:** the ladder is fixed by rule. Still open: whether ef 4096
   (grid max) is added, and whether the probe-result option is used by every
   variant (#4, #8a).
5. **Model family and hyperparameters:**
   - LR: C, solver, scaling;
   - DT: depth set, min_samples_leaf, criterion;
   - whether models are fitted separately per candidate or as one ordinal
     model (#7a).
6. **Monotonicity method:** cumulative max (as proposed) versus isotonic
   regression in c (#7a).
7. **Probability calibration:** none, Platt or isotonic (#7a).
8. **τ rule:**
   - one global τ per S*;
   - the OOF success target exactly S* or with margin;
   - grid resolution of τ (#7a).
9. **Model-selection rule:** criterion, the 1% tie tolerance, and whether
   selection happens per S* or per seed (#7a).
10. **Handling unachievable training labels:** queries with no succeeding
    candidate: dropped, or labelled "beyond the last candidate".
11. **B1 construction:** smallest grid ef versus an exactly calibrated two-ef
    mix (#9).
12. **B2 construction:** interpolation method (linear in success rate).
13. **Tolerances:** −0.01 non-inferiority (E0) and S* − 0.02 (#15).
14. **Statistics:**
    - bootstrap scheme (#19), number of resamples, seed;
    - Wilcoxon input (pooled vs per-query seed-averaged);
    - McNemar scope;
    - α levels.
15. **Seeds:** values and count, and how a failed seed build is handled.
16. **Query set:** source, sample seed, and size; whether integrity checks
    (exact duplicates against base and queries) run before use (#20).
17. **Which secondaries are reported:** every pre-declared secondary must be
    reported whatever its result, and no non-declared analysis may be promoted
    after unblinding.
18. **Stopping rules:** what happens on partial passes, failed builds, or
    integrity failures.

## 4. Choices that are already defensible and can remain

- **Feature set** (three Phase 3 features; LID ablation only). Fixed before E1.
- **Per-query recall threshold** (tie-aware recall@10 ≥ 0.95 = 10/10). Locked
  by you before E1.
- **The doubling ladder rule.** Data-independent; it removes a resolution
  asymmetry that favoured the baseline. Freeze the exact values, and state
  that ef 4096 is excluded.
- **Reopening 18/48/327 as router outputs**, justified by the resolution
  asymmetry and audit M1. The 20% floor argument must be withdrawn as a
  justification.
- **Objective formulation:** per-candidate success probability plus
  cheapest-satisfying choice. This is the research question expressed as a
  decision rule. Its free parameters must be frozen (§3, items 5–10).
- **Fully additive probe as primary**, with shared descent and probe-free as
  labelled sensitivities. The probe-result option applies to every router
  variant and to the contract oracle.
- **The three oracle references.**
- **The regret decomposition**, provided total cost and success rate (both
  unconditional) remain the headline and gate metrics, and overspend is always
  shown with the success rate.
- **E0 gate thresholds:** ≥ 10% saving with CI lb > 0 and Wilcoxon p < 0.01;
  non-inferiority margin −0.01.
- **B2 as a gating robustness check.** Baseline-favouring, but motivated by a
  specific threat.
- **Every seed in the right direction**, and seeds {42, 43, 44}.
- **Contract-aligned descriptive baselines for later phases.**
- **The old test split as a disclosed, non-gating second look.**

## 5. Choices that need to be frozen differently before Phase 4b

All flagged; none applied.

| Item | Current proposal | Problem | Proposed replacement (needs approval) |
|---|---|---|---|
| Primary S* | 0.95 alone | No independent basis; it selects one point on a monotone headroom curve | **Co-primary S\* ∈ {0.90, 0.95, 0.99}.** Phase 4 "passes" only if the gate passes at **all three**. Per-level results are always reported. |
| Contract hierarchy | Per-query primary; mean secondary | Post-hoc switch; the alternative changes the conclusion | Keep per-query as the **gate** family, justified by §2.3/§12/§14. Pre-declare the mean-contract family R ∈ {0.95, 0.97, 0.99}, evaluated with the **identical procedure and criteria**, and reported with **equal prominence** whatever it shows. Claims are scoped to the contract. |
| Feasibility floor | < 20% → NOT TESTABLE | Unjustified value; router-protective escape; used circularly | **Remove the escape.** Before fitting, report perfect-information savings for the **full grid** and for the **ladder**, per seed and per S*, ex ante. (a) Full-grid saving < 10% at some S*: the gate at that S* is recorded as **FAIL (no routing headroom)**. (b) Ladder < 10% but full grid ≥ 10%: recorded as **candidate-set defect**, stop for review. No operating point is ever discarded. |
| B1 | Smallest grid ef with train success ≥ S* | Overshoots S*; the router's τ does not | B1 = a train-calibrated **randomised two-ef mix** reaching train success S* exactly (bracketing grid ef values, mixing weight from train). The pure smallest-ef B1 is reported as secondary. |
| Router-only tolerance | Router success ≥ S* − 0.02 | Asymmetric | Replace by a **symmetric calibration report**: realised success of router and B1 versus S*, per seed. If **both** fall below S* − 0.02, report "query-set shift" (not a router failure). If only the router does, the non-inferiority criterion (#14) already captures it. |
| Bootstrap and tests | Resample within each seed independently; pooled Wilcoxon | Ignores shared queries across seeds | **Query-clustered bootstrap:** resample query ids, carrying each query's results under all seeds. Wilcoxon on per-query **seed-averaged** paired differences (n = number of queries). McNemar per seed (exact), reported. |
| Model-side free parameters | Partly unspecified | Researcher degrees of freedom | Freeze everything listed in §10 before fitting. |
| Query set | Fresh `sift_learn` set primary | Design change; unknown shift | Approval needed (§9 of this document). Whichever is chosen, freeze it now, with integrity checks, before any router fitting. |

## 6. Choices that must be justified from the research question, not from existing results

- **The contract.** Justify it by what "a specified retrieval-quality target"
  means for users (per-query guarantees, design §2.3/§12/§14), **not** by E4's
  headroom. The redesign document currently cites E4 headroom next to the
  contract choice; the justification must stand without it.
- **The operating point(s).** Justify them as a range of service levels
  (90 / 95 / 99%) covering typical targets, **not** as the level where the
  bound clears a floor.
- **Reopening the tiers.** Justify it by the resolution asymmetry and the
  class-balance construction, **not** by "+15.1% < 20%".
- **The ladder.** Justify it by the data-independent rule (k = 10, factor 2,
  grid snapping), **not** by its E4 bounds.
- **The fresh query set.** Justify it by the once-seen status of the old test
  split, **not** by any expectation about router performance on it.

## 7. Can S* = 0.95 legitimately remain primary?

**Not as the sole primary.**
- Its stated justification conflates the design's per-query recall threshold
  (0.95) with a system-level success rate. The two quantities share only a
  number.
- There is no other independent basis for 0.95 over 0.90 or 0.99.
- E2 had already established that headroom increases with S*, so any single
  choice is open to the charge of selection.

The selection-free alternative: treat **{0.90, 0.95, 0.99} as co-primary**
with a **conjunctive** pass rule.
- A conjunction cannot inflate false positives.
- It avoids picking the level after the fact.
- It is not tuned toward the router: 0.90 has the least headroom of the three
  (+37.5% on seed 42), and the gate must pass there too.

If you prefer a single primary level, it has to be justified by an external
service-level argument, which this project does not currently have.

## 8. Can the 20% feasibility floor legitimately remain?

**No.**
- No rationale was ever given for 20%.
- It was set after E2.
- It functions as an escape hatch: an operating point where even perfect
  routing cannot materially win is a negative result about routing, and
  labelling it "not testable" hides that result.
- It was also used as evidence against the old tiers, which is circular.

Feasibility should be an **ex-ante property reported for every pre-declared
operating point**, never a rule that selects or discards operating points. The
replacement in §5 records "no routing headroom" as a FAIL and reserves "stop
for review" for a demonstrable candidate-set defect.

## 9. Can the fresh query set legitimately remain primary?

**Conditionally. This is a decision for you; it is UNRESOLVED.**

**For a fresh set:**
- The old test split has been seen once, and its result (E1) triggered the
  redesign. Some redesign reasoning was first prompted by test-level analyses:
  misrouting leaves, the error-asymmetry narrative. Decisions themselves were
  made on training evidence, but a second look at the same queries is not fully
  clean.
- Nothing has ever been computed on `sift_learn`, so it carries no
  outcome-based selection.

**Against a fresh set:**
- It breaks §7.2 ("drawn once … never changed") and comparability with Phases
  1–4.
- The difficulty and feature distribution of `sift_learn` relative to
  `sift_query` is unknown. Router and B1 are both calibrated on `sift_query`,
  but the router additionally relies on the feature-to-difficulty mapping
  transferring, so a shift is more likely to hurt the router than B1. A PASS on
  the fresh set would therefore be robust; a FAIL would be ambiguous between
  "no effect" and "shift".
- Unused `sift_query` vectors don't exist (all 10,000 are split). The training
  split cannot supply a clean hold-out, because all 8,000 training queries
  entered the audit and redesign evidence.

**Conditions under which the fresh set can be primary:**
1. Sample seed and size frozen now (seed 20261005, n = 2,000 — already
   declared).
2. Created only after all routers and baselines are frozen and hashed.
3. A pre-registered **integrity check** before use, with no outcome
   information: exact duplicates against the base set and against
   `sift_query`.
4. A pre-registered, **label-free shift report** (feature distributions only),
   reported but never used to change anything.
5. The old test split's second look reported with **equal prominence**.
   Disagreement in direction between the two sets is reported as a finding, not
   resolved by choosing one.
6. A separate decision on which set Phases 5+ use, since §7.2 requires one
   fixed query set across states.

If you consider the §7.2 change unacceptable, the alternative is a disclosed
second look at the old test split as primary. That is less clean on reuse but
fully comparable.

## 10. Proposed final frozen contract

Proposed only where justifiable independently of router performance. Items
marked **[APPROVAL]** reflect the outcome-dependent findings above and require
your decision.

| Element | Frozen value |
|---|---|
| **Quality metric** | Tie-aware recall@10. Per-query success ⇔ recall ≥ 0.95 (10/10). |
| **Contract families** | **[APPROVAL]** Gate family: per-query success rate. Co-reported family: mean recall@10, evaluated with the identical procedure and criteria, with equal prominence. Claims scoped to the contract. |
| **Operating points** | **[APPROVAL]** Per-query family: S* ∈ {0.90, 0.95, 0.99}, co-primary, conjunctive pass. Mean family: R ∈ {0.95, 0.97, 0.99}, reported. |
| **Feasibility** | **[APPROVAL]** Ex-ante report of perfect-information savings (full grid and ladder) per seed and per level. Full-grid < 10%: FAIL at that level (no headroom). Ladder < 10% with full grid ≥ 10%: stop for review (candidate-set defect). No level is discarded. |
| **Effort candidates** | {probe result (ef 10, zero extra search), 20, 40, 83, 164, 327, 647, 1343, 2661}. Rule: smallest grid ef ≥ 10·2^i; ef 4096 excluded. Used identically by every router variant and by the candidate and contract oracles. |
| **Probe accounting** | Primary: fully additive (`CountingL2Space`, probe k = 10, ef = 10). Sensitivities, labelled and non-gating: shared upper-layer descent; probe-free bound. LID ablation: charged its own ef = 20 probe instead of the core probe. |
| **Router family** | Per candidate c, a binary model of P(y ≤ c \| x), y = the candidate-oracle index. Families {LR: StandardScaler + L2 logistic regression C = 1.0, lbfgs, max_iter 2000; DT2/DT3/DT4: max_depth 2/3/4, min_samples_leaf 100, gini, random_state 20261002}. No probability recalibration. Monotonicity: cumulative maximum over c. Queries with no succeeding candidate: label y = "beyond last" (P(y ≤ c) target 0 for every c). |
| **Decision rule** | Cheapest c with P(y ≤ c \| x) ≥ τ. One global τ per (seed, S*), chosen on 5-fold stratified OOF predictions (seed 20261002) as the smallest τ on a 0.001-step grid with OOF success ≥ S*. |
| **Model selection** | Per (seed, S*): lowest OOF mean total cost among families with OOF success ≥ S*. Ties within 1% → simplest (DT2 < DT3 < LR < DT4). Refit on the full training split, then frozen with SHA-256. |
| **Feature policy** | knn_dist, centroid_dist, score_concentration only. LID: ablation, non-gating. No other features. |
| **B1** | **[APPROVAL]** Train-calibrated randomised mix of the two grid ef values bracketing training success S*, reaching S* exactly on the training split. Deterministic hash-based per-query assignment, seed declared. Secondary: smallest grid ef with training success ≥ S*. |
| **B2** | Hindsight-optimal fixed policy on the evaluation set at the router's realised success rate (two-ef mix, linear in success). Gating robustness check; labelled baseline-favouring. |
| **Oracles** | Grid oracle (Phase 2, unchanged); candidate oracle (stable reach within candidates); contract oracle (perfect information over candidates at the success target, same accounting). |
| **Regret** | Avoidable-failure rate; overspend over successes (always shown with the success rate); underspend on failures (never netted); probe overhead; contract regret = mean total cost − contract oracle at the router's realised success. Legacy regret labelled as legacy. Headline and gate metrics: mean total cost and success rate (unconditional). |
| **Statistics** | **[APPROVAL]** Query-clustered bootstrap (resample query ids; all seeds of a query together), 2,000 resamples, seed 20261002, 95% percentile CIs. Wilcoxon signed-rank on per-query seed-averaged paired cost differences, with rank-biserial. McNemar exact per seed. One gate decision per S* level; conjunctive across levels, so no α inflation. |
| **Pass criteria (per S\*)** | (1) Relative saving vs B1 ≥ 10%, CI lb > 0, Wilcoxon p < 0.01. (2) Success difference (router − B1) CI lb ≥ −0.01. (3) Saving vs B2 CI lb > 0. (4) Every seed: point saving vs B1 > 0 and point success difference ≥ −0.01. **[APPROVAL]** The router-only "≥ S* − 0.02" is replaced by the symmetric calibration report. **Phase 4 PASS** = pass at all three S*. |
| **Index seeds** | {42, 43, 44}, single-threaded builds. Seed 42 noted as the development seed; per-seed results reported. |
| **Query-set protocol** | **[APPROVAL]** Training: the existing 8,000. Evaluation: either (a) the fresh `sift_learn` set (seed 20261005, n = 2,000, created after freezing, integrity-checked, used once) as primary with the old test split as an equal-prominence second look; or (b) the old test split as a disclosed second-look primary. |
| **Stopping rules** | Stop and report, with no re-analysis and no substitution, if any of these occurs: a seed build or oracle run fails; an integrity check fails; a candidate-set defect is flagged; the gate fails at any S*; any pre-declared secondary was not computed. After unblinding, only pre-declared analyses are run. No further Phase 4 redesign without your explicit approval. |

**Can this contract be justified independently of router performance?** Mostly.
- The ladder, the objective, the accounting, the baselines, the statistics and
  the conjunctive multi-level gate are justified from the research question
  and the design.
- **The contract family is not fully separable from E2.** The choice was made
  after evidence showed which contract favours routing. That cannot be undone
  by argument. It can only be **neutralised by reporting both families with
  equal prominence and scoping claims**, which this contract does.
- **The query-set choice has unknown direction** and is your decision.

---

### Items requiring approval before anything is implemented

1. Gate on the per-query success family, with the mean-recall family evaluated
   identically and reported with equal prominence (instead of mean-recall as a
   minor secondary).
2. Co-primary S* ∈ {0.90, 0.95, 0.99} with a conjunctive pass, replacing the
   single primary 0.95.
3. Remove the 20% "not testable" floor. Replace it with the ex-ante
   full-grid / ladder feasibility report: no headroom = FAIL; candidate-set
   defect = stop.
4. B1 as an exactly calibrated two-ef mix (pure smallest-ef as secondary).
5. Replace the router-only "≥ S* − 0.02" criterion with a symmetric
   calibration report.
6. Query-clustered bootstrap; Wilcoxon on seed-averaged per-query differences.
7. Freeze the model-side parameters listed in §10.
8. Query set: (a) fresh `sift_learn` primary under the §9 conditions, or (b) the
   old test split as a disclosed second-look primary. Plus which set Phases 5+
   use.
9. Withdraw "+15.1% < 20%" as a justification for reopening the tiers; justify
   it by the resolution asymmetry and audit M1 only.
