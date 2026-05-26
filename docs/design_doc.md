# Router Robustness Under Index Staleness

**Research Design Document — Project 1**

---

## 1. Overview

**Core research question:** Do per-query ANN search-effort routing decisions, calibrated on one index state, remain accurate as that index evolves through inserts and deletes — and can properties of the index change, rather than simple update counts, predict when they stop being accurate?

This document specifies what the project investigates, why it matters, how the system and experiments will be built, and how the results will be judged. It assumes the reader understands general computer science and machine learning, but not ANN-specific internals, and it is written to stand on its own without reference to any prior discussion that produced it.

---

## 2. Background

### 2.1 Approximate nearest-neighbour search

Given a query vector, a nearest-neighbour search finds the vectors in a large collection that are closest to it under some distance measure. Doing this exactly requires comparing the query against every vector in the collection, which becomes too slow once the collection reaches millions of vectors. **Approximate nearest-neighbour (ANN) search** trades a small, controllable amount of accuracy for large gains in speed, using an index structure built in advance.

### 2.2 HNSW and the search-effort parameter

**HNSW (Hierarchical Navigable Small World)** is one of the most widely used ANN index structures. It organises vectors into a multi-layer graph, where higher layers contain fewer points and allow the search to move quickly toward the right region of the space, and lower layers refine the search locally. A query is answered by greedily traversing this graph, starting from an entry point and repeatedly moving to closer neighbours.

The search does not have to explore the graph exhaustively. A parameter commonly called **efSearch** controls how large a candidate list the traversal keeps at each step: a small efSearch stops exploring earlier and is fast but may miss the true nearest neighbours; a large efSearch explores more of the graph and is slower but more accurate. efSearch is the main knob controlling the speed–accuracy trade-off of a query.

### 2.3 Why different queries need different amounts of effort

A single global efSearch setting is a compromise. Some queries — those falling in sparse, easily separable regions of the vector space — reach the correct answer with a small candidate list. Others — those falling near cluster boundaries, in dense regions, or in areas with many similarly-distant points — need a much larger candidate list to reach the same accuracy. A fixed global efSearch is therefore either wasteful for easy queries or insufficiently accurate for hard ones.

### 2.4 Query-aware ANN routing

**Query-aware routing** addresses this by choosing efSearch on a per-query basis instead of globally. The idea is to compute cheap, query-level characteristics at search time — for example, how far a query is from the overall data centroid, or how crowded its local neighbourhood appears to be — and use them to predict how much search effort that particular query needs. A **router** is the component that makes this prediction and selects a search budget accordingly.

### 2.5 Why this works when the index is static

A router is typically calibrated against a fixed index: its notion of "this query looks easy" or "this query looks hard" is learned from, or reasoned about in terms of, the vector distribution and graph structure as they exist at calibration time. As long as the index does not change, the relationship between a query's characteristics and the effort it actually requires stays stable, so a router calibrated once continues to make sensible decisions.

### 2.6 What changes when the index evolves

In practice, vector indexes are rarely static. New vectors are inserted as new data arrives; old vectors are deleted as data is removed or expires. Each such change can alter the local neighbourhood structure around nearby queries — the graph edges near a query may change, the local density may increase or decrease, and the true nearest neighbours themselves may change. The query's own characteristics, if recomputed, may shift; and even where they do not shift, the amount of search effort actually required to reach the true neighbours may.

### 2.7 Why an old routing decision may become unreliable

A router that was calibrated against the index at one point in time has learned a relationship between query characteristics and required effort that was true of that index state. If the index evolves and this relationship changes — either because the query's own features drift, or because the *same* feature values now correspond to a *different* amount of required effort — the router's decisions may no longer be accurate, even though nothing about the router itself has changed.

### 2.8 Routing robustness under index staleness

This project studies exactly that gap: **whether, how, and why a frozen routing policy's decisions degrade as the index it was calibrated on becomes stale**, and whether the degree of index change can be measured in a way that predicts when a router's decisions have stopped being trustworthy.

---

## 3. Research Motivation

It is important to separate two related but distinct problems:

- **Dynamic ANN indexing** is about how the index structure itself is maintained as data changes — for example, how to insert and delete efficiently, when to rebuild versus incrementally repair the graph, and how index quality degrades over time. This is a systems and data-structures problem.
- **Routing robustness under index staleness**, the subject of this project, takes index maintenance as given (via an existing library) and instead asks how the *decision layer on top of the index* — the router — behaves as the index beneath it changes.

This project does **not** attempt to build a new HNSW variant, improve index maintenance algorithms, or reinvent index update mechanics. hnswlib's existing insert and delete operations are used as-is. The research focus is the interaction between four things: **query characteristics, ANN search effort, index evolution, and the robustness of a routing policy connecting the first two as the third occurs.**

This question sits at the intersection of query processing (how a system decides how to answer a query) and index maintenance (how the underlying data structure changes), which is why it is framed as a query-processing robustness question rather than an indexing-algorithm question.

---

## 4. Research Questions and Hypotheses

**Primary research question:** Do per-query ANN routing decisions, calibrated on one index state, remain accurate as the index evolves — and does a structural measure of index change predict routing failure better than a simple update count does?

**Secondary questions:**
- Does routing degradation, if it occurs, spread evenly across queries or concentrate in a structurally identifiable minority?
- Do different routing signals degrade at different rates — are some more robust to index evolution than others?
- Is a raw update count (how many vectors were inserted or deleted) a poor proxy for staleness compared to a measure of structural change?
- Can a small amount of recalibration, using a small freshly-sampled query set, recover most of any lost routing accuracy without a full router retrain?

**Hypotheses.** These are stated as testable claims to be evaluated by experiment, not as assumed conclusions. Any of them may be confirmed, partially confirmed, or refuted, and each outcome is treated as a valid scientific result.

- **H1 — Degradation with magnitude.** As the index accumulates more change, the router's decisions become progressively less well-matched to what the query actually needs, measured as increasing "routing regret" (defined in Section 8).
- **H2 — Concentration.** If degradation occurs, it is not spread evenly across all queries but concentrates in a structurally distinct subset of them.
- **H3 — Differential feature stability.** Different routing signals are not equally robust to index evolution; some (for example, a query's distance to the overall data centroid, which reflects global position) may remain stable under local changes, while others (for example, local neighbourhood density, which reflects the immediate surroundings) may be more sensitive.
- **H4 — Structural measures outperform update count.** A measure of the structural change in the index — such as how much a query's true nearest-neighbour set has shifted — predicts routing regret better than the raw number or fraction of vectors that were inserted or deleted.
- **H5 — Cheap recalibration recovers most of the loss.** Refitting the router on a small, freshly sampled set of queries against the evolved index recovers most of the routing accuracy lost to staleness, without requiring a full retrain on the whole dataset.

**What a meaningful negative result looks like.** If H1 fails and routing remains accurate well beyond the range of index change tested, this is a genuinely useful finding: it would suggest that, at least for the update patterns and magnitudes studied, routers can be recalibrated infrequently, which has direct practical value. Similarly, if H4 fails and raw update count predicts regret just as well as any structural measure, that is a useful simplification, not a failed experiment — it means practitioners do not need to compute more expensive structural statistics.

---

## 5. System Concept

### 5.1 The base pipeline (single index state)

```
Dataset → vector index (HNSW) → query
       → query-level feature extraction
       → router → selected ANN search budget (efSearch tier)
       → search → retrieved neighbours
       → evaluation (recall, effort) against ground truth
```

A query arrives, cheap features are computed about it relative to the current index, the router uses those features to choose a search-effort tier, HNSW search is run at that effort level, and the result is evaluated for accuracy and cost against the exact (brute-force) nearest neighbours.

### 5.2 The evolution pipeline (across index states)

```
Initial index S0 → controlled updates → evolved states S1, S2, ...
   → same fixed query set, at every state
   → same frozen router (calibrated only on S0)
   → evaluate whether routing decisions remain appropriate
```

The index begins at an initial state S0. Controlled updates (insertions, and separately, deletions) are applied to produce a series of evolved states. The same set of queries is issued against the index at every state, and the router — fitted once, on S0, and never modified — is used to select search effort at every state. The central measurement is whether the router's S0-calibrated decisions remain good decisions as the index moves away from S0.

### 5.3 Four points of comparison

- **Static (frozen) router** — fitted once on S0 and evaluated unchanged at every subsequent state. This is the main object of study: does *this* router's behaviour hold up?
- **Recalibrated router** — refit at each state on a small, freshly sampled set of queries. This is not the primary subject of study; it exists to show how much of any lost accuracy is recoverable, and at what cost, providing an upper bound on what recalibration could achieve.
- **Oracle search effort** — for each query at each state, the minimum efSearch that reaches a target recall, found by direct search against that state's own ground truth. This represents the best possible per-query effort choice and is the reference point against which routing quality is measured.
- **Fixed-efSearch baselines** — a small number of constant efSearch values (low/medium/high), applied identically at every state with no adaptation at all. These represent the simplest alternative to routing and establish whether adaptive routing is worth its complexity in the first place.

---

## 6. Research Scope

### 6.1 In scope, and why

| Choice | Reason |
|---|---|
| **HNSW via hnswlib** | Small enough to read and instrument in full; supports native incremental insertion and lazy deletion; the reference implementation most comparable adaptive-ANN work already builds on. |
| **Insertion as the primary evolution mechanism**, with both in-distribution and out-of-distribution inserts | Insertion is the most common form of real-world index growth, and separating in-distribution from out-of-distribution inserts directly tests whether *how much* changes matters as much as *what* changes — a central part of the primary research question. |
| **Lazy deletion as a secondary mechanism** | hnswlib supports this natively; deletion is a distinct and practically important form of evolution, but studied with a narrower experimental grid than insertion to keep scope manageable. |
| **~1M-vector index scale** | Large enough that update magnitudes and structural changes are meaningful rather than dominated by noise, small enough that ground truth can be recomputed repeatedly across many index states within reasonable time and hardware. |
| **SIFT1M as the primary dataset** | A standard, well-understood ANN benchmark, useful for developing and validating the methodology quickly before committing to more expensive runs. |
| **One modern high-dimensional embedding dataset as secondary validation** | Confirms that findings are not an artifact of SIFT1M's relatively low (128) dimensionality, and reflects the kind of embeddings actually used in current retrieval systems. |
| **Controlled, discrete update magnitude levels** (not a continuous sweep) | Keeps the experimental matrix tractable while still covering a meaningful range of index change. |
| **Query-level routing** (not index-structure changes) | Matches the project's focus, described in Section 3, on the decision layer rather than the index itself. |
| **Structural staleness analysis** | Directly required to test H4, the comparison between naive and structural staleness measures. |

### 6.2 Deliberately out of scope

- **GPU/CUDA** — the research question concerns routing accuracy and structural analysis, not raw throughput at scale; nothing here requires GPU acceleration.
- **Multiple ANN algorithm families** (IVF, DiskANN, etc.) — a single, well-understood index family is sufficient to answer the research question and keeps the study tractable.
- **Reinventing or modifying HNSW itself** — the project consumes hnswlib's insert/delete/search operations as-is; it does not alter or reimplement them.
- **A full online-learning router** — the project studies a frozen router's degradation and a simple recalibration comparison, not a continuously adapting router; that would be a different research question.
- **Graph-maintenance algorithm design** — how hnswlib internally repairs the graph on update is out of scope; it is treated as a black box.
- **Adversarially engineered local updates** — updates are drawn from controlled distributional conditions (in- or out-of-distribution), not hand-crafted to target specific queries; any locally concentrated structural change is analysed where it emerges naturally, not manufactured.
- **A production-scale vector database system** — an ANN library plus a custom experimental harness is sufficient; no full database system is needed.
- **Vector replacement/update as a distinct mechanism** — mechanically equivalent to a delete followed by an insert of the same identity, so it does not introduce a genuinely new axis of variation beyond what insertion and deletion already cover.

---

## 7. Research Design

1. **Initial index state (S0).** An HNSW index is built once over the base dataset. All subsequent evolved states are derived from this same starting point.
2. **Query set.** A single set of query vectors is drawn once, before any experiments are run, and is never changed for the remainder of the project.
3. **Train/test split.** The query set is split into a training portion, used only to fit the router and to generate the oracle labels the router is trained on, and a test portion, used for every reported result. The two never overlap.
4. **Query difficulty stratification.** Test queries are divided into difficulty tertiles (easy/medium/hard) based on their S0 oracle-effort value, decided at design time so that subgroup analysis (Section 12) is planned rather than fitted to the data afterward.
5. **Ground-truth computation.** Exact brute-force nearest neighbours are computed for every query, separately at every index state.
6. **Fixed-efSearch baselines.** Three constant efSearch values, chosen from the S0 recall-versus-effort curve, applied unchanged at every state.
7. **Oracle search effort.** For each query, at each state, the smallest efSearch that reaches a target recall (for example, 0.95) against that state's own ground truth.
8. **Router training.** The router is fitted once, using only the S0 training split and its corresponding oracle labels.
9. **Frozen router evaluation.** The same fitted router, unmodified, is applied to the test query set at every subsequent index state.
10. **Recalibrated router comparison.** A separately refit version of the router, retrained at each state on a small freshly sampled query set, is evaluated alongside the frozen router as an upper-bound recovery reference — not as the project's central result.
11. **Index evolution.** Controlled updates are applied to S0 to produce a series of evolved states, varying in magnitude and, for insertion, in-distribution versus out-of-distribution character.
12. **Re-running the same queries after each state.** The identical test query set is issued against the index at every state, so that results are directly comparable query-by-query across states.
13. **Measuring recall and search effort.** At every state, for every policy (fixed, static-router, recalibrated-router, oracle), recall against that state's ground truth and the actual search effort expended (distance computations, and latency as a secondary measure) are recorded.
14. **Measuring routing regret.** The central robustness measure — the gap between the effort the router chose and the effort the oracle would have chosen at matched recall — is computed per query, per state, for the static router.

**Why the same query set is retained across states.** Using an identical, fixed set of queries at every index state is what makes state-to-state comparisons *paired*: any change observed in a given query's outcome can be attributed to the index having changed beneath it, rather than to a different, possibly easier or harder, set of queries being used. This is what enables the paired statistical analysis described in Section 12.

**Why ground truth must be recomputed at every state.** Once vectors are inserted or deleted, the true nearest neighbours for a query can change. Reusing S0's ground truth at a later state would silently mean recall is being measured against the wrong answer, which would invalidate every downstream result — this is treated as a critical safeguard, not an optional refinement.

---

## 8. Index Evolution / Staleness Model

**S0** is the initial, unmodified index. From it, two primary families of evolved states are produced:

- **Insertion**, at four to six controlled magnitude levels, crossed with two distribution conditions: **in-distribution** inserts (drawn from the same distribution as the original data) and **out-of-distribution** inserts (drawn from a distinctly different distribution). This directly tests whether the *amount* of change or the *character* of what changed matters more.
- **Deletion**, via hnswlib's native lazy-delete mechanism, studied secondarily with a narrower set of magnitude levels.
- **Mixed insert/delete churn** is treated as an optional, tertiary condition, not part of the core grid.

**Why update count alone may not adequately describe staleness.** The number or fraction of vectors changed is an easy quantity to report, but it says nothing about *where* in the vector space those changes occurred or how much they altered the structure that routing decisions actually depend on. A small number of updates concentrated near a query's neighbourhood could plausibly matter more than a much larger number of updates far away from it. This motivates comparing update count directly against structural alternatives, rather than assuming either is adequate in advance.

**Candidate structural measures:**

1. **Update fraction** — the proportion of the index changed since S0. The naive baseline every structural measure must be compared against.
2. **Neighbourhood-overlap decay** — the overlap between a query's true nearest-neighbour set at S0 and at the current state, measured via ground truth on a sampled set of anchor points.
3. **Local density / centroid drift** — change in a cheap, live k-NN-distance-based proxy for a query's local surroundings, computable without ground truth.
4. **Distributional drift** — a simple statistical distance between the original and newly inserted vector distributions, meaningful primarily for the out-of-distribution insertion condition.
5. **Graph edge churn** — the fraction of a node's HNSW adjacency list that has changed since S0. This is the measure closest to the actual mechanism of interest, but requires instrumenting hnswlib internals to snapshot adjacency lists; it is included if this proves straightforward, and otherwise deferred to a later phase.

**Which are usable where:**

- **Live/router-usable:** local density/centroid drift only — this is the sole candidate that is both informative and computable without knowing the answer in advance, so it is the only structural measure that could plausibly be used as a real router input in a deployed system.
- **Post-hoc research measurements only:** neighbourhood-overlap decay, distributional drift, and graph edge churn. These are used to *characterise* staleness and to test H4, but never as router features.
- **Optional/extension:** graph edge churn, conditional on instrumentation cost.

**Leakage note.** Neighbourhood-overlap decay requires ground truth at both the original and current state to compute, which means it directly encodes information about the outcome (how much the true answer has changed) that the project is trying to predict. Using it as a router input would leak the answer into the question being asked, so it is restricted to post-hoc analysis only, never to router calibration or prediction.

---

## 9. Router

**Inputs.** A small, deliberately minimal set of features, each computed live and cheaply relative to the current index state:

- **Local k-NN distance proxy** — an inexpensive measure of how crowded a query's immediate surroundings are, reused from a shallow initial graph probe.
- **Centroid distance** — the query's distance from the overall data centroid, reflecting how central or peripheral it is in the vector space.
- **Score-concentration ratio** — how much the distances to the closest few candidates in a shallow probe differ from one another, an information-retrieval-style measure of search confidence.
- **Local intrinsic dimensionality (LID)**, included as a second-tier, ablation-only addition rather than a core feature, since it is more expensive to compute and its value is uncertain relative to the three core features above.

**Why this minimal set.** Each additional feature adds cost and interpretive complexity. Starting from three inexpensive, well-motivated features keeps the router simple enough that, if its decisions do degrade under index evolution, it will be possible to trace the degradation back to a specific feature's behaviour — directly supporting the failure analysis in Section 13.

**Output.** A discrete search-effort tier (for example, low/medium/high efSearch) rather than a continuous efSearch value. This choice is more robust to noisy oracle labels, easier to calibrate, and closer to how such a system would plausibly be deployed in practice.

**Model.** A small decision tree or logistic regression over the feature set, chosen deliberately over any neural model. The router's purpose in this project is to be simple and interpretable enough to diagnose *why* it fails, not to maximise predictive accuracy as an end in itself; a small interpretable model is more informative for that purpose than a more powerful but opaque one. A single-feature threshold router is included as a further, deliberately simplified baseline, since comparing how a one-feature and a three-feature router degrade differently is itself informative for H3.

**Static versus recalibrated.** The **static router** is fitted exactly once, on the S0 training split, and is never modified afterward — this is the primary object of the study. The **recalibrated router** is refit at each evolved state on a small, freshly drawn sample of queries — this exists only to bound how much of any lost accuracy is recoverable through recalibration, at what data cost, and is treated as a secondary reference comparison rather than a headline result.

---

## 10. Experimental Matrix

**Core experiments** (SIFT1M):

| Dimension | Levels |
|---|---|
| Update type | Insertion |
| Update magnitude | 4 levels (log-spaced) |
| Distribution condition | In-distribution, out-of-distribution |
| Resulting states | S0 + 8 evolved states (4 magnitudes × 2 conditions) |
| Policy | Fixed-ef (×3), static router, recalibrated router, oracle |
| Query set | Fixed test set (~1,000–2,000 queries), 3 difficulty strata |
| Metrics | Recall@k, distance computations, latency, routing regret |

**Secondary experiments** (modern embedding dataset): a reduced version of the same grid (2–3 magnitude levels, in-distribution insertion, plus one deletion condition), testing whether core findings generalise beyond SIFT1M's dimensionality.

**Optional extensions:** mixed insertion/deletion churn; graph edge-churn as a staleness measure; targeted local structural change (studied where it emerges naturally, not engineered); cross-checking a subset of results against a second ANN library; a live recalibration-trigger policy based on the drift-detection methodology from prior work.

This matrix is deliberately bounded rather than exhaustive: it is designed to cover the two dimensions — magnitude and distributional character — most directly implicated in the primary research question, without expanding into a combinatorial sweep that would not add proportional research value.

---

## 11. Datasets

**SIFT1M.** A standard 128-dimensional ANN benchmark (1M base vectors, 10K queries). Chosen as the primary dataset because its properties are well understood, it is fast to iterate on, and it allows the full core experimental matrix to be run at reasonable computational cost while the methodology itself is being developed and checked.

**Modern high-dimensional embedding dataset.** A subset (roughly 500K–1M vectors) of a contemporary embedding corpus at 768+ dimensions. Chosen as secondary validation to answer a specific generalisation question: do the findings from SIFT1M hold in the higher-dimensional, more modern embedding regime that current retrieval systems actually use, or are they an artifact of SIFT1M's comparatively low dimensionality?

**Synthetic data.** A small, separately generated dataset (for example, Gaussian mixtures) used exclusively to validate the staleness metrics themselves, since with synthetic data the true distributional shift is known by construction. This provides an early, cheap sanity check that the structural staleness measures behave as expected before they are trusted on real data, and is not used for any of the project's headline results.

No further datasets are added beyond these three, since each of the three serves a distinct and necessary purpose, and additional datasets would add cost without covering a genuinely new axis of the research question.

---

## 12. Metrics

| Metric | Plain-language meaning | Research question it answers |
|---|---|---|
| **Recall@k** | Fraction of the true k nearest neighbours actually found | Basic search-quality outcome, per query per state |
| **Distance computations** | Number of vector-distance evaluations performed | Hardware-independent measure of search effort/cost |
| **Latency / QPS** | Wall-clock time / queries answered per second | Practical throughput; secondary to distance computations, since it is environment-dependent |
| **Routing regret** | Difference between the effort the router chose and the effort the oracle would have chosen at matched recall | The core robustness measure; directly tests H1 |
| **Feature drift** | How much a given routing feature's value changes for the same query across states | Which signals are stable versus fragile; tests H3 |
| **Staleness-vs-regret correlation** | Spearman correlation between each candidate staleness measure and routing regret | Whether structural measures predict degradation better than update count; tests H4 |
| **Fraction of failed queries** | Proportion of queries falling below the recall target under a given policy | Whether degradation is widespread or concentrated |
| **Subgroup analysis by difficulty tertile** | Regret and failure rate broken down by the pre-registered easy/medium/hard groups | Whether degradation concentrates in a minority; tests H2 |

---

## 13. Experimental Validity

This section documents the safeguards built into the design to avoid the kinds of evaluation mistakes that can invalidate a study's conclusions.

| Risk | Safeguard |
|---|---|
| Router train/test leakage | Router fitting and test evaluation use strictly disjoint query splits, established before any experiments are run |
| Oracle-label leakage | Oracle labels used to train the router are computed only from the S0 training split; the router never sees labels from later states during training |
| Unfair baseline comparisons | Fixed-ef values are identical across all index states, with no per-state retuning; the oracle is honestly recomputed at every state rather than reused |
| Changing query sets across conditions | Forbidden by design — the same fixed test query set is used at every index state, which is what makes state-to-state comparisons paired and valid |
| Timing and warm-cache effects | Timing is single-threaded, warm-up queries are discarded, each condition is repeated 3–5 times and the median reported; distance computations, not latency, are treated as the primary effort measure |
| Implementation confounds | Identical build configuration and compiler flags across all conditions, with the hnswlib commit hash and build flags logged per run |
| Random-seed issues | All sources of randomness (query sampling, update ordering, HNSW's internal level assignment) are seeded and logged; core conditions are repeated across 2–3 seeds |
| Update-order effects | A single, fixed update order (interleaved/random) is used as the primary condition; sensitivity to update order is treated as an optional secondary check, not ignored |
| Hyperparameter leakage | The fixed-ef tier values and the recall target used to define the oracle are fixed from training-split behaviour only, never adjusted based on test-set outcomes |
| Accidental router retraining | The frozen static router is guaranteed, by construction, to be the literal same fitted object reused across every evolved state, never silently refit |
| Incorrect ground truth after index evolution | Ground truth is recomputed for every distinct index state and is never reused from a previous state, since inserts and deletes can change the true nearest neighbours |

This is treated as a first-class part of the research design, not a checklist added after the fact, because the project's central claims all depend on state-to-state comparisons remaining valid.

---

## 14. Statistical Analysis

Analysis is conducted primarily **per query and paired across states**, rather than by comparing aggregate averages between conditions: because the identical query set is used at every state, each query's outcome at S0 can be directly paired with its outcome at any later state, enabling **paired Wilcoxon signed-rank tests** rather than unpaired comparisons of group means. **Bootstrap confidence intervals**, computed over queries, accompany all reported aggregate metrics rather than point estimates alone. **Effect sizes** (such as matched-pairs rank-biserial correlation) are reported alongside any significance test, not in place of one. **Spearman correlation** between each candidate staleness measure and routing regret provides the core statistical test of H4. **Subgroup analysis** across the pre-registered difficulty tertiles provides the test of H2.

Distributions and per-query behaviour are emphasised throughout, rather than aggregate averages alone, because an average can obscure exactly the kind of concentrated failure pattern (H2) the project is specifically designed to detect; a mean recall figure that looks stable could still conceal a small set of queries failing badly. Formal significance testing is reserved for claims that appear in the final report; purely descriptive or exploratory diagnostics (for example, a feature-drift visualisation used to build intuition, or the synthetic staleness-metric sanity check) do not require it.

---

## 15. Failure Analysis

Failure analysis is designed as a core part of the experimental process from the outset, not as an activity added after the main results are in. It investigates:

1. **Queries the router handled correctly before an index update but incorrectly afterward** — inspected individually, tracing each such query's feature values across states.
2. **Systematic conservative drift** — the router beginning to over-spend search effort (safe, but wasteful) as the index evolves.
3. **Systematic aggressive drift** — the router beginning to under-spend search effort (cost-efficient, but damaging to recall) as the index evolves. These two failure directions are reported separately, since they have different practical consequences.
4. **Whether degradation is concentrated among difficult queries** specifically, or occurs independently of query difficulty.
5. **Which individual features become unstable** for the queries that fail, and whether this pattern is consistent with H3.
6. **Cases where a small global update percentage produces a large local structural change** for a specific query — directly checking whether that query's own local structural measure predicts its regret better than the global update fraction does.
7. **Whether structural staleness measures predict failure better than update count** at the level of individual failing queries, not just in aggregate correlation.

---

## 16. Possible Results

No particular outcome is assumed in advance. The following are plausible patterns, along with what each would mean if observed:

| Pattern | Scientific meaning |
|---|---|
| **A. Routing remains robust** across the tested range of index change | A genuinely useful negative result: recalibration may be needed only infrequently in practice, within the tested conditions |
| **B. Gradual degradation** with increasing index change | The expected-textbook pattern; still requires the structural-versus-count comparison to be informative on its own |
| **C. Degradation concentrated in a minority** of queries | Motivates targeted rather than blanket recalibration strategies |
| **D. Feature-specific stability** — some signals degrade faster than others | The most directly actionable outcome for designing more robust routers in the future |
| **E. Structural measures outperform update count** as predictors of regret | The highest-novelty possible finding: the intuitive assumption that "more updates means more staleness" would be shown to be an oversimplification |
| **F. Small-sample recalibration recovers most of the lost accuracy** | A practically useful finding, motivating a lightweight recalibration policy rather than full retraining |

Both confirming and disconfirming results for each hypothesis are treated as valid, reportable findings; none is assumed as the project's conclusion in advance.

---

## 17. Potential Contribution

It is important to distinguish three different things this project can produce, which should not be conflated:

- **The research question itself** — whether and how routing degrades under index evolution, and whether structural staleness measures outperform update count — is the contribution regardless of which way the results fall.
- **Possible findings**, contingent on the actual experimental results, may include: evidence that update count is an insufficient staleness measure; characterisation of which routing features are more or less stable under index evolution; evidence for or against concentrated (minority) degradation; and an estimate of how much recalibration cost is needed to recover lost accuracy.
- **Future extensions**, explicitly not part of this project's scope, include: a live, staleness-aware router that adapts automatically; a formal policy for triggering recalibration; and a connection to filter-aware search effort (a separate project), noted only as a forward pointer.

No claim of novelty is made in advance of the results; this section describes where a genuine contribution could emerge if the experiments support it, not a predetermined conclusion.

---

## 18. Implementation Architecture

The system separates performance- and correctness-critical components from analysis and orchestration components.

```
cpp/
  index management     — builds and maintains the HNSW index; wraps insert/delete
  update generation     — produces controlled insertion/deletion sequences
  feature extraction    — computes the live routing features against the current index state
  search execution      — runs HNSW search at a given efSearch and records effort/timing
  ground truth           — brute-force exact nearest-neighbour computation, parallelised
  metrics                — recall, distance-computation counts, latency computation
  CLI                     — drives a single experiment run from a configuration file

python/
  router training        — fits the static and recalibrated routers from extracted features and oracle labels
  statistical analysis    — paired tests, correlations, bootstrap intervals, subgroup analysis
  plotting                — all figures
  orchestration            — sweeps the experimental matrix by invoking the C++ CLI across configurations
```

**Why this split.** Index construction, search, ground-truth computation, and feature extraction are all performance-sensitive and must also be exactly correct, since errors here (for example, in ground truth) would invalidate every downstream result; C++ is used for all of these. Router fitting, statistical analysis, and plotting are neither a performance bottleneck nor safety-critical in the same sense, and are far faster to develop and iterate on in Python using established libraries — so Python is used for these, and for orchestrating the overall sweep across the experimental matrix.

---

## 19. Technology Stack

- **C++20**, confirmed at Phase 0 against the development VM's toolchain (g++ 13.3.0, supports C++20 cleanly). This was an open decision in earlier drafts of this document and is now resolved.
- **hnswlib**, used directly rather than reimplemented, for the reasons given in Section 6.1.
- **CMake** as the build system.
- **OpenMP**, used to parallelise brute-force ground-truth computation and the experiment sweep loop.
- **GoogleTest**, for unit tests on index correctness, feature extraction, and ground truth on small, verifiable cases.
- **clang-format / clang-tidy**, intended to be on by default per the project's engineering conventions. *Open item as of Phase 0:* not yet installed on the development VM (requires sudo); Phase 0 proceeds without them, to be installed and enabled before Phase 1 code is considered final.
- **Python**, with **NumPy** and **Pandas** for data handling, **scikit-learn** for router fitting, **SciPy/statsmodels** for statistical testing, and **Matplotlib** for plotting.

**CUDA is not introduced.** Nothing in the current research design requires GPU acceleration; it would only be considered if a later phase demonstrated a genuine computational bottleneck that CPU parallelism could not address.

---

## 20. Reproducibility

Every experimental condition is defined by a **YAML configuration file**, rather than hardcoded values, so that any run can be exactly reproduced or modified. Each run is assigned a unique **experiment ID** (derived from a hash of its configuration and a timestamp), and produces a metadata record alongside its results containing the configuration used, the Git commit hash of the codebase, all random seeds, and the compiler/build flags in effect. Results are stored as structured, row-level output (one row per query, per index state, per policy), not as ad hoc printed summaries, so that all downstream analysis operates on a well-defined dataset. A smoke-test configuration allows the entire pipeline to be run end-to-end in under a minute on a small subsampled dataset, and is run after any change to the codebase. The repository README documents the project overview, build instructions, how to run the smoke test, and how to reproduce the full experimental matrix from configuration files alone. A separate technical report documents the research question, methodology, results, and limitations in full.

---

## 21. Phased Development Plan

| Phase | Objective | Implementation | Experiment | Expected output | Validation checkpoint | Condition to proceed |
|---|---|---|---|---|---|---|
| **0** | Environment and repository | CMake project, hnswlib submodule, GoogleTest wired in, directory skeleton, dataset download scripts | — | A building, empty project | Build succeeds; empty tests pass | Clean build achieved |
| **1** | Static ANN baseline | Index management, ground truth, search execution, result serialisation | Fixed-efSearch sweep on SIFT1M at S0 | Recall-vs-ef curve, per-query results | Recall approaches 1.0 at high efSearch against independently-verified brute-force ground truth; curve shape matches known SIFT1M behaviour | Ground truth and search are independently verified correct |
| **2** | Oracle computation | Per-query search for minimal efSearch reaching a target recall | Oracle-effort computation on the S0 query set | Oracle-effort labels per query | Oracle distribution is non-degenerate; oracle beats fixed-ef baselines at matched recall | Oracle behaves sensibly before being used as a training label |
| **3** | Feature extraction | The 3–4 live routing features | Feature computation on the S0 query set | Feature values per query | No NaNs or degenerate constants; at least one feature visibly correlates with the oracle label | Features are informative before router training begins |
| **4** | Initial router | Router training and evaluation code | Fit and evaluate the static router on S0 splits | Trained router, S0 test-set performance | Router clearly outperforms the best fixed-ef baseline on S0 | **Must succeed before proceeding** — a router with no S0 advantage cannot meaningfully be said to "degrade" |
| **5** | Controlled index evolution | Update generation, state production, ground-truth recomputation per state | Synthetic-data validation of staleness metrics first, then real-data state production | A series of evolved index states with correct, state-specific ground truth | Staleness metrics behave sensibly on the synthetic validation set, where the true answer is known by construction | Staleness metrics are trustworthy before being applied to real data |
| **6** | Core robustness experiments | — | Full core experimental matrix (Section 10) | Complete, serialised results across all core conditions | All conditions present, none missing | Full dataset available before any analysis begins |
| **7** | Structural staleness analysis | — | Correlation analysis between staleness measures and routing regret | Preliminary answer to H4 | A clear correlation result, positive or negative, for each candidate measure | — |
| **8** | Failure analysis and recalibration | Recalibration routine (if time allows) | Failure-mode breakdowns (Section 15); recalibration comparison | Documented, concrete failure-case examples; recalibration recovery estimate | At least several individually inspected, explained failure cases exist | — |
| **9** | Statistical analysis and secondary dataset | — | Full statistical treatment (Section 14); reduced matrix on the modern embedding dataset | Statistically supported findings for H1–H5; a generalisation check | Findings from the secondary dataset are compared against SIFT1M, whether they replicate or diverge | Generalisation question explicitly answered, not skipped |
| **10** | Reproducibility and final report | README, one-command reproduction scripts, technical report | — | Complete, reproducible repository and written report | A clean checkout can regenerate at least one headline figure from configuration alone | — |

---

## 22. First Implementation Milestone

The first concrete deliverable is **Phase 1 in full**: a reproducible static HNSW benchmark on SIFT1M at S0 only, consisting of an hnswlib-based index build, parallelised brute-force ground truth, a fixed-efSearch sweep, and per-query recall, distance-computation, and latency logging to a structured result file, together with a script producing a recall-versus-efSearch plot.

This milestone exists to validate the two components everything else in the project depends on — index correctness and ground-truth correctness — before any further complexity (features, routing, index evolution) is introduced on top of them. If either is wrong, every subsequent result would be wrong as well, which is why this is the appropriate place to concentrate the first round of careful validation.

**Explicitly not included in this milestone:** query-level features, the router, index evolution of any kind, staleness measures, or the recalibrated-router comparison. These begin only once Phase 1 is validated and Phase 2 is under way.

---

## Addendum A1 — Phase 5 Reframe: Feature–Effort Signal Stability Under Index Evolution

**Status.**
- **Decision:** approved by the user on 2026-10-02.
- **Parameters marked [APPROVAL]:** proposed, awaiting approval.
- **Implementation:** none begins until this text is approved.
- This addendum is appended. It does not edit or replace any earlier section. Where it
  differs from §4, §10 or §21 for Phases 5–6, it takes precedence for those phases only.
  Everything else in this document stays in force.

### A1.1 Why the reframe

Phase 4b ended in a validated negative result (`docs/phase4b_final_report.md`,
`docs/phase4b_post_result_audit.md`):
- the frozen router did not beat the fixed baseline B1 by the required 10% at
  S\* ∈ {0.90, 0.95, 0.99};
- its adaptive allocation did save 6–10% against a same-class fixed policy;
- perfect per-query information would cost about 47% less than B1 at S\* = 0.95.

The §21 Phase 4 condition ("a router with no S0 advantage cannot meaningfully be said to
'degrade'") therefore blocks the original Phase 5–6, which measured router degradation.

That condition protected a more basic prerequisite: **the query-level signal a router
relies on must exist at S0 and survive index change.** Phase 3 and Phase 4b showed the
signal exists at S0 (|Spearman ρ| ≈ 0.37–0.57 between the features and oracle effort,
replicated on new queries). Whether it survives index evolution is untested, and it is
logically prior to any router-robustness question. Phase 5 is therefore **reframed, not
skipped**.

### A1.2 Reframed research question

> **RQ5′. Does the relationship between cheap query features (knn_dist, centroid_dist,
> score_concentration; LID as the ablation feature) and required oracle search effort
> stay stable as the index evolves through inserts and deletes?**

The original primary question (§4) is not answered by this phase. RQ5′ is a prerequisite
for it.

### A1.3 Scope: what is reused, what is not touched

**Reused essentially unchanged**, run on evolved states S1…Sn as well as S0:
- the Phase 2 oracle (`ars_oracle`): per-query smallest grid ef from which tie-aware
  recall@10 == 1.0 holds at every larger grid ef; grid 10–4096, ratio 1.05; censored if
  never reached;
- the Phase 3 feature extractor (`ars_features`): ef = 10 / k = 10 probe; ef = 20 /
  k = 20 LID probe; centroid of the **live** indexed vectors.

Ground truth is recomputed fresh for every state (§13). The only new code is index-state
production (inserts and lazy deletes on top of S0) and the structural staleness measures
of §8.

**Not touched or reopened:**
- Phase 4b in any form: its 54 frozen routers, models and τ, B1/B2, the candidate
  ladder, the S\* gate, the confirmation sets, and all Phase 4b result files.
- No router is trained, evaluated or recalibrated in this phase.
- No cost-saving claim is made.

The per-query success definition (10/10) is reused only as the definition of
"required effort".

### A1.4 Index-evolution model: §8 unchanged, instantiated as follows [APPROVAL]

§8 and §10 apply unchanged:
- **primary family:** insertion at 4 log-spaced magnitudes × {in-distribution,
  out-of-distribution};
- **secondary family:** lazy deletion, narrower grid;
- **optional:** mixed churn (not part of the core grid).

Concrete parameters, proposed:

1. **Insertion pool.**
   - Source: the `sift_learn` vectors that are not exact copies of any of the 10,000 SIFT
     queries, deduplicated. That is the 89,788-vector population of the Phase 4b v2
     construction.
   - The 2,000 confirmation-v2 vectors are removed, so that set stays clean, leaving
     **87,788 vectors**.
   - Query vectors are never inserted. Doing so would give a query a distance-0
     neighbour.
2. **Magnitudes.**
   - **1%, 2%, 4%, 8%** of |S0| = 1M, i.e. 10k / 20k / 40k / 80k inserts (factor-2 log
     spacing).
   - The cap of 8% comes from the pool size. SIFT1M has no larger in-distribution pool,
     and §11 forbids a new dataset. This is a known limitation (A1.10).
3. **In-distribution (ID) inserts.**
   - A seeded permutation of the pool (Fisher–Yates on `mt19937_64`, seed **20261008**).
   - States are **nested prefixes** of this one fixed insertion order (§13: a single,
     fixed update order).
4. **Out-of-distribution (OOD) inserts: regionally concentrated real vectors.**
   *(Amended 2026-10-02 per user decision. The earlier coordinate-permutation proposal
   is withdrawn.)*
   - **Region:** the nearest-neighbour ball in the insertion pool around one anchor
     vector. The anchor is the pool vector at index `rng(20261010).integers(0, |pool|)`,
     a neutral, seeded choice; it is not selected for its effect. Exact float64
     Euclidean distance; ties broken by pool row.
   - **OOD set at magnitude m:** the m·10⁶ pool vectors nearest to the anchor, so sets
     are nested by distance.
   - **Insertion order:** each increment (ball_m \ ball_previous) is inserted in the
     order given by the same seeded permutation as the ID inserts (20261008). So
     insertion order is random within each increment, and ID and OOD differ in *which*
     vectors are inserted, not in how they are ordered.
   - **Measured concentration** (label-free, pool only; regional / uniform mean distance
     to the anchor, and overlap with the ID set of the same size):

     | Magnitude | Share of pool | Distance ratio | Overlap with ID set |
     |---|---|---|---|
     | 1% | 11% | 0.67 | 12% |
     | 2% | 23% | 0.71 | 23% |
     | 4% | 46% | 0.78 | 45% |
     | 8% | 91% | 0.97 | 91% |

   - **Consequence:** concentration necessarily weakens with magnitude, because the pool
     is finite. At 8% the OOD set is 91% identical to the ID set, so OOD and ID are
     nearly the same condition by construction.
     **DECIDED by the user on 2026-10-02, before any real-data state existed (frozen
     design decision; §8 evolution model unchanged):**
     - **Pool-size limitation.** The eligible pool holds 87,788 vectors. An OOD set of
       m·10⁶ vectors must therefore take 11% / 23% / 46% / 91% of the pool at
       1 / 2 / 4 / 8%, and the anchor ball necessarily grows toward the whole pool.
     - **Measured overlap with the matched ID set:** 12% (1%), 23% (2%), 45% (4%),
       91% (8%).
     - **1%, 2%, 4%:** the ID-vs-OOD comparison at these levels is the inferential test
       of H2′.
     - **8%:** generated and measured exactly as planned (it counts for H1′, H3′ and
       all index-change analyses like every other evolved state), but **descriptive
       only** for any ID-vs-OOD comparison. It is labelled **"≈ ID by construction"**.
     - **Mandatory wording** for the 8% OOD condition in every Phase 5′/6′ report:
       > "At 8% insertion magnitude, the available pool forces the regional-OOD
       > construction to overlap approximately 91% with the matched ID set, so this
       > level does not provide a clean ID-vs-OOD comparison."

       An ID/OOD difference or similarity at 8% is never presented as evidence about a
       genuine OOD condition.
     - Not permitted as a remedy: a new OOD construction, dataset, clustering method or
       insertion magnitude.
   - **Rejected alternative:** a single k-means cluster as the region. The largest of 4
     clusters holds 23,941 vectors (27% of the pool), which caps pure-cluster OOD at
     about 2%.
5. **Deletion (secondary).**
   - hnswlib `markDelete` (lazy) of **2% and 8%** of base vectors.
   - Nested; uniform seeded selection from the base set (seed **20261009**).
   - Deleted vectors are excluded from ground truth and from the centroid.
6. **Index seeds:** 42, 43, 44, each starting from its own frozen S0.
   - For insertion trajectories, S0 is **rebuilt in-process** (single-threaded) and
     insertion continues, so hnswlib's level RNG continues its seeded stream.
   - The rebuilt S0 must be byte-identical to the cached S0 file; otherwise stop.
   - This avoids the documented issue that a *loaded* index's level RNG is not
     config-seeded, without touching hnswlib internals.
   - Deletion trajectories load the cached S0, since `markDelete` uses no RNG.
7. **States per seed:** 4 ID + 4 OOD + 2 deletion = 10 evolved states (+ S0), so **30
   evolved states** in total.
   - Estimated cost: about 17 min of oracle search per state on 16 cores, roughly 10 h of
     wall clock overall.

### A1.5 Measurements per state

Same fixed **query set** at every state: all **10,000 original SIFT queries** (§7.2).
No model is fitted, so the train/test split is not needed for the primary analysis. It
is still recorded for secondary use.

For each (seed, state, query):
- **Features**, recomputed live against that state's index: probe searches on the evolved
  graph; centroid of the live vectors.
- **Oracle effort**, from fresh ground truth on the live vectors.
- **Structural measures** from §8:
  - update fraction (state level);
  - neighbourhood-overlap decay (|top-10 at S0 ∩ top-10 at s| / 10, per query, post-hoc);
  - local-density drift (relative change in knn_dist, per query);
  - distributional drift (state level; meaningful for OOD).

Graph edge churn stays deferred (§8 optional).

### A1.6 Hypotheses (replace H1–H5 for Phases 5–6 only)

Notation:
- ρ_s(f) = Spearman correlation between feature f and oracle effort at state s, over the
  fixed query set.
- Censored efforts are ranked above every finite effort (tied among themselves).
- **Δ_s(f) = |ρ_s(f)| − |ρ_0(f)|**, paired on the same queries and the same index seed.

**H1′ (primary): signal stability.** For every core feature
f ∈ {knn_dist, centroid_dist, score_concentration} and every evolved state s,
**Δ_s(f) ≥ −X with X = 0.05**. LID is evaluated identically as a secondary (ablation).

Justification for **X = 0.05** [APPROVAL]:
1. **Substantive.**
   - 0.05 is about 11–14% of the core features' S0 signal (|ρ| 0.37–0.46).
   - For knn_dist, 0.46 → 0.41 removes about 21% of the rank variance explained
     (ρ² 0.21 → 0.17).
   - The Phase 4b router gained only 6–10% in-class from this signal at S0, so a loss of
     this size is material to any router.
2. **Above natural variation.**
   - Between three independently built S0 indexes, ρ varies by at most 0.011.
   - Training vs new queries (Phase 3 vs confirmation v2) differs by at most about 0.05.
3. **Adequately powered** (checked before any evolved state is evaluated; the lesson of
   Phase 4b criterion 2).
   - Measured on existing S0 data (seed 43 vs 42, all 10,000 queries, query-clustered
     bootstrap), the paired standard error of Δ is 0.0029 (knn), 0.0029 (centroid),
     0.0048 (concentration), 0.0035 (LID).
   - The minimum detectable margin at 80% power is ≤ 0.014.
   - So a true Δ = 0 passes with near certainty, and X = 0.05 is not a power artefact.

**H2′ (character vs amount).** At matched magnitude, OOD insertion changes Δ_s(f)
differently from ID insertion (two-sided). Statistic: Δ_OOD,m(f) − Δ_ID,m(f), per
magnitude m.

**H3′ (differential stability, from original H3).** Features differ in stability.
Expectation: global centroid_dist is more stable than the local knn_dist and
score_concentration. Statistic: pairwise differences of Δ at the largest magnitude.

**H4′ (structural vs count, from original H4; tested in Phase 7′).** Per query, the change
in required effort, |log₂(effort_s / effort_0)|, is predicted better (Spearman over all
query×state pairs) by neighbourhood-overlap decay or local-density drift than by update
fraction.

Original H5 (recalibration) is **not applicable** without a router. It is dropped for
these phases.

### A1.7 Statistical analysis and decision rules

- **Bootstrap:** query-clustered (resample query ids; every seed and state of a query
  moves together), 2,000 resamples, seed 20261002, 95% percentile CIs.
- **Per (feature, state), three-valued, pre-specified:**
  - **STABLE** if the CI lower bound of Δ ≥ −X;
  - **DEGRADED** if the CI upper bound of Δ < −X;
  - **INCONCLUSIVE** otherwise.
- **Pooling:** the primary pooled Δ averages over the 3 seeds. Per-seed results are
  always reported, and a seed-specific DEGRADED result is never averaged away (reported
  explicitly).
- **H1′ per feature:**
  - **supported** if STABLE at every evolved state;
  - **refuted** if DEGRADED at any state;
  - **inconclusive** otherwise.

  H1′ overall is supported only if all three core features are supported.
- **Trend:** Spearman correlation between log magnitude and Δ within each trajectory
  (descriptive, 4 points).
- **Also reported (§12):**
  - feature drift (per-query |f_s − f_0|);
  - shift in the oracle-effort distribution;
  - censored counts per state.
- **H2′, H3′:** paired CIs of the stated differences, with no further thresholding
  (secondary).
- **Multiplicity:** H1′ uses an intersection over states, which is conservative. No α
  adjustment; all cells are reported.

### A1.8 Validity safeguards (additions to §13)

- **S0 reproduction:** before any evolved state, the new pipeline run at S0 must
  reproduce the canonical Phase 2 labels and Phase 3 features **byte-for-byte**, per
  seed.
- **Rebuilt S0** must be byte-identical to the cached S0 index (A1.4.6).
- **Ground truth per state** is verified independently: NumPy brute force on a seeded
  sample of 200 queries per state, exact match up to ties.
- **Insert pool integrity:** 0 exact duplicates with the base set, with the queries, with
  the confirmation-v2 set, and within the pool.
- **Distance check:** pool-vector-to-query distances are compared, label-free, with
  base-to-query distances, so inserts are not unusually close to queries.
- **No outcome-driven changes:** nothing in A1.4–A1.7 may be changed after any
  evolved-state result is seen.

### A1.9 Phase plan for the reframed phases (replaces the §21 rows for Phases 5–6 only)

| Phase | Objective | Validation checkpoint | Condition to proceed |
|---|---|---|---|
| **5′** | State production and measurement pipeline | (a) **Synthetic validation** (§11, Gaussian mixture with known shift): update fraction exact; neighbourhood-overlap decay non-decreasing in ID magnitude; distributional drift OOD > ID at every magnitude; feature/oracle pipeline runs end-to-end. (b) S0 reproduction byte-identical, all seeds. (c) Rebuilt S0 byte-identical. (d) Pool integrity. (e) Ground-truth check per state. | All of (a)–(e) met; otherwise stop and report. |
| **6′** | Full matrix: 30 evolved states × 10,000 queries; H1′–H3′ | All cells present, none missing; per-state ground truth verified | Complete dataset before analysis |

Phases 7–10 are re-scoped by a later addendum before they start. Phase 7′ is planned to
carry H4′. Phase 9's secondary dataset and Phase 10 are unchanged in intent.

### A1.10 Known limitations, declared in advance (strengthened 2026-10-02)

**These statements must be carried, in substance and wording, into every Phase 5′/6′
result and report. They are not only addendum text.**

1. **"OOD" is a narrow construction.** Here it means *regionally concentrated real
   SIFT vectors*: a nearest-neighbour ball, around one anchor, in the same `sift_learn`
   pool that supplies the ID inserts. It is **not** an independently sourced or
   semantically different distribution. Its concentration weakens with magnitude: the
   distance ratio is 0.67 at 1% and 0.97 at 8%, where OOD ≈ ID. Findings about "OOD"
   apply to this construction only.
2. **Magnitude cap.** Inserts are capped at **8% of |S0|** by the size of the
   in-distribution pool, and deletion at the stated levels. Any stability finding must
   be stated as **"stable within 1–8% insertion churn (and 2–8% lazy deletion)"**, not
   as a general claim about index evolution.
3. **Deletion is lazy only** (`markDelete`); there is no physical removal or graph
   repair.
4. **One dataset (SIFT1M) at this stage.**
5. **Signal stability is necessary, not sufficient.** Stable feature–effort correlation
   does not imply that a router would remain robust or would save cost. A result here
   says nothing directly about router cost savings.

### A1.11 Items requiring approval

1. **Insertion pool and magnitudes:** 87,788 eligible `sift_learn` vectors; 1 / 2 / 4 /
   8%.
2. **OOD construction:** regionally concentrated anchor ball. Approved 2026-10-02.
   8% handling **decided 2026-10-02** (A1.4 item 4): H2′ tested at 1–4% only; 8% OOD is
   descriptive only for the ID-vs-OOD comparison, "≈ ID by construction".
3. **Deletion levels:** 2% and 8%.
4. **Seeds:** insertion order 20261008, deletion 20261009, OOD permutation 20261010.
5. **Index RNG handling:** rebuild S0 in-process with a byte-identity check.
6. **Query set:** all 10,000 SIFT queries.
7. **X = 0.05** and the three-valued decision rule.
8. **H2′–H4′ as secondary.**
9. **Phase 5′/6′ checkpoints.**

## Addendum C1 — Experiment C: S0 Runtime Adaptive Actuator (DARTH re-implementation)

**Status.** Decided by the user on 2026-10-04. Appended; earlier sections, Phase 4b and
A1/Phase 5′ are unchanged. Written before any DARTH code or result existed.

### C1.1 Purpose and position
Experiment C establishes one credible **runtime, in-search adaptive actuator** at S0, whose
calibration can be frozen and later applied unchanged to evolved indexes (Experiment E;
not started). C is a prerequisite, not a contribution: no novelty is claimed.

### C1.2 Actuator selection
- **Ada-ef (SIGMOD'26) rejected.** Its ef-estimator rests on full-distance-list
  distributions derived only for inner product and cosine. The paper (§6.2) leaves L2 open,
  and the code accepts only "cd" / "ipd" (notes 2026-10-04). Not modified, not used.
- **DARTH selected** (Chatzakis, Papakonstantinou, Palpanas, PACMMOD 3(4) art. 242,
  SIGMOD'26; arXiv 2505.19001; reference code github.com/MChatzakis/DARTH @ 0d9bafcf).
  It is a faithful **re-implementation of the published mechanism** on hnswlib v0.8.0's
  public stop-condition API (`searchStopConditionClosest` / `BaseSearchStopCondition`). It
  is not the authors' FAISS code, and hnswlib is not modified.

### C1.3 Mechanism reproduced (published values; nothing tuned)
- Features (order fixed): nstep, ndis, ninserts, firstNN, closestNN, furthestNN (k-th),
  avg, var (population), median, p25, p75. The statistics are over the current k-best
  distances; percentile = element at floor(p·(k−1)) (reference implementation). ndis counts
  base-layer distance computations, excluding the entry point.
- Predictor: LightGBM regressor, defaults, 100 estimators, learning rate 0.1,
  random_state 42, trained on all observation rows. Label = recall@10 by id at the
  observation (the reference definition).
- Training traces: an observation after **every** base-layer distance computation (li = 1:
  paper text and the reference SIFT run configuration), logged once ≥ k insertions exist.
  Plain search at the search cap.
- Runtime: the predictor is invoked when the distances since the last call reach pi.
  Predicted recall is clipped to [0, 1]. If it is ≥ Rt, the search stops with the current
  result. Otherwise pi = mpi + (ipi − mpi)·(Rt − Rp), truncated to int. With fewer than k
  insertions the call returns 0 and pi is unchanged (reference behaviour).
- Intervals (published heuristic): dists_Rt = mean over training queries of the first
  observation's ndis with recall ≥ Rt (queries never reaching Rt excluded, counted).
  ipi = int(dists_Rt / 2), mpi = int(dists_Rt / 10).
- Search cap (the paper's rule: plain search reaching mean recall ≥ 0.99), applied on the
  **training split** (no test leakage): the smallest Phase-2 grid ef with mean recall@10
  (by id) ≥ 0.99 over the 8,000 training queries at S0 seed 42 = **ef 142**.

### C1.4 Data, index, queries
- SIFT1M, L2, k = 10, frozen S0 index (seed 42, data/cache/hnsw_S0_c86ffb2a63e7325b.bin).
- **Training** = the 8,000 Phase-2 training-split queries
  (splits/sift1m_query_split_seed20261001.csv, sha256 e0e3c6fa…). Not used: the A1
  insertion pool, confirmation v2, the Phase-2 test split.
- **Evaluation** = the 2,000 Phase-2 test-split queries, S0 seed 42, once.

### C1.5 Contract, baseline, gate (fixed before the run)
- Target: **mean recall@10 = 0.95**. Quality is measured as tie-aware recall@10 (project
  convention); id recall is also reported.
- Baseline: the frozen Phase-4b **mean-recall B1 at R = 0.95**, seed 42 (ef 50/53,
  w_hi 0.538, splitmix64 hash seed 20261006, query_set_id sift_query), per-query values
  read from results/phase4b_eval_old_test_second_look_20261002T175740Z. This is the
  secondary mean-recall contract, distinct from the Phase-4b primary per-query S* contract.
  Phase 4b is not modified.
- Cost: total distance computations per query (CountingL2Space, upper layers included),
  as in Phase 4b. Also recorded: predictor invocations, predictor time, single-thread
  wall-clock latency (DARTH vs plain hnswlib at the B1-assigned ef).
- **C passes** only if all of:
  1. faithful mechanism (C1.3) and the correctness mode passes: with prediction
     disabled, the stop-condition search reproduces hnswlib searchKnn at ef 142 exactly
     (labels and distance counts) on all 10,000 queries;
  2. plausibility: recall values in [0, 1], distance counts ≥ 1 and ≤ the plain ef-142
     count, predictor calls ≥ 1 per query, termination before the cap for most queries;
  3. **meaningful improvement over B1** (Phase-4b gate conventions, single seed): saving
     vs B1 in mean distance computations ≥ 10% with 95% CI lower bound > 0
     (query-clustered bootstrap, 2,000 resamples, seed 20261002) and Wilcoxon p < 0.01;
     **and** quality non-inferiority: the 95% CI lower bound of the mean tie-aware recall
     difference (DARTH − B1) ≥ −0.01.
- No published-number gate: DARTH has no SIFT1M / L2 / M16 / k10 result (its SIFT run is
  SIFT100M, M32, efC500, efS500, reported as speedups).
- If the result is poor it is reported as is. No tuning, no hyperparameter search, no
  alternative models.

### C1.6 Freeze (if C passes)
DARTH code revision; LightGBM model file and sha256; feature definitions and order; model
parameters; Rt; ipi, mpi; search cap; training query ids and hash; S0 index sha256;
evaluation query ids and hash. This is the immutable S0-calibrated actuator for Experiment E.
E is not started without explicit instruction.

### C1.7 Scope and time-box
One focused implementation session. Dependency added: LightGBM 4.6.0 (C API for search,
Python package for training). Stop and report on any ambiguity, correctness or
compatibility problem rather than redesigning.

### C1.8 Feasibility estimate (from the existing Phase-2 oracle curves, before any trace)
At ef 142 the 8,000 training queries perform 18.74 M distance computations in total
(mean 2,343, p99 3,112, max 3,409). At li = 1 that gives ≈ 18.6 M observation rows × (11
features + label + qid); about 1 GB as float32 binary, a few seconds of search and ≈ 1–3 min
of LightGBM training. Lightweight; it proceeds.

### C1.9 Frozen DARTH feature interface and policy (2026-10-04, before any trace, model or evaluation)
Supersedes the feature *order* listed in C1.3, which follows the paper's Table 1; the
semantics there are unchanged. Immutable from here on.

**Independent re-implementation.** The code in include/ars/darth.h, src/darth.cpp,
apps/exp_c_darth.cpp and python/exp_c_*.py is written for this project from the paper's
description and our reading of the reference repository's behaviour. No DARTH or modified
FAISS source is copied (attribution: THIRD_PARTY.md).

**Model input order (= the DARTH training/model specification; all published all-features
models use it):**

| # | name | semantics | collected |
|---|---|---|---|
| 0 | step | base-layer candidates expanded before the current one (0-based) | should_stop_search pops |
| 1 | dists | base-layer distance computations so far, excluding the base-layer entry point | should_consider_candidate calls |
| 2 | inserts | insertions into the running k-best set (strictly closer than its current k-th, or the set not yet full) | add_point_to_result |
| 3 | first_nn_dist | distance of the first point entered into the result (base-layer entry point) | first add_point_to_result |
| 4 | nn_dist | smallest distance in the current k-best | k-best set |
| 5 | furthest_dist | largest (k-th) distance in the current k-best | k-best set |
| 6 | avg_dist | mean of the k-best distances | k-best set |
| 7 | variance | population variance of the k-best distances | k-best set |
| 8 | percentile_25 | element at index floor(0.25·(k−1)) of the sorted k-best distances | k-best set |
| 9 | percentile_50 | element at index floor(0.50·(k−1)) | k-best set |
| 10 | percentile_75 | element at index floor(0.75·(k−1)) | k-best set |

Distances are hnswlib L2Space values (squared L2), k = 10. The k-best set holds every
non-deleted point hnswlib adds to its result, so it is exactly the best k of all computed
non-deleted base-layer distances (ef ≥ k).

**Deviation from the published runtime code (deliberate, documented).** The reference
runtime (DARTHPredictorHNSW::predict_recall) passes the features positionally in the order
step, dists, inserts, first_nn, nn, avg, furthest, variance, median, p25, p75. That
differs from the order the models are trained with (avg ↔ furthest swapped; the p25 slot
receives the median and the p50 slot receives p25). We feed the model in its training
order, so the *intended* trained DARTH policy is evaluated, not an apparent
implementation-level feature-order defect. This is not a tuning step; no other choice
depends on it.

**Observation and training (frozen).**
- Traces: plain search at the search cap; one observation after every base-layer distance
  computation (li = 1), recorded only once inserts ≥ k. Label = recall@10 by id of the
  current k-best vs the S0 ground truth (data/cache/gt_S0_01564caeeb173bc1).
- Model: LightGBM 4.6.0, LGBMRegressor(objective="regression", n_estimators=100,
  random_state=42, verbose=-1), all other parameters default (learning_rate 0.1,
  num_leaves 31, …), fitted on all observation rows of the 8,000 training queries. Saved
  as LightGBM text model; sha256 recorded.

**Runtime policy (frozen).**
- Search cap / beam: hnswlib ef = 142 (C1.3). k = 10. Target Rt = 0.95.
- Checks: a counter of base-layer distances since the last check starts at 0. After each
  distance (and after that point's possible insertion), if counter == pi a check runs:
  - if inserts < k, the prediction is 0, pi is unchanged and the counter is reset;
  - otherwise Rp = clip(model(x), 0, 1); stop if Rp ≥ Rt;
  - else pi = int(mpi + (ipi − mpi)·(Rt − Rp)), counter reset.
- Initial pi = ipi, with ipi = int(dists_Rt / 2) and mpi = int(dists_Rt / 10).
  dists_Rt = mean over training queries of `dists` at the first observation with
  recall ≥ Rt; non-reaching queries are excluded and counted.
- On stop the result is the current result set. hnswlib finishes the current node's
  remaining neighbour distances (counted in the cost; no further insertions) and returns.
- Without a stop, the search ends by hnswlib's own rule at ef 142 (the plain search).
- Seeds: LightGBM random_state 42; bootstrap seed 20261002; the B1 hash seed is the frozen
  Phase-4b 20261006 (read-only). No other randomness.
- Other differences from the reference, each a consequence of using hnswlib, not a choice:
  hnswlib's beam termination replaces FAISS's efSearch-step rule; termination takes
  effect after the current node's neighbour list.


## Addendum E1 — Experiment E: Design Freeze v2 (canonical implementation specification; frozen 2026-10-05)

Verbatim copy of docs/exp_e_design_freeze_v2.md; only the heading levels are nested.

**Status: FROZEN — FINAL (user-approved 2026-10-05; appended to docs/design_doc.md as Addendum E1).**
- Becomes binding only when the user approves it. On approval it is appended verbatim to
  docs/design_doc.md as Addendum E1.
- Written before any Experiment E estimand existed. No B1 or DARTH outcome at any evolved state
  has been computed or inspected.
- Known beforehand, and disclosed: the Phase 5′ results (docs/phase5b_report.md).
- Margins and policies are as frozen by the user. Every other choice comes from the A1 /
  Phase-4b / C1 conventions or from S0 evidence, and the source is cited.
- **One genuine logical contradiction was found** (§6.3). It is resolved in §6.3
  (user-approved 2026-10-05).

### 1. Research question
When the ANN index evolves through insertions and lazy deletions, does a policy calibrated once
at S0 become *stale*, i.e. cost and contract quality drifting relative to a policy freshly
calibrated at the evolved state? And do the A1 structural index-change measures track that
staleness?

Two S0-calibrated policies are studied:
- the static fixed-effort B1;
- the runtime-adaptive DARTH (C1 failed; studied as a calibrated actuator, not as a winner).

No direction is presupposed.

### 2. Experimental units and states
- **Index seeds:** 42, 43, 44.
- **States per seed:** the existing A1/Phase-5′ states, unchanged (manifest
  data/phase5/sift/states/run_manifest.json):
  - **insertion family (8 states):** id_{10000, 20000, 40000, 80000} and
    ood_{10000, 20000, 40000, 80000};
  - **deletion family (2 states):** del_{20000, 80000}.
  - Total 30 (seed, state) cells, plus S0 per seed.
- **Magnitude:** m = update fraction = count / 1,000,000. Insertion m ∈ {0.01, 0.02, 0.04,
  0.08}; deletion m ∈ {0.02, 0.08}.
- **Evaluation unit:** query q ∈ the 2,000 Phase-2 test-split queries
  (splits/sift1m_query_split_seed20261001.csv, split == "test").

### 3. Frozen policies (per seed σ; no retuning, no retraining)
- **B1(σ):** the Phase-4b mean-recall R = 0.95 two-ef mix for seed σ (results/phase4b_train_311634ee3011_20261002T145649Z):

  | Seed | ef_lo / ef_hi | w_hi |
  |---|---|---|
  | 42 | 50 / 53 | 0.538 |
  | 43 | 50 / 53 | 0.596 |
  | 44 | 50 / 53 | 0.637 |

  - Per-query ef = ef_hi if splitmix64(fnv1a64("sift_query:<query_id>") XOR 20261006)/2⁶⁴ < w_hi,
    else ef_lo (router4b_lib.assign_mix).
  - **At every state each query keeps its S0-assigned ef.**
- **DARTH(σ):** the frozen policy files (no other DARTH policy may be used):

  | Seed | Policy file |
  |---|---|
  | 42 | derived/exp_c/darth_policy_s0.json |
  | 43 | derived/exp_e/darth_policy_s0_seed43.json |
  | 44 | derived/exp_e/darth_policy_s0_seed44.json |

  - Each policy is run unchanged on seed σ's evolved index with apps/exp_c_darth `eval`.
- **Cost:** total distance computations per query (CountingL2Space, upper layers included), the
  Phase-2/4b/C1 accounting.
  - B1 and reference costs and recalls are read from each state's existing oracle_curves.csv at
    the assigned ef (identical accounting, verified at S0 in C1).
  - DARTH cost is measured live.
- **Quality:** tie-aware recall@10 against the state's own ground truth. Contract success ⇔
  tie-aware recall@10 = 1 (10/10).

### 4. Evaluation split
- All hypotheses are evaluated on the 2,000 test queries only.
- The 8,000 training queries are used only to calibrate the state-local reference (§5).
- No estimand, test or descriptive result is computed on training queries.

### 5. State-local reference REF(σ, s)
- **Per (σ, s)**, independently, seeds never pooled:
  `router4b_lib.two_ef_mix(level_by_ef, grid, 0.95)`.
  - `grid` = the canonical 121-ef grid.
  - `level_by_ef` = the mean over the 8,000 training queries of tie-aware recall@10 at each grid
    ef, from (σ, s)'s oracle_curves.csv.
- **Assignment:** the same hash as B1 (query_set_id "sift_query", seed 20261006).
- **Test-query cost and recall** come from the test rows of the same oracle_curves.csv.
- **Integrity requirement:** REF(σ, S0) must equal B1(σ) exactly (ef_lo, ef_hi, w_hi to 1e-12
  and per-query ef). Otherwise STOP.
- **For DARTH, REF is a freshly calibrated simple B1-style policy, not a state-local DARTH.**
- If two_ef_mix returns None (no grid ef reaches 0.95) → failed cell (§16).

### 6. H-E1 — cost staleness

#### 6.1 Cost estimand: ratio of means (option A), preferred cost-ratio construction (the primary H-E1 estimand is D, §6.2)
For policy P ∈ {B1, DARTH}, seed σ and state s:

  R(P, σ, s) = ( C̄_P(σ, s) − C̄_REF(σ, s) ) / C̄_REF(σ, s)

with C̄ = the mean cost over the evaluation queries (in a bootstrap replicate, over the
resampled queries).

**Why:**
- it is the Phase-4b/C1 "saving vs B1" definition (sign flipped);
- it is stable when per-query costs are small (B1 minimum 314);
- it is a smooth function of paired means, so the paired percentile bootstrap applies.

**Secondary (descriptive, no decision):** the mean over queries of (C_P,q − C_REF,q)/C_REF,q,
with its CI.

#### 6.2 Decision quantity (primary H-E1 estimand): proportional S0-anchored drift D

  D(P, σ, s) = [ C̄_P(σ, s) / C̄_REF(σ, s) ] / [ C̄_P(σ, S0) / C̄_REF(σ, S0) ] − 1

- **C̄** = the mean distance computations per query over the 2,000 test queries. In each
  bootstrap replicate (§10), every C̄ (state and S0 terms alike) is recomputed over the same
  resampled query ids.
- **S0 terms** come from seed σ's S0 rows: frozen B1/REF from the S0 oracle curves, DARTH from
  that seed's S0 evaluation.
- **D = 0 at S0 for both policies**, by construction.
- **For B1,** C̄_B1(σ, S0) = C̄_REF(σ, S0) exactly (§5 integrity requirement). Its S0 reference
  ratio is therefore 1, and D(B1, σ, s) equals the ordinary relative regret R(B1, σ, s) of §6.1.
- **For DARTH,** D isolates the proportional change from DARTH's own S0 cost relationship to
  the reference. It does not re-test DARTH's known C1 S0 inefficiency.
- **Descriptive only (no decision):** raw R(P, σ, s) (§6.1), the additive
  ΔR(P, σ, s) = R(P, σ, s) − R(P, σ, S0), and the per-query relative cost of §6.1.

#### 6.3 Logical contradiction found, and its resolution (DECIDED: user-approved 2026-10-05)
**The contradiction:**
- The raw R measures staleness only when the frozen policy equals the reference at S0. That
  holds for B1.
- It does not hold for DARTH: at S0, DARTH already costs +11.0% / +11.5% / +10.8% vs the S0
  reference (= B1), per C1 and Stage 1.
- With a ±10 pp margin, DARTH's raw-R verdict would be decided by its known S0 inefficiency (C1),
  not by any change caused by index evolution. That contradicts the research question
  ("calibrated at S0 → stale later").

**Design decision (approved by the user on 2026-10-05):**
- **Raw R is retained as descriptive only.** DARTH has a pre-existing S0 cost disadvantage vs
  the reference, so raw R would mix that known C1 result with evolution-induced change.
- **The additive ΔR = R(s) − R(S0) is rejected.** It scales DARTH's evolution drift by its S0
  relative-cost ratio (≈ 1.11): a proportional drift p appears as ≈ 1.11·p. So the ±10% margin
  would not carry the same proportional meaning for DARTH as for B1.
- **The proportional D (§6.2) is adopted as the primary H-E1 estimand.** It is 0 at S0 for both
  policies, equals R for B1, and gives the same proportional interpretation of the ±10% margin
  for both policies.
- **Basis of the decision:** only the conceptual estimand and already-established S0 evidence
  (C1 and Stage-1 S0 cost ratios 1.110 / 1.115 / 1.108; B1 ≡ REF at S0). No Experiment E
  evolved-state result existed or was consulted. Margins, policies, references, states, queries
  and bootstrap are unchanged.

#### 6.4 Decision rule (per pooled state, three-valued; A1.7 style)
- **Pooled estimate:** D̄(P, s) = (1/3) Σ_σ D(P, σ, s), equal weights.
- **95% CI:** percentile, query bootstrap (§10).
- **Classification of D̄(P, s), with margin M₁ = 0.10:**
  - **STABLE:** the 95% CI lies entirely within [−0.10, +0.10];
  - **STALE-COSTLIER:** CI lower bound > +0.10;
  - **STALE-CHEAPER:** CI upper bound < −0.10;
  - **INCONCLUSIVE:** otherwise.
- **Per seed:** D(P, σ, s) is classified the same way and always reported. Any seed-specific
  STALE-COSTLIER / STALE-CHEAPER is listed explicitly next to the pooled verdict and is never
  averaged away.
- **Quality context (required):** mean tie-aware recall@10 is reported for every (P, σ, s),
  including REF and S0, and for the seed means. Cost drift is then interpreted alongside quality
  drift: D can partly reflect a change in recall rather than in efficiency.
- **Family verdict**, per policy × family (insertion, deletion separately):
  - STABLE-FAMILY if every pooled state in the family is STABLE;
  - STALE-FAMILY if any is STALE;
  - INCONCLUSIVE-FAMILY otherwise.
- **Trend across magnitude:** secondary, not part of the verdict (§11).
- **S0 power context** (computed before E, S0 only): the paired per-query SD of DARTH − B1 cost
  is 494 (SE of mean ≈ 1.05% of cost per seed). Seed-to-seed B1 cost differences have SD ≈ 55
  (≈ 0.12% relative SE). Expected CI half-widths are ≈ 1–2 pp, well inside the ±10 pp margin.

### 7. H-E2 — contract staleness (newly failing)

#### 7.1 Cohorts and transitions
- **Pass:** pass_P(σ, s, q) = 1 iff tie-aware recall@10 = 1 (implemented as ≥ 1 − 1e-9).
- **S0 cohorts:**
  - K_P(σ) = { q : pass_P(σ, S0, q) = 1 } for P = B1 and DARTH;
  - **K_REF(σ) = K_B1(σ).** The reference at S0 is B1(σ) exactly (§5 integrity requirement), so
    the reference's S0 cohort is B1's S0 cohort. This applies whichever frozen policy it is
    compared with.
- **Transition categories per (P, σ, s, q)** (S0 → s), for P ∈ {B1, DARTH, REF}:
  stable-pass, newly-failing, stable-fail, improved.
  - All four category counts are reported for every cell.

#### 7.2 Estimand: conditional rate, denominator option B
  NF(P, σ, s) = #{ q ∈ K_P(σ) : pass_P(σ, s, q) = 0 } / |K_P(σ)|

The same formula applies to REF with K_REF(σ). Then:

  E(P, σ, s) = NF(P, σ, s) − NF(REF, σ, s)   (absolute; fraction units, 0.05 = 5 pp)

**Why denominator B:**
- The frozen policies have different S0 pass rates (B1 ≈ 0.69, DARTH ≈ 0.73).
- With denominator A (all 2,000 queries), the excess would mechanically mix in the base pass
  rate.
- B is the probability that an S0 success is lost, which is the staleness quantity.
- At S0, E = 0 by construction, so H-E2 needs no anchoring.

**Disclosed caveat (DARTH):** DARTH's cohort and the reference cohort differ in composition
(C1: 192 queries pass under DARTH but fail under B1). E(DARTH) therefore includes a cohort-mix
component. It is reported, not corrected.

#### 7.3 Decision rule
Identical to §6.4, applied to Ē(P, s) = mean over σ of E(P, σ, s), with:
- margin M₂ = 0.05: STABLE if CI ⊂ [−0.05, +0.05]; STALE if CI lower bound > +0.05
  (STALE-WORSE) or CI upper bound < −0.05 (STALE-BETTER); INCONCLUSIVE otherwise;
- 95% percentile CI;
- per-seed classification reported;
- family verdicts per policy × family;
- trend secondary (§11).

**S0 power context:**
- the B1 seed-to-seed newly-failing rate is 1.4–2.0% of ≈ 1,380 S0 passers;
- the binomial SE per seed is ≈ 0.35 pp;
- expected CI half-widths are ≲ 1 pp, inside ±5 pp.

### 8. H-E3 — structural predictors of staleness
- **Predictors:** exactly the A1.5 (§8) measures, taken from the existing Phase-5′ artifact
  results/phase5b_analysis_20261004T230023Z. Nothing is recomputed, and no new measure is added.
  1. update fraction m (state level);
  2. neighbourhood-overlap decay = 1 − overlap_top10_with_S0/10 (per query; column
     overlap_top10_with_S0 of rows_query_state_seed.csv.gz);
  3. local-density drift = (knn_dist_s − knn_dist_S0)/knn_dist_S0 (per query; column
     local_density_drift). This is the A1 "local-density drift". Centroid drift is not an A1
     structural measure and is not used.
  4. distributional drift D (state level; analysis.json → distributional_drift_D).
- **Representation at the (σ, s) level:** measures 2 and 3 are averaged over the 2,000 test
  queries of (σ, s) (in a bootstrap replicate, over the resampled queries). Measures 1 and 4 are
  constants per state.
- **Outcomes:** D(P, σ, s) (§6.2, the primary H-E1 quantity) and E(P, σ, s) (§7.2), for
  P ∈ {B1, DARTH}. Raw R and additive ΔR are descriptive only and are not H-E3 outcomes.
- **Unit and statistic (primary, insertion family only):** Spearman ρ between predictor and
  outcome over the 24 (σ, s) insertion cells (3 seeds × 8 states; ties by average ranks).
  - **CI:** 95% percentile, query bootstrap (§10); every replicate recomputes outcomes,
    query-level predictors, then ρ.
  - **Two-sided bootstrap p:** p = min(1, 2 · min(Pr*(ρ* ≤ 0), Pr*(ρ* ≥ 0))), with a floor of
    1/2000.
- **Multiple comparisons:** Holm–Bonferroni over the family of 16 tests (2 policies × 2
  outcomes × 4 predictors), at familywise α = 0.05.
  - **"Associated"** ⇔ Holm-adjusted p < 0.05.
- **Deletion family:** descriptive only, because n = 6 cells with only 2 values of m. Spearman ρ
  and CI are reported; there is no test.
- **Secondary descriptive (A1 H4′ spirit; no decision):**
  - |ρ(structural)| − |ρ(update fraction)| for predictors 2–4, with bootstrap CI;
  - a per-query analysis: Spearman over all insertion (σ, s, q) test rows of overlap decay and
    density drift vs the per-query cost difference C_P − C_REF and vs the newly-failing
    indicator (S0-pass cohort).

### 9. H-E4 — DARTH predicted vs realised recall (descriptive)
- **Per (σ, s):**
  - Q = queries with stopped_early = 1;
  - stopped fraction f = |Q|/2000 (always reported with the metrics);
  - bias = mean_{q∈Q}(pred_q − realised_id_q);
  - MACE = mean_{q∈Q} |pred_q − realised_id_q|.
  - pred_q = last_predicted, the clipped regression output at the stop decision. realised_id_q
    = recall@10 by id.
- **Per-query MACE, no binning:** the predictor is a point regression of id recall, not a
  probability, so a reliability diagram is not the estimand.
- **Pooled:** the mean over σ, with 95% bootstrap CI (§10; Q recomputed per replicate).
- **Change from S0:** bias(σ, s) − bias(σ, S0) and MACE(σ, s) − MACE(σ, S0) are reported, with
  CIs. S0 values come from the Stage-1/C1 evaluations.
- **Secondary (disclosed):** the same metrics against realised tie-aware recall.
- **Required disclosures:**
  - it is conditional on the stopped subset (selection at pred ≥ 0.95), so it is not
    probabilistic calibration;
  - realised recall is discrete (tenths), so MACE has an irreducible floor;
  - cap-run queries are excluded (their last_predicted is a stale check value).
- **No margin and no decision rule** (none was frozen). If |Q| < 30 for a cell, that cell's
  metrics are reported but flagged "small-n".

### 10. Bootstrap specification
- **Resamples:** B = 2,000; numpy.random.default_rng(20261002).
- **Index matrix:** generated once: an integer matrix I of shape (2000, 2000) with entries
  uniform on the 2,000 test-query positions (sorted query_id order).
- **Sharing:** the same I is reused for every seed, state, policy, reference, hypothesis and
  outcome (A1 convention: a query and all its seeds/states/policies move together, so pairing
  is preserved).
- **Unit resampled:** query id; there is no other clustering level. Seeds are fixed (not
  resampled), and their variability enters through the per-seed reporting and §11.
- **Interval:** 95% percentile [2.5%, 97.5%].
- **Ratios and conditional rates** are recomputed from the resampled numerators and
  denominators in each replicate.

### 11. Mixed-effects trend specification (secondary; not part of the H-E1/H-E2 verdicts)
- **Package:** statsmodels (`smf.mixedlm`), version pinned in python/requirements.txt at install.
  Fit with REML = True and default optimiser settings.
- **Outcomes:** D(P, σ, s) (§6.2, the primary H-E1 quantity; raw R and additive ΔR are not
  modelled) and E(P, σ, s), separately for P = B1 and DARTH. That is 4 models,
  each with point estimates on the full test set.
- **Insertion model** (24 cells; S0 excluded):
  `y ~ L + T + L:T`, groups = seed, random intercept only (`re_formula="1"`).
  - L = log2(m / 0.01) ∈ {0, 1, 2, 3}.
  - T = 1 for OOD, 0 for ID.
- **Inferential trend quantity:** the coefficient of L (the ID slope).
  - Reported with its 95% Wald CI.
  - **"Changes with magnitude"** ⇔ the CI excludes 0.
  - The T and L:T terms are descriptive only (ID-vs-OOD is not an E hypothesis, and 8% OOD is
    ≈ ID by construction).
- **Deletion:** no model (2 magnitudes). Report the pooled difference
  y(del_80000) − y(del_20000) with its bootstrap CI (descriptive).
- **Failure handling:**
  - a boundary variance estimate (0) is reported as fitted;
  - a non-converged or failed fit is reported as "fit failed" with the error;
  - no alternative model is substituted.
- **Robustness (descriptive):** the bootstrap CI of the slope of the pooled-over-seed estimate
  on L (ordinary least squares on 4 points per trajectory).

### 12. Multiple-comparison policy (frozen)
- **H-E1 and H-E2 per-state three-valued classifications:** no α adjustment. The family verdict
  is an intersection rule, which is conservative. All cells are reported (A1.7 convention).
- **H-E3:** Holm–Bonferroni over the 16 primary insertion tests (§8).
- **Trend (§11) and H-E4:** secondary/descriptive; unadjusted 95% CIs, labelled secondary.

### 13. Treatment of 8% OOD states
- **H-E1, H-E2, H-E3:** ood_80000 is an ordinary insertion state. It counts in the per-state
  classifications, the family verdicts, the trend model and the H-E3 correlations. It is a valid
  index-change state.
- **ID vs OOD:** any contrast (T and L:T in §11, or a reported ID/OOD difference) is descriptive
  only. At 8% the mandatory A1 wording is attached: "At 8% insertion magnitude, the available
  pool forces the regional-OOD construction to overlap approximately 91% with the matched ID
  set, so this level does not provide a clean ID-vs-OOD comparison."

### 14. Confidence intervals and significance rules (summary)
- **All CIs:** 95%.
- **H-E1/H-E2:** query-bootstrap percentile CIs with the three-valued margin rules (§6.4,
  §7.3).
- **Trend:** MixedLM Wald CI excluding 0 (secondary).
- **H-E3:** Holm-adjusted bootstrap p < 0.05.
- **H-E4:** CIs only, no test.

### 15. Seed handling
- Each seed's frozen policies run only on that seed's states. The reference is calibrated per
  (σ, s).
- Pooled estimates are equal-weight means over σ = 42, 43, 44.
- Per-seed results are always reported, and a seed-specific STALE is listed explicitly.
- Seeds are a random intercept only in §11.
- DARTH seeds 43/44 have caps 142/135 and intervals from their own S0 traces (Option A, Stage 1).

### 16. Missing or failed cells
- A (σ, s) cell fails if any of these hold:
  - a DARTH run errors;
  - a run returns ≠ 2,000 rows;
  - any recall lies outside [0, 1];
  - any deleted or out-of-range label is returned;
  - two_ef_mix returns None;
  - the §5 integrity check fails;
  - any oracle-curve test row is missing.
- **Any failed cell → STOP before analysis and report.** No imputation, no partial analysis, no
  dropping of states or queries.

### 17. Anti-p-hacking rules
- **Nothing is retuned, retrained or changed:**
  - no retuning of B1 and no retraining of DARTH;
  - no change of caps, intervals, target, margins, states, split, cohorts, predictors, models
    or bootstrap;
  - no removal of outlier queries or difficult states;
  - no metric switching;
  - no choosing between policies or estimands after seeing results.
- **The analysis script is committed and its sha256 is recorded before** any evolved-state
  DARTH run.
- **One run only:** the analysis is run once. A re-run is allowed only for a reproducibility
  check (identical output required).
- **Report findings as obtained:** negative, null or inconclusive results are reported as such.
  B1 and DARTH, and insertion and deletion, are reported separately.

### 18. Execution order
1. User approves this document. Append it to design_doc.md as Addendum E1, with any §6.3
   decision recorded.
2. Pin statsmodels in python/requirements.txt.
3. Implement (no outcomes yet):
   - python/exp_e_b1.py: frozen B1 + REF per (σ, s) from the oracle curves, with the §5
     integrity check;
   - python/exp_e_run.py: per-(σ, s) DARTH eval configs and runs;
   - python/exp_e_analysis.py: §6–§13.
   Then record the analysis-script sha256 in notes.
4. **Pre-run checks** (no evolved-state outcomes):
   - REF(σ, S0) == B1(σ) for σ = 42/43/44;
   - the DARTH eval via the E pipeline at S0 reproduces the Stage-1/C1 S0 rows exactly
     (non-timing columns);
   - existing gtests and pytest pass.
5. Run DARTH(σ) on all 30 (σ, s) cells: single-threaded eval, the state's index and ground
   truth, and the frozen policy hash verified.
6. Compute B1 and REF per-query rows for all cells from the oracle curves.
7. Check §16 completeness. If anything fails, STOP.
8. Run python/exp_e_analysis.py once; write results/exp_e_analysis_<ts>/.
9. Determinism check: re-run the analysis; the output must be identical.
10. Write the report (docs/exp_e_report.md) with the A1.10 limitations and the 8% wording.

### 19. Implementation checklist for Claude Code
- [ ] **B1 assignment** via router4b_lib.assign_mix(mix, query_ids, "sift_query", 20261006),
  using the seed's frozen mix from the Phase-4b training record.
- [ ] **REF(σ, s)** via router4b_lib.two_ef_mix on mean training tie-aware recall over the
  121-ef grid, and the same assign_mix.
- [ ] **Integrity check** REF(σ, S0) ≡ B1(σ).
- [ ] **Costs and recalls** of B1 and REF read from oracle_curves.csv (columns
  distance_computations, recall_tie_aware) of (σ, s) at the assigned ef, test rows only.
- [ ] **DARTH eval config per (σ, s):** index_path = the state index; gt_prefix = the state's
  ground-truth cache path from that state's oracle-run metadata; queries/split unchanged;
  ef_cap from the frozen policy; policy_path = the frozen policy file. The deletion path is
  automatic (DeletedCount > 0 → general hnswlib rule).
- [ ] **DARTH columns used:** distance_computations, recall_tie_aware, recall_id, stopped_early,
  last_predicted (and the overhead columns, reported descriptively).
- [ ] **Bootstrap matrix I** built once (§10) and shared everywhere.
- [ ] **H-E1:** primary D (§6.2) with the S0 terms recomputed per replicate; seed-averaged D̄
  classification (STABLE / STALE-COSTLIER / STALE-CHEAPER / INCONCLUSIVE), per-seed,
  family verdict; mean tie-aware recall@10 per (P, σ, s) incl. REF and S0; descriptive only:
  raw R, additive ΔR, per-query relative cost.
- [ ] **H-E2:** cohorts K_P (K_REF = K_B1), the four transition counts, NF, E, pooled
  classification, per-seed, family verdict.
- [ ] **H-E3:** the 16 Spearman tests with bootstrap p and Holm; deletion descriptive;
  secondary descriptives.
- [ ] **H-E4:** f, bias, MACE, changes from S0, tie-aware secondary, small-n flags, disclosures.
- [ ] **§11 MixedLM:** 4 models plus deletion differences and the robustness slope.
- [ ] **Row-level output:** one row per (policy ∈ {B1, DARTH, REF}, σ, state incl. S0, q), with
  cost, recalls, pass, transition and assigned ef / DARTH fields.
- [ ] **Metadata:** config, script hashes, policy/model hashes, submodule commits, seeds.

### Final Freeze Status
| Item | Status |
|---|---|
| 1 H-E1 decision rule (three-valued, 95% percentile, per pooled state, family intersection, per-seed reported) | FROZEN |
| 1a H-E1 primary estimand = S0-anchored proportional D (§6.2/§6.3); raw R and additive ΔR descriptive only | FROZEN (user-approved 2026-10-05) |
| 2 H-E2 decision rule | FROZEN |
| 3 H-E1 cost definition (ratio of means primary; per-query relative secondary) | FROZEN |
| 4 H-E2 denominator (S0-pass cohort of the same policy) | FROZEN |
| 5 Reference S0 cohort = frozen-B1 S0 cohort (same for both comparisons) | FROZEN |
| 6 8% OOD states (ordinary insertion states; ID-vs-OOD descriptive) | FROZEN |
| 7 H-E4 MACE (per-query) + stopped fraction | FROZEN |
| 8 Mixed-effects specification | FROZEN |
| 9 H-E3 predictors / aggregation / statistic / unit / multiplicity | FROZEN |
| 10 Bootstrap | FROZEN |
| 11–19 remaining sections | FROZEN |
