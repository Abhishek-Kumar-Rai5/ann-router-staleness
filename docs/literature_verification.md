# Literature verification audit (read-only; 2026-10-05)

Scope: verify the literature claims needed to position the completed project (Phases 4/4b, C1,
Phase 5′, Experiment E). No project file other than this one was created or modified, and no
experiment or analysis was run.

**Verification levels used below:**
- **[FULL]:** primary PDF text read for the relevant sections.
- **[ABS]:** primary abstract or metadata page read.
- **[CITED]:** known only through another primary paper's description.

All sources were accessed 2026-10-05. Search coverage is web search plus arXiv, ACM and author
pages; it is **not an exhaustive systematic review** (see §12).

---

## 1. Research question being positioned
"On SIFT1M/hnswlib, do fixed-effort and runtime-adaptive ANN search policies calibrated at S0
remain valid, in cost and recall-contract terms, relative to a freshly calibrated reference, under
bounded index evolution through 1–8% insertion and 2–8% lazy deletion? And does the
cheap-feature→effort relationship used by such policies remain stable?"

**Headline finding of this audit:** the *general* question is not new. Prior work already
evaluates a frozen versus recalibrated adaptive policy after index updates for one method:
Ada-ef §7.5 compares "Stale", incrementally updated and recomputed Ada-ef after 10%/50%
insertions and deletions. Another reports that a learned search model needs retraining after
index compaction (OMEGA, Fig. 7a). The project's earlier documents understated this; see §5.7.

## 2. The seven questions (kept distinct)
| # | Question | Who addresses it (verified) |
|---|---|---|
| 1 | Does an ANN index degrade after updates? | FreshDiskANN, IP-DiskANN, MN-RU (HNSW unreachable points), SPFresh, Big-ANN'23 streaming track; OMEGA Fig. 1 (QPS without compaction) |
| 2 | Does an adaptive method work on a fixed index? | LAET, Tao, DARTH, PiP, Ada-ef, OMEGA, QBAT |
| 3 | Does a learned stopping model generalise across datasets/queries? | DARTH (harder/noisy and OOD queries), OMEGA (across K), QBAT (ScaNN → preliminary HNSW port) |
| 4 | Does a policy remain valid after the index evolves? | **Ada-ef §7.5** (its own policy); OMEGA Fig. 7a (motivational); QBAT §4.5 (argument only, no experiment) |
| 5 | Does an S0-calibrated policy become stale? | **Ada-ef §7.5** ("Stale Ada-ef"); OMEGA Fig. 7a |
| 6 | Can structural index change predict policy drift? | None found |
| 7 | Can recalibration restore the contract? | **Ada-ef §7.5** (incremental vs recomputed); OMEGA (retrain per compaction) |

## 3. Paper-by-paper verification

### 3.1 LAET [FULL, author PDF]
- **Citation:** C. Li, M. Zhang, D. G. Andersen, Y. He. *Improving Approximate Nearest Neighbor
  Search through Learned Adaptive Early Termination.* SIGMOD 2020, pp. 2539–2554.
  DOI 10.1145/3318464.3380600.
- **Code:** github.com/efficient/faiss-learned-termination (Faiss 1.5.2 + LightGBM 2.3.1).
- **Policy:** GBDT models predict a per-query termination condition (amount of search) from the
  query and intermediate search results after a fixed amount of search. As described in DARTH,
  declarative recall needs a tuned multiplier per recall target.
- **Signals:** query features plus runtime intermediate results, collected once during the search.
- **Setting:** Faiss IVF, IMI and HNSW; DEEP10M/1B, SIFT10M/1B, GIST1M (L2). Static indexes.
- **Index evolution, insertion, deletion, frozen-vs-recalibrated, staleness, structural drift:**
  none. "Insert" appears only in describing index construction.
- **Relevance:** background for runtime-feature learned termination (question 2).

### 3.2 Tao [FULL, arXiv 2110.00696]
- **Citation:** K. Yang, H. Wang, B. Xu, W. Wang, Y. Xiao, M. Du, J. Zhou. *Tao: A Learning
  Framework for Adaptive Nearest Neighbor Search using Static Features Only.* arXiv 2021.
- **Policy:** maps the query to local intrinsic dimensionality (LID) *before* execution and
  predicts the termination condition from static features only.
- **Setting:** IMI and HNSW; Deep1B, Sift1B and others. Static indexes; no updates.
- **Relevance:** closest to Phase 4/4b in spirit (static, pre-query features, including LID).

### 3.3 DARTH [FULL, arXiv 2505.19001 and official code]
- **Citation:** M. Chatzakis, Y. Papakonstantinou, T. Palpanas. *DARTH: Declarative Recall
  Through Early Termination for Approximate Nearest Neighbor Search.* Proc. ACM Manag. Data
  3(4), Art. 242, Sept. 2025 (SIGMOD). DOI 10.1145/3749160.
- **Code:** github.com/MChatzakis/DARTH (a FAISS fork).
- **Policy:**
  - a LightGBM recall predictor on 11 in-search features (step, distance computations,
    insertions, first/closest/furthest NN, avg/var/median/p25/p75 of the k-best);
  - invoked at adaptive intervals pi = mpi + (ipi − mpi)(Rt − Rp);
  - stops when predicted recall ≥ Rt.
- **Calibration:** model trained on 10K learning-set queries for one index. The interval
  hyperparameters use a heuristic from the training traces. Training is tied to that index.
- **Setting:** FAISS HNSW (and IVF); SIFT100M, DEEP100M, T2I100M, GLOVE1M, GIST1M; L2 stated
  as the metric. SIFT100M uses M32/efC500/efS500, k mostly 50.
- **Robustness claims:** about **harder (noise-added) query workloads and OOD queries** (T2I) on
  static indexes ("DARTH is the most robust approach"). **No index-update experiments.**
- **Evolution, insertion, deletion, frozen-vs-recalibrated, staleness, structural drift:** none.
- **Formal guarantee:** none; the recall is declarative but empirical.
- **Relevance:** the runtime actuator re-implemented in C1 and frozen in E.

### 3.4 PiP [FULL, author PDF]
- **Citation:** T. Teofili, J. Lin. *Patience in Proximity: A Simple Early Termination Strategy
  for HNSW Graph Traversal in Approximate k-NN Search.* ECIR 2025, pp. 401–407.
  DOI 10.1007/978-3-031-88714-7_39.
- **Policy:** a saturation (patience) heuristic. It halts when the candidate set stops improving
  for a fixed number of steps. No learning; ef must be preset (Ada-ef §2).
- **Setting:** Apache Lucene HNSW; BEIR datasets with dense text embeddings. Static indexes.
- **Evolution, recalibration, staleness:** none.
- **Relevance:** background (non-learned adaptive stopping).

### 3.5 Ada-ef [FULL, arXiv 2512.06636 and official code]
- **Citation:** C. Zhang, R. J. Miller. *Distribution-Aware Exploration for Adaptive HNSW
  Search.* Proc. ACM Manag. Data 4(1), Art. 25, Feb. 2026 (SIGMOD). DOI 10.1145/3786639.
- **Code:** github.com/chaozhang-cs/hnsw-ada-ef (a patched hnswlib v0.8.0).
- **Policy:**
  - rule-based;
  - dataset statistics (mean, covariance) model the query's full-distance-list distribution;
  - a score is computed from distances collected over about the first two base-layer hops
    (runtime);
  - an ef-estimation table, built offline from 200 sampled database vectors, maps score to ef.
- **Calibration:** statistics, samples and table are built for one dataset/index state.
- **Metrics:** FDL distributions are derived for inner product, cosine similarity and cosine
  distance. "The Euclidean (L2) case remains open" (§6.2). The code accepts only "cd"/"ipd".
- **Datasets:** GloVe-100, DeepImage-96, MS MARCO (v1/v2.1), Cohere, LAION (I2I/T2I), and
  synthetic clusters. All are angular. **No SIFT/L2 experiment.**
- **Index evolution — directly studied (§6.3, §7.5, Tables 4–7):**
  - DeepImage and MS MARCO V2.1; batch sizes 10% and 50%.
  - Insertion: build on existing data, then insert the batch.
  - Deletion: "Since HNSWlib does not support in-place deletions … we report the indexing time
    for rebuilding the index". Deletion is a rebuilt index, not lazy markDelete.
  - Three variants: **Stale Ada-ef** (unmodified, pre-update), **Incr. Ada-ef** (incrementally
    updated statistics and sample ground truth, rebuilt table) and **Reco. Ada-ef** (recomputed
    from scratch).
  - Reported: query time and average / P5 / P1 recall.
  - Findings, quoted:
    - "Stale Ada-ef maintains robust performance but may fall slightly below the target recall,
      particularly with large batch sizes (e.g., 50% insertion on DeepImage)";
    - after deletion, Stale Ada-ef "consistently achieves recall above the target … though at
      the expense of slightly higher search time";
    - "Stale Ada-ef performs robustly under small batch updates, whereas Incr. Ada-ef is
      preferable for large batch updates."
  - Not present: no multi-seed CIs, no pre-registered margins, no fixed-effort policy staleness,
    no learned-policy (DARTH/LAET) staleness, no structural predictors.
- **Relevance:** the **closest prior work to questions 4, 5 and 7.**

### 3.6 OMEGA [FULL, arXiv 2603.06159v2, preprint]
- **Citation:** Y. Peng, J. Fan, X. Wei, et al. (Shanghai Jiao Tong University, Alibaba).
  *Efficient K-generalizable Learned Search.* arXiv 2026, v2 21 Sep 2026.
- **Code:** github.com/driPyf/OMEGA.
- **Policy:**
  - reduces top-K learned search to repeated masked top-1 refinement with one model on
    trajectory features (runtime);
  - rank-wise confidence allocation;
  - statistical recall forecast to skip model calls.
- **Setting:** hnswlib and Alibaba's HNSW (also Vamana); BIGANN, BIGANN-1B, DEEP, GIST,
  Text2Image, MS MARCO and production traces.
- **Index evolution:**
  - **Fig. 1:** emulates evolution on DEEP (1M base plus 1M inserted). Without compaction, QPS
    drops 18.5–28.7% (an index-degradation result, question 1).
  - **Fig. 7a, "The necessity of retraining after index compaction to maintain high recall":**
    mean recall versus retrain frequency (every compaction down to every fifth). The text states
    that "each compaction requires retraining to maintain accuracy". The production evaluation
    retrains after every compaction.
  - This is figure-level motivation, not a controlled frozen-vs-recalibrated study. The protocol
    details (states, CIs) are not reported in the main text.
- **Relevance:** direct (if informal) evidence that **learned-search staleness under index
  evolution is a recognised operational problem.** Its regime (compaction rebuilds, large update
  ratios, production) differs from ours.

### 3.7 QBAT [FULL, PVLDB PDF]
- **Citation:** J. Bae, T. J. Ham, A. Li, S. Chockchowwat, Y. Papakonstantinou (Google).
  *QBAT: Model-based Query Budget Autotuner for Clustering-based Approximate Nearest Neighbor
  Search.* PVLDB 19(11): 3091–3104, 2026. DOI 10.14778/3836663.3836675.
- **Policy:**
  - GBDT (or an AlphaEvolve-derived formula) predicts a per-query budget before the main search;
  - features: distances to coarse centroids (top-1 distance, std of centroid distances);
  - performance and consistency (per-query recall target) modes.
- **Setting:** ScaNN (clustering); BigANN, Cohere, OpenAI, T2I. Preliminary HNSWlib port
  predicting ef from graph features (§5.9): a "substantial gap … to the Oracle"; graph-based
  budget modelling "remain[s] open".
- **Evolution:**
  - §4.5 argues that predicted budgets are invariant under stationary updates (features depend
    on fixed centroids), and that non-stationary updates may degrade results and call for
    retraining when "a measured drift metric exceeds a certain threshold".
  - Future work: "online learning and adaptation techniques for incremental indexing".
  - **No update experiments.**
- **Relevance:** closest to Phase 4/4b (pre-search, query-feature budget prediction). It also
  shows graph-index budget prediction is hard and treats incremental indexing as open.

### 3.8 Dynamic / updatable ANN indexes (question 1; background only)
- **FreshDiskANN** [ABS]: Singh, Subramanya, Krishnaswamy, Simhadri. arXiv 2105.09613 (2021).
  Real-time inserts and deletes for graph indexes with batch consolidation; >95% 5-recall@5.
- **IP-DiskANN** [ABS]: Xu, Manohar, Bernstein, Chandramouli, Wen, Simhadri. arXiv 2502.13826
  (2025). In-place insertion and deletion without batch consolidation; "stable recall over
  various lengthy update patterns".
- **MN-RU** [ABS]: Xiao, Zhan, Xi, Hou, Liao. *Enhancing HNSW Index for Real-Time Updates:
  Addressing Unreachable Points and Performance Degradation.* arXiv 2407.07871 (2024). HNSW
  updates create unreachable points and degrade accuracy and efficiency.
- **SPFresh** [metadata only, via search snippets]: Xu et al. SOSP 2023,
  DOI 10.1145/3600006.3613166. In-place partition-index updates with incremental rebalancing.
- **Big-ANN NeurIPS'23 report** [metadata, via search]: Simhadri et al. arXiv 2409.17424. Includes
  a streaming (insert/delete runbook) track ranked by average recall over checkpoints.
- **None of these studies search-effort policies** (fixed ef calibration or adaptive stopping);
  they study index quality and update mechanisms.

### 3.9 Formal recall guarantees / certification (question E)
- **Adaptive Beam Search** [FULL]: Al-Jazzazi, Diwan, Gou, Musco, Musco, Suel. *Distance Adaptive
  Beam Search for Provably Accurate Graph-Based Nearest Neighbor Search.* arXiv 2505.15636
  (2025). A distance-based termination rule with an approximation guarantee on navigable graphs;
  evaluated on HNSW/Vamana. Static; it cites dynamic-data methods only as related work.
- **Certify-then-Rectify** [FULL, abstract and intro]: M. Li, R. Mittal, S. Rana, S. Shetiya,
  G. Das, N. Koudas. *HNSW with Accuracy Guarantees Using Graph Spanners.* arXiv 2607.02338
  (2026). A distribution-free statistical certifier for HNSW results, plus exact recovery.
  Static.
- **ProS** [CITED, via DARTH]: Echihabi et al., VLDB Journal 2023. Progressive k-NN search with
  probabilistic quality guarantees (data-series indexes). Not read.
- None evaluates guarantee validity after index evolution.

## 4. Implementation claims that shaped our decisions
**Ada-ef:**
- Index dependency: a patched hnswlib (adds an adaptive base-layer search) plus Eigen and
  Boost.Math.
- Metrics: inner product and cosine only. L2 is explicitly open in the paper, and the code rejects
  other metrics.
- No SIFT/L2 demonstration.
- Using it on SIFT1M/L2 would require deriving an L2 estimator (a methodological extension) or
  changing the metric. **The project's rejection is verified correct.**

**DARTH:**
- Index dependency: a FAISS fork; FAISS HNSW and IVF.
- The paper states L2; experiments are on SIFT100M/GIST1M (L2) and others.
- It does not match our SIFT1M / hnswlib / M16 / efC200 / k = 10 setting.
- C1 is a **mechanism-faithful independent re-implementation**, not a reproduction:
  - hnswlib stop-condition API in place of the FAISS search loop;
  - the feature order fed in the training order, deviating from the reference runtime code;
  - the 8,000 Phase-2 training queries in place of a 10K learning set;
  - the search cap set by the paper's rule.

  No published number exists for our configuration.
- Robustness/generalisation claims concern harder and OOD *queries* on static indexes, **not
  index updates**.

## 5. Flags: claims in our project documents vs the primary sources
1. **docs/paper_readiness_audit.md §14:** "Ada-ef's repository contains incremental
   update/deletion experiments for its own method".
   - **Understated:** the *paper* (§7.5) directly evaluates Stale vs Incremental vs Recomputed
     Ada-ef after 10%/50% insertion and deletion. That is a frozen-vs-recalibrated policy-validity
     evaluation (questions 4, 5, 7).
2. **docs/paper_readiness_audit.md §14, "Supported empirical gap":** "pre-registered measurement
   of S0-calibration validity of fixed and runtime-adaptive effort policies under controlled
   insertion/deletion".
   - **Must be narrowed:** Ada-ef and OMEGA already address policy validity / retraining under
     evolution, for their own methods.
   - What remains distinct is listed in §7.
3. **"The literature audit motivated the shift toward policy validity":** the literature does
   motivate it (OMEGA states retraining is necessary; QBAT names incremental indexing as open;
   Ada-ef reports stale behaviour).
   - The earlier literature discussion that drove the reframing is still undocumented in the
     repository.
   - This document can serve as the documented basis.
4. **DARTH venue:** the project documents say "SIGMOD'26 … PACMMOD 3(4) art. 242", which is
   consistent with the official record (Proc. ACM Manag. Data 3(4), Sept. 2025, DOI
   10.1145/3749160). Cite the DOI.
5. **Our "OOD" construction vs DARTH's OOD:** DARTH's OOD means OOD *queries* (T2I). Ours means
   regionally concentrated *inserted vectors*. Do not equate them.
6. **No project claim that "no prior work studies staleness" was found in the reports.** The
   readiness audit correctly marked novelty as unverified; the verification now shows partial
   prior coverage.

## 6. Research-gap test
**Verdict: B. Closely related work exists, and the specific question studied here has not, in the
works identified, been directly evaluated in this form.** It is not A: no identified work
measures the same estimands on the same kind of policies. It is not C: Ada-ef §7.5 and OMEGA are
close.

The closest papers, and what they do **not** evaluate:
1. **Ada-ef §7.5:**
   - only its own rule-based estimator, on angular datasets;
   - 10% / 50% batch updates; deletion by rebuilding the index;
   - recall and latency only (no distance-computation cost);
   - no multi-seed CIs or pre-registered margins;
   - no fixed-effort baseline staleness, no learned (DARTH-type) policy staleness;
   - no separate cost-vs-contract staleness estimands, no S0-anchored drift;
   - no structural predictors of drift; no feature→effort signal analysis.
2. **OMEGA Fig. 7a:**
   - motivational evidence that a learned search model needs retraining after compaction
     (rebuilt index, production setting);
   - no controlled states, CIs, fixed-effort comparison, structural analysis or bounded-churn
     regime.
3. **QBAT §4.5:**
   - an analytical argument (stationary updates leave budgets valid) and a retraining policy
     suggestion;
   - no update experiments; graph-index budget prediction left open.

Index-update papers (FreshDiskANN, IP-DiskANN, MN-RU, SPFresh, Big-ANN streaming) measure index
recall and throughput under updates, not the validity of search-effort policies.

## 7. Novelty language
**SAFE TO SAY:**
- Adaptive search-effort methods (LAET, Tao, DARTH, PiP, Ada-ef, OMEGA, QBAT) are primarily
  developed and evaluated on static indexes.
- Ada-ef reports that its stale (pre-update) estimator stays near target under small batch
  updates but can fall below target after large insertions; OMEGA reports that learned search
  models need retraining after index compaction.
- Ada-ef's published estimator does not support L2.
- DARTH's published robustness evaluation concerns harder and OOD queries on static indexes.
- QBAT identifies adaptation for incremental indexing as future work.

**POSSIBLE GAP (cautious wording):**
- "We are not aware of work that evaluates, with pre-registered margins and multiple index
  seeds, whether S0-calibrated *fixed-effort* and *learned runtime* (DARTH-style) policies remain
  valid under controlled insertion and lazy deletion, relative to a freshly calibrated
  reference."
- "Existing studies we identified do not separate cost drift from recall-contract drift, or
  relate such drift to structural measures of index change."
- "Prior work primarily evaluates update robustness for a single proposed method (Ada-ef) or
  reports retraining necessity operationally (OMEGA)."

**DO NOT SAY:**
- "first" / "novel" / "no prior work studies policy staleness under index updates"; this is
  contradicted by Ada-ef §7.5 and OMEGA.
- That DARTH was reproduced (it was re-implemented), or that DARTH is robust to index evolution
  per its authors.
- That our STABLE result contradicts OMEGA: the regimes differ (compaction rebuilds and large
  update ratios vs ≤ 8% lazy churn).
- Any generalisation beyond SIFT1M/hnswlib/k = 10/≤ 8% churn.

## 8. Positioning the project
- **Phase 4/4b (static query-feature routing):** same family as Tao (static LID features) and
  QBAT (pre-search budget prediction). The negative S0 result on HNSW is consistent with QBAT's
  report that graph-index budget prediction leaves a "substantial gap" to the oracle. Position it
  as scoped evidence on SIFT1M/HNSW, not a general verdict.
- **C1 (runtime-adaptive, DARTH-style):** a mechanism-faithful re-implementation on hnswlib. The
  negative S0 efficiency result (≈ −11% vs a calibrated two-ef B1 at mean recall 0.95) is specific
  to that comparison. DARTH's paper compares mainly against plain fixed-ef search, LAET and REM,
  not a calibrated mixed-ef baseline at matched contract. That difference in comparator, not a
  contradiction, should be stated.
- **Phase 5′ (feature→effort stability):** no identified work studies how query-difficulty
  features' predictive relationship evolves with index updates. It is the most distinct piece,
  still to be worded cautiously.
- **Experiment E (frozen-policy validity):** closest to Ada-ef §7.5 and OMEGA Fig. 7a. It adds:
  - a fixed-effort policy and a DARTH-style learned policy under the same protocol;
  - a state-local recalibrated reference;
  - S0-anchored proportional cost drift and excess contract loss with pre-registered margins and
    three-seed CIs;
  - lazy deletion (markDelete) rather than rebuild;
  - L2/SIFT1M;
  - structural-change associations.
- **Coherence:** yes, as a progression: S0 adaptivity did not pay off → are the ingredients
  (signal) and calibrations stable under evolution? The framing should cite Ada-ef §7.5 and OMEGA
  as prior evidence that staleness matters, and present E as a controlled, policy-comparative,
  cost-vs-contract measurement in a bounded regime.
- **Misleading if left unchanged:** any framing implying policy staleness under updates is an
  unstudied question.

## 9. Literature matrix
| Paper | Year | Adaptive mechanism | Query features (pre-search) | Runtime signals | Index evolution | Insertions | Deletions | Frozen-policy evaluation | Recalibration | Structural drift | Formal quality guarantee | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LAET | 2020 | GBDT termination prediction | yes (query) | yes (intermediate results) | no | no | no | no | no | no | no | Background (q2) |
| Tao | 2021 | LID-based termination prediction | yes (static only) | no | no | no | no | no | no | no | no | Phase 4/4b analogue |
| DARTH | 2025 | In-search recall predictor + early stop | no | yes (11 features) | no | no | no | no | no | no | no (declarative, empirical) | C1/E actuator |
| PiP | 2025 | Patience/saturation heuristic | no | yes (improvement saturation) | no | no | no | no | no | no | no | Background |
| Ada-ef | 2026 | Distribution-based ef estimation | dataset statistics | yes (early distances) | **yes** | **yes (10/50%)** | **yes (rebuild)** | **yes (Stale)** | **yes (Incr./Reco.)** | no | no (approximate) | **Closest (q4, q5, q7)** |
| OMEGA | 2026 (preprint) | K-generalizable learned search | no | yes (trajectory) | **yes (compaction)** | yes | yes (via compaction) | partial (retrain frequency, Fig. 7a) | yes (retrain per compaction) | no | no | **Close (q5, q7)** |
| QBAT | 2026 | Budget prediction (GBDT / formula) | yes (centroid distances) | no | discussed only | no | no | no | proposed only | no | no | Phase 4/4b analogue; open problem |
| FreshDiskANN | 2021 | — (index) | — | — | yes | yes | yes | — | — | index quality | no | Background (q1) |
| IP-DiskANN | 2025 | — (index) | — | — | yes | yes | yes | — | — | index quality | no | Background (q1) |
| MN-RU (HNSW updates) | 2024 | — (index) | — | — | yes | yes | yes | — | — | unreachable points | no | Background (q1) |
| Adaptive Beam Search | 2025 | Distance-adaptive termination | no | yes | no | no | no | no | no | no | **yes** (navigable graphs) | Guarantee background |
| Certify-then-Rectify | 2026 (preprint) | Statistical certification + exact recovery | no | yes | no | no | no | no | no | no | **yes** (statistical) | Guarantee background |

## 10. Final paper positioning
**A. Title:** "Do Calibrated Search-Effort Policies Go Stale? A Pre-registered Study of Fixed and
Runtime-Adaptive HNSW Policies under Bounded Index Evolution".

**B. Research question:** within 1–8% insertion and 2–8% lazy deletion on SIFT1M/hnswlib, do
S0-calibrated fixed-effort and DARTH-style runtime-adaptive policies drift in cost or
recall-contract validity relative to a freshly calibrated reference? Does the cheap-feature→effort
relationship stay stable? And is any drift associated with structural index change?

**C. Research-gap statement:** Adaptive search-effort methods for graph indexes (LAET, Tao,
DARTH, PiP, Ada-ef, OMEGA, QBAT) are developed and evaluated mainly on static indexes, while
production indexes evolve. Two lines of work show this matters:
- Ada-ef reports that its stale estimator can fall below its recall target after large batch
  insertions;
- OMEGA reports that learned search models need retraining after index compaction.

These evaluations cover single proposed methods, report recall and latency, and do not separate
cost drift from recall-contract drift, compare against a freshly calibrated reference with
pre-registered margins across index seeds, include a fixed-effort policy, or relate drift to
structural change. We are not aware of a controlled study doing so. We provide one, in a bounded
regime.

**D. Contribution statement:**
1. A pre-registered, multi-seed protocol with S0-anchored cost-drift and excess-contract-loss
   estimands against a state-local recalibrated reference.
2. Evidence that, within ≤ 8% insertion and lazy deletion on SIFT1M/hnswlib, both a fixed-effort
   and a DARTH-style policy stay within ±10% / ±5 pp, while measurable policy-specific drift
   occurs.
3. Stability of the cheap-feature→effort signal under the same evolution.
4. Scoped negative S0 results for static-feature routing and a DARTH re-implementation against a
   calibrated fixed-effort baseline.

**E. Related-work structure:**
1. Adaptive search effort and early termination: LAET, Tao, DARTH, PiP, Ada-ef, OMEGA, QBAT.
2. Dynamic and updatable graph indexes: FreshDiskANN, IP-DiskANN, MN-RU, SPFresh, Big-ANN
   streaming.
3. Policy staleness under index updates: Ada-ef §7.5, OMEGA, QBAT §4.5.
4. Quality guarantees and certification: Adaptive Beam Search, Certify-then-Rectify, ProS.

**F. Three strongest comparison papers:** Ada-ef (Zhang & Miller 2026), DARTH (Chatzakis et al.
2025), OMEGA (Peng et al. 2026, preprint).

**G. Novelty claimable:** ONLY WITH CAUTIOUS WORDING.

**H. Safe claims:** those under "SAFE TO SAY" in §7, plus the audited results in
docs/exp_e_report.md §H, phrased as "in the tested regime".

## 11. Limitations revealed by the literature (recorded only; no redesign)
- **Update magnitude:** Ada-ef tests 10–50% batches and OMEGA large update ratios with
  compaction; our ≤ 8% lazy churn is milder. Results may differ under larger or rebuilt-index
  evolution (future work).
- **Datasets:** prior adaptive work emphasises modern embedding datasets (cosine); ours is
  SIFT1M/L2 only.
- **Comparators:** we do not compare against Ada-ef (L2-incompatible), OMEGA or QBAT.
- **Metric:** prior work reports latency; ours reports distance computations only.

## 12. Audit limitations
- Not an exhaustive systematic review; the 2025–2026 literature is moving quickly (OMEGA and
  Certify-then-Rectify are preprints).
- Not read in full: SPFresh, Big-ANN'23 report, FreshDiskANN, IP-DiskANN and MN-RU (abstracts
  only), ProS (cited only), and an OpenReview paper on evaluating deletions in graph ANN indexes
  (access blocked).
- A structured DBLP/Scholar search for "learned ANN" × "update/drift/retrain" is recommended
  before submission.

---

## Final verdict
1. **What prior work establishes:**
   - adaptive search-effort methods work on static indexes;
   - graph indexes degrade under updates without maintenance;
   - for specific methods, staleness after updates has been observed (Ada-ef: stale estimator
     robust to small batches, below target after large insertions) and retraining is used
     operationally (OMEGA).
2. **What our experiments add:**
   - a controlled, pre-registered, multi-seed measurement of S0-calibration validity for a
     fixed-effort and a DARTH-style learned policy under bounded insertion and lazy deletion on
     L2/hnswlib, with cost and contract drift separated and S0-anchored, against a state-local
     recalibrated reference;
   - feature→effort signal stability;
   - structural-change associations;
   - scoped negative S0 efficiency results.
3. **Strongest defensible gap (cautious):** a controlled, policy-comparative, cost-vs-contract
   staleness measurement with pre-registered margins. "We are not aware of" wording; not "first".
4. **Should the framing change?** Yes, modestly. Related work must cite Ada-ef §7.5 and OMEGA as
   prior evidence on policy staleness. E should be framed as a controlled complement in a bounded
   regime, not as opening a new question. The readiness audit's §14 gap statement should be
   narrowed accordingly (not edited here).
5. **Ready for figures and drafting?** Yes, provided Introduction and Related Work use the
   positioning above. A structured bibliographic search (§12) is advisable before submission,
   not before drafting.
