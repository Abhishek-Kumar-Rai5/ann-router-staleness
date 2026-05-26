# Third-party code, methods and attribution

## Libraries (used as dependencies, under their own licences)
| Path | Project | Licence | Use |
|---|---|---|---|
| third_party/hnswlib | https://github.com/nmslib/hnswlib (v0.8.0) | Apache-2.0 | HNSW index; used as a black box via its public API |
| third_party/yaml-cpp | https://github.com/jbeder/yaml-cpp | MIT | experiment configs |
| third_party/googletest | https://github.com/google/googletest | BSD-3-Clause | unit tests |
| third_party/LightGBM | https://github.com/microsoft/LightGBM (v4.6.0) | MIT | Experiment C recall predictor (C API inference); Python package `lightgbm==4.6.0` for training |

Each library keeps its own licence file in its directory.

All four libraries are pinned git submodules: hnswlib v0.8.0 (3f342966), googletest
52eb8108, yaml-cpp 0.8.0 (f7320141), LightGBM v4.6.0 (d02a01ac). CMake links LightGBM's
static library, built once with:
`cmake -S third_party/LightGBM -B third_party/LightGBM/build -DBUILD_STATIC_LIB=ON
-DBUILD_CLI=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build third_party/LightGBM/build -j`.
Without it, the Experiment C targets are skipped and everything else builds.

## Published methods re-implemented in this repository

### DARTH (Experiment C)
Manos Chatzakis, Yannis Papakonstantinou, Themis Palpanas. *DARTH: Declarative Recall
Through Early Termination for Approximate Nearest Neighbor Search.* Proc. ACM Manag. Data
3(4), Article 242 (SIGMOD 2026). arXiv:2505.19001.
Official implementation (a FAISS fork): https://github.com/MChatzakis/DARTH.

- Our implementation (`include/ars/darth.h`, `src/darth.cpp`, `apps/exp_c_darth.cpp`,
  `python/exp_c_*.py`) is an **independent re-implementation** of the published DARTH
  mechanism. It is adapted to hnswlib's public stop-condition API and was written from the
  paper and from our reading of the reference implementation's behaviour.
- **No DARTH source code and no modified FAISS source is copied into this repository.**
- **Documented deviation:** the reference runtime passes the 11 features to the model in
  an order that differs from the order the models are trained with (average ↔ furthest
  distance swapped; median and 25th percentile swapped). We use the training order, so the
  intended DARTH policy is evaluated. Full details: `docs/design_doc.md`, Addendum C1
  (C1.9).

### Considered and not used
Ada-ef (Chao Zhang, Renée J. Miller, *Distribution-Aware Exploration for Adaptive HNSW
Search*, SIGMOD 2026; https://github.com/chaozhang-cs/hnsw-ada-ef) was evaluated for
Experiment C and rejected: its published estimator does not support L2 distance (see
`docs/notes.md`). No Ada-ef code is used.
