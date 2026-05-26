#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ars/features.h"
#include "ars/hnsw_index.h"

namespace ars {

// Every field comes from the YAML file; there are no in-source defaults for
// experiment parameters, so a missing key is an error.

// Sections shared by every run against the static S0 index: dataset, index,
// ground truth and paths. Identical values here mean the same cached S0
// index and ground truth are reused across phases.
// Optional evolved index state (design_doc.md Addendum A1). Absent = S0.
// Labels: base rows 0..N0-1, then inserted vectors N0.. in insertion order.
struct StateSpec {
  bool present = false;
  std::string name;                // e.g. "id_0.02"; also keys the GT cache
  std::string index_path;          // saved evolved index (never built here)
  std::string inserted_vectors;    // fvecs, insertion order ("" = none)
  std::size_t inserted_count = 0;  // first N rows of inserted_vectors are live
  std::string deleted_labels;      // CSV "label" ("" = none): lazily deleted
};

struct S0Config {
  std::string experiment_name;

  std::string base_path;
  std::string query_path;
  std::size_t max_base = 0;     // 0 = all rows (first N rows otherwise)
  std::size_t max_queries = 0;  // 0 = all rows

  HnswParams index;
  int build_threads = 1;

  std::size_t gt_k = 0;  // depth of stored ground truth (>= search k)

  StateSpec state;

  std::string cache_dir;
  std::string output_dir;

  std::string raw_yaml;  // verbatim file contents, logged into metadata
};

// Phase 1 run: fixed-efSearch sweep against the static S0 index.
struct StaticSweepConfig : S0Config {
  std::size_t k = 0;  // recall@k
  std::vector<std::size_t> ef_values;
  int timing_repeats = 0;
  std::size_t warmup_queries = 0;
};

// Phase 2 run: seeded train/test query split + per-query oracle efSearch.
struct OracleConfig : S0Config {
  std::uint64_t split_seed = 0;
  std::size_t split_test_size = 0;
  // Persistent split file. Written on first run; on later runs the freshly
  // generated split must match it exactly or the run aborts.
  std::string split_path;
  // true: read the row tagging from split_path as-is (no generation). Used for
  // evaluation-only query sets (all rows tagged "test"); split.seed and
  // split.test_size are then not used.
  bool split_fixed_file = false;

  std::size_t k = 0;  // recall@k
  double target_recall = 0.0;
  std::string recall_definition;  // "tie_aware" or "id"
  std::size_t ef_grid_min = 0;
  std::size_t ef_grid_max = 0;
  double ef_grid_ratio = 0.0;
  int search_threads = 1;
};

// Phase 3 run: live routing features for every query on the S0 index.
struct FeatureConfig : S0Config {
  // Existing split file (from Phase 2); used only to tag output rows. The
  // features themselves never depend on it.
  std::string split_path;
  FeatureParams params;
  int threads = 1;
};

// Phase 4 validation run: re-search every query at the ef a policy assigned
// to it (independent check of curve-lookup based evaluation).
struct PolicyEvalConfig : S0Config {
  std::string policy_path;  // CSV "query_id,ef"
  std::size_t k = 0;
  int threads = 1;
};

// Draws a fixed-size query subset from a vector file (Fisher-Yates with
// mt19937_64, as for the train/test split; see query_split.h).
struct QuerySubsetConfig {
  std::string experiment_name;
  std::string source_path;
  std::size_t n_select = 0;
  std::uint64_t seed = 0;
  std::string query_set_id;
  std::string output_fvecs;
  std::string output_ids;      // CSV: row,source_row (ascending source_row)
  std::string output_tagging;  // CSV: query_id,split (all "test")
  std::string output_dir;
  // Population definition (optional; omitted = whole source file, as used
  // for the rejected first confirmation attempt). Rows whose vector is a
  // byte-exact copy of any vector in these files are excluded; with
  // deduplicate_population, each distinct remaining vector is kept once
  // (lowest source row). The seed is applied to this filtered population.
  std::vector<std::string> exclude_exact_duplicates_of;
  bool deduplicate_population = false;
  std::string raw_yaml;
};

// Phase 5' state production (design_doc.md Addendum A1): nested insertion
// states built on an in-process S0 (byte-checked against the cached S0), or
// nested lazy-deletion states on the loaded cached S0.
struct EvolveConfig : S0Config {
  std::string trajectory;       // "insert" or "delete"
  std::string trajectory_name;  // e.g. "id", "ood", "del"
  std::string pool_path;        // insert: fvecs of the insertion pool
  std::string order_path;       // CSV "row": pool rows / base labels, in order
  std::vector<std::size_t> counts;  // nested state sizes (ascending)
  std::string state_dir;
  std::string verify_s0;  // cached S0 index file
};

StaticSweepConfig LoadStaticSweepConfig(const std::string& path);
OracleConfig LoadOracleConfig(const std::string& path);
FeatureConfig LoadFeatureConfig(const std::string& path);
PolicyEvalConfig LoadPolicyEvalConfig(const std::string& path);
QuerySubsetConfig LoadQuerySubsetConfig(const std::string& path);
EvolveConfig LoadEvolveConfig(const std::string& path);

}  // namespace ars
