#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ars/features.h"
#include "ars/hnsw_index.h"

namespace ars {

// Every value comes from the YAML config. There are no in-code defaults for
// experiment parameters, so a missing key is an error.
struct StateSpec {
  bool present = false;
  std::string name;
  std::string index_path;
  std::string inserted_vectors;
  std::size_t inserted_count = 0;  // only the first N rows are live
  std::string deleted_labels;
};

struct S0Config {
  std::string experiment_name;

  std::string base_path;
  std::string query_path;
  std::size_t max_base = 0;
  std::size_t max_queries = 0;

  HnswParams index;
  int build_threads = 1;

  std::size_t gt_k = 0;

  StateSpec state;

  std::string cache_dir;
  std::string output_dir;

  std::string raw_yaml;
};

struct StaticSweepConfig : S0Config {
  std::size_t k = 0;
  std::vector<std::size_t> ef_values;
  int timing_repeats = 0;
  std::size_t warmup_queries = 0;
};

struct OracleConfig : S0Config {
  std::uint64_t split_seed = 0;
  std::size_t split_test_size = 0;
  std::string split_path;
  // split_path is written on the first run, and later runs must regenerate
  // exactly the same split or they abort. With split_fixed_file it is read
  // as-is instead (used for evaluation-only query sets).
  bool split_fixed_file = false;

  std::size_t k = 0;
  double target_recall = 0.0;
  std::string recall_definition;
  std::size_t ef_grid_min = 0;
  std::size_t ef_grid_max = 0;
  double ef_grid_ratio = 0.0;
  int search_threads = 1;
};

struct FeatureConfig : S0Config {
  std::string split_path;
  FeatureParams params;
  int threads = 1;
};

struct PolicyEvalConfig : S0Config {
  std::string policy_path;
  std::size_t k = 0;
  int threads = 1;
};

struct QuerySubsetConfig {
  std::string experiment_name;
  std::string source_path;
  std::size_t n_select = 0;
  std::uint64_t seed = 0;
  std::string query_set_id;
  std::string output_fvecs;
  std::string output_ids;
  std::string output_tagging;
  std::string output_dir;
  std::vector<std::string> exclude_exact_duplicates_of;
  bool deduplicate_population = false;
  std::string raw_yaml;
};

struct EvolveConfig : S0Config {
  std::string trajectory;
  std::string trajectory_name;
  std::string pool_path;
  std::string order_path;
  std::vector<std::size_t> counts;
  std::string state_dir;
  std::string verify_s0;
};

StaticSweepConfig LoadStaticSweepConfig(const std::string& path);
OracleConfig LoadOracleConfig(const std::string& path);
FeatureConfig LoadFeatureConfig(const std::string& path);
PolicyEvalConfig LoadPolicyEvalConfig(const std::string& path);
QuerySubsetConfig LoadQuerySubsetConfig(const std::string& path);
EvolveConfig LoadEvolveConfig(const std::string& path);

}  // namespace ars
