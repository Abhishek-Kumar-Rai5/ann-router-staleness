#include "ars/config.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ars {
namespace {

YAML::Node Require(const YAML::Node& parent, const std::string& key,
                   const std::string& where) {
  const YAML::Node node = parent[key];
  if (!node) {
    throw std::runtime_error("config: missing required key '" + where + key +
                             "'");
  }
  return node;
}

template <typename T>
T Get(const YAML::Node& parent, const std::string& key,
      const std::string& where) {
  return Require(parent, key, where).as<T>();
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open config " + path);
  }
  std::stringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

void ParseS0(const YAML::Node& root, S0Config& c) {
  c.experiment_name = Get<std::string>(root, "experiment_name", "");

  const YAML::Node ds = Require(root, "dataset", "");
  c.base_path = Get<std::string>(ds, "base", "dataset.");
  c.query_path = Get<std::string>(ds, "queries", "dataset.");
  c.max_base = Get<std::size_t>(ds, "max_base", "dataset.");
  c.max_queries = Get<std::size_t>(ds, "max_queries", "dataset.");

  const YAML::Node idx = Require(root, "index", "");
  c.index.m = Get<std::size_t>(idx, "m", "index.");
  c.index.ef_construction = Get<std::size_t>(idx, "ef_construction", "index.");
  c.index.seed = Get<std::uint64_t>(idx, "seed", "index.");
  c.build_threads = Get<int>(idx, "build_threads", "index.");

  const YAML::Node gt = Require(root, "ground_truth", "");
  c.gt_k = Get<std::size_t>(gt, "k", "ground_truth.");

  const YAML::Node paths = Require(root, "paths", "");
  c.cache_dir = Get<std::string>(paths, "cache_dir", "paths.");
  c.output_dir = Get<std::string>(paths, "output_dir", "paths.");

  if (c.build_threads < 1) {
    throw std::runtime_error("config: index.build_threads must be >= 1");
  }

  if (const YAML::Node st = root["state"]) {
    c.state.present = true;
    c.state.name = Get<std::string>(st, "name", "state.");
    c.state.index_path = Get<std::string>(st, "index_path", "state.");
    c.state.inserted_vectors =
        Get<std::string>(st, "inserted_vectors", "state.");
    c.state.inserted_count = Get<std::size_t>(st, "inserted_count", "state.");
    c.state.deleted_labels = Get<std::string>(st, "deleted_labels", "state.");
    if (c.state.name.empty() || c.state.index_path.empty()) {
      throw std::runtime_error(
          "config: state.name and state.index_path required");
    }
  }
}

}  // namespace

StaticSweepConfig LoadStaticSweepConfig(const std::string& path) {
  StaticSweepConfig c;
  c.raw_yaml = ReadFile(path);
  const YAML::Node root = YAML::Load(c.raw_yaml);
  ParseS0(root, c);

  const YAML::Node search = Require(root, "search", "");
  c.k = Get<std::size_t>(search, "k", "search.");
  c.ef_values = Get<std::vector<std::size_t>>(search, "ef_values", "search.");
  c.timing_repeats = Get<int>(search, "timing_repeats", "search.");
  c.warmup_queries = Get<std::size_t>(search, "warmup_queries", "search.");

  if (c.k == 0 || c.gt_k < c.k) {
    throw std::runtime_error("config: need 0 < search.k <= ground_truth.k");
  }
  if (c.ef_values.empty() || c.timing_repeats < 1) {
    throw std::runtime_error(
        "config: ef_values must be non-empty, timing_repeats >= 1");
  }
  return c;
}

OracleConfig LoadOracleConfig(const std::string& path) {
  OracleConfig c;
  c.raw_yaml = ReadFile(path);
  const YAML::Node root = YAML::Load(c.raw_yaml);
  ParseS0(root, c);

  const YAML::Node split = Require(root, "split", "");
  c.split_path = Get<std::string>(split, "path", "split.");
  c.split_fixed_file = split["fixed_file"] && split["fixed_file"].as<bool>();
  if (!c.split_fixed_file) {
    c.split_seed = Get<std::uint64_t>(split, "seed", "split.");
    c.split_test_size = Get<std::size_t>(split, "test_size", "split.");
  }

  const YAML::Node oracle = Require(root, "oracle", "");
  c.k = Get<std::size_t>(oracle, "k", "oracle.");
  c.target_recall = Get<double>(oracle, "target_recall", "oracle.");
  c.recall_definition =
      Get<std::string>(oracle, "recall_definition", "oracle.");
  const YAML::Node grid = Require(oracle, "ef_grid", "oracle.");
  c.ef_grid_min = Get<std::size_t>(grid, "min", "oracle.ef_grid.");
  c.ef_grid_max = Get<std::size_t>(grid, "max", "oracle.ef_grid.");
  c.ef_grid_ratio = Get<double>(grid, "ratio", "oracle.ef_grid.");
  c.search_threads = Get<int>(oracle, "search_threads", "oracle.");

  if (c.k == 0 || c.gt_k < c.k) {
    throw std::runtime_error("config: need 0 < oracle.k <= ground_truth.k");
  }
  if (std::isnan(c.target_recall) || c.target_recall <= 0.0 ||
      c.target_recall > 1.0) {
    throw std::runtime_error("config: oracle.target_recall must be in (0, 1]");
  }
  if (c.recall_definition != "tie_aware" && c.recall_definition != "id") {
    throw std::runtime_error(
        "config: oracle.recall_definition must be 'tie_aware' or 'id'");
  }
  if (c.ef_grid_min == 0 || c.ef_grid_max < c.ef_grid_min ||
      !(c.ef_grid_ratio > 1.0) || c.search_threads < 1) {
    throw std::runtime_error(
        "config: need 0 < ef_grid.min <= ef_grid.max, ef_grid.ratio > 1, "
        "search_threads >= 1");
  }
  return c;
}

FeatureConfig LoadFeatureConfig(const std::string& path) {
  FeatureConfig c;
  c.raw_yaml = ReadFile(path);
  const YAML::Node root = YAML::Load(c.raw_yaml);
  ParseS0(root, c);

  const YAML::Node split = Require(root, "split", "");
  c.split_path = Get<std::string>(split, "path", "split.");

  const YAML::Node f = Require(root, "features", "");
  c.params.probe_k = Get<std::size_t>(f, "probe_k", "features.");
  c.params.probe_ef = Get<std::size_t>(f, "probe_ef", "features.");
  c.params.lid_k = Get<std::size_t>(f, "lid_k", "features.");
  c.params.lid_ef = Get<std::size_t>(f, "lid_ef", "features.");
  c.threads = Get<int>(f, "threads", "features.");

  if (c.params.probe_k < 2 || c.params.lid_k < 2 || c.threads < 1) {
    throw std::runtime_error(
        "config: need features.probe_k >= 2, features.lid_k >= 2, "
        "features.threads >= 1");
  }
  return c;
}

PolicyEvalConfig LoadPolicyEvalConfig(const std::string& path) {
  PolicyEvalConfig c;
  c.raw_yaml = ReadFile(path);
  const YAML::Node root = YAML::Load(c.raw_yaml);
  ParseS0(root, c);
  const YAML::Node p = Require(root, "policy", "");
  c.policy_path = Get<std::string>(p, "path", "policy.");
  c.k = Get<std::size_t>(p, "k", "policy.");
  c.threads = Get<int>(p, "threads", "policy.");
  if (c.k == 0 || c.gt_k < c.k || c.threads < 1) {
    throw std::runtime_error(
        "config: need 0 < policy.k <= ground_truth.k, policy.threads >= 1");
  }
  return c;
}

QuerySubsetConfig LoadQuerySubsetConfig(const std::string& path) {
  QuerySubsetConfig c;
  c.raw_yaml = ReadFile(path);
  const YAML::Node root = YAML::Load(c.raw_yaml);
  c.experiment_name = Get<std::string>(root, "experiment_name", "");
  c.source_path = Get<std::string>(root, "source", "");
  c.n_select = Get<std::size_t>(root, "n_select", "");
  c.seed = Get<std::uint64_t>(root, "seed", "");
  c.query_set_id = Get<std::string>(root, "query_set_id", "");
  c.output_fvecs = Get<std::string>(root, "output_fvecs", "");
  c.output_ids = Get<std::string>(root, "output_ids", "");
  c.output_tagging = Get<std::string>(root, "output_tagging", "");
  c.output_dir = Get<std::string>(root, "output_dir", "");
  if (root["exclude_exact_duplicates_of"]) {
    c.exclude_exact_duplicates_of =
        root["exclude_exact_duplicates_of"].as<std::vector<std::string>>();
  }
  c.deduplicate_population = root["deduplicate_population"] &&
                             root["deduplicate_population"].as<bool>();
  if (c.n_select == 0) {
    throw std::runtime_error("config: n_select must be > 0");
  }
  return c;
}

EvolveConfig LoadEvolveConfig(const std::string& path) {
  EvolveConfig c;
  c.raw_yaml = ReadFile(path);
  const YAML::Node root = YAML::Load(c.raw_yaml);
  ParseS0(root, c);
  const YAML::Node e = Require(root, "evolution", "");
  c.trajectory = Get<std::string>(e, "trajectory", "evolution.");
  c.trajectory_name = Get<std::string>(e, "name", "evolution.");
  c.order_path = Get<std::string>(e, "order", "evolution.");
  c.counts = Get<std::vector<std::size_t>>(e, "counts", "evolution.");
  c.state_dir = Get<std::string>(e, "state_dir", "evolution.");
  c.verify_s0 = Get<std::string>(e, "verify_s0", "evolution.");
  if (c.trajectory == "insert") {
    c.pool_path = Get<std::string>(e, "pool", "evolution.");
  } else if (c.trajectory != "delete") {
    throw std::runtime_error(
        "config: evolution.trajectory must be insert|delete");
  }
  if (c.counts.empty() || c.build_threads != 1) {
    throw std::runtime_error(
        "config: evolution.counts must be non-empty; build_threads must be 1");
  }
  for (std::size_t i = 1; i < c.counts.size(); ++i) {
    if (c.counts[i] <= c.counts[i - 1]) {
      throw std::runtime_error("config: evolution.counts must be ascending");
    }
  }
  if (c.state.present) {
    throw std::runtime_error("config: ars_evolve starts from S0 (no state:)");
  }
  return c;
}

}  // namespace ars
