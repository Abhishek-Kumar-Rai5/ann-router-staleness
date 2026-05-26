// Phase 2: oracle search effort on the static S0 index.
//
// 1. Creates (or re-verifies) the seeded router train/test query split. This
//    happens first and depends only on (num_queries, test_size, seed).
// 2. Searches every query at every ef of a fixed grid (parallel over queries;
//    results are deterministic and identical to single-threaded search).
// 3. Derives each query's oracle label: the smallest grid ef from which the
//    target recall holds at every larger grid ef.
//
// Writes split.csv, oracle_curves.csv (one row per query x grid ef: the full
// per-query recall/effort curve, kept for routing-regret analysis),
// oracle_labels.csv (one row per query) and metadata.json.
//
// Usage: ars_oracle <config.yaml>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ars/build_info.h"
#include "ars/config.h"
#include "ars/hnsw_index.h"
#include "ars/metrics.h"
#include "ars/oracle.h"
#include "ars/query_split.h"
#include "ars/run_metadata.h"
#include "ars/s0_setup.h"

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct CurvePoint {
  double recall = 0.0;
  double recall_tie_aware = 0.0;
  std::uint64_t distance_computations = 0;
};

// Returns the split, persisting it to cfg.split_path on first use and
// otherwise verifying the regenerated split matches the stored file exactly.
std::vector<ars::Split> EstablishSplit(const ars::OracleConfig& cfg,
                                       std::size_t num_queries, bool& created) {
  if (cfg.split_fixed_file) {
    created = false;
    std::vector<ars::Split> fixed = ars::ReadSplitCsv(cfg.split_path);
    if (fixed.size() != num_queries) {
      throw std::runtime_error(
          "fixed split file row count != number of queries");
    }
    return fixed;
  }
  const std::vector<ars::Split> split =
      ars::MakeQuerySplit(num_queries, cfg.split_test_size, cfg.split_seed);
  created = !fs::exists(cfg.split_path);
  if (created) {
    fs::create_directories(fs::path(cfg.split_path).parent_path());
    std::ofstream(cfg.split_path) << ars::SplitToCsv(split);
  } else if (ars::ReadSplitCsv(cfg.split_path) != split) {
    throw std::runtime_error(
        "stored split " + cfg.split_path +
        " differs from the split regenerated from split.seed; refusing to "
        "continue (the query split must never change)");
  }
  return split;
}

std::string JsonSizeArray(const std::vector<std::size_t>& v) {
  std::string out = "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    out += (i == 0 ? "" : ", ") + std::to_string(v[i]);
  }
  return out + "]";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <config.yaml>\n";
    return 2;
  }
  try {
    const std::string config_path = argv[1];
    const ars::OracleConfig cfg = ars::LoadOracleConfig(config_path);
    const std::string exp_id = ars::MakeExperimentId(cfg.raw_yaml);
    const fs::path out_dir = fs::path(cfg.output_dir) / exp_id;
    fs::create_directories(out_dir);
    fs::create_directories(cfg.cache_dir);
    std::ofstream(out_dir / "config.yaml") << cfg.raw_yaml;
    std::cerr << "[ars] experiment " << exp_id << " -> " << out_dir << "\n";

    const ars::S0Data data = ars::LoadS0Data(cfg);
    const std::size_t nq = data.queries.rows;

    // Step 1: the split, before any search or label exists.
    bool split_created = false;
    const std::vector<ars::Split> split =
        EstablishSplit(cfg, nq, split_created);
    const std::string split_csv = ars::SplitToCsv(split);
    std::ofstream(out_dir / "split.csv") << split_csv;
    const auto n_test = static_cast<std::size_t>(
        std::count(split.begin(), split.end(), ars::Split::kTest));
    std::cerr << "[ars] split ("
              << (cfg.split_fixed_file
                      ? std::string("fixed file")
                      : "seed " + std::to_string(cfg.split_seed))
              << "): " << nq - n_test << " train / " << n_test << " test, "
              << (split_created ? "created" : "verified against/read from")
              << " " << cfg.split_path << "\n";

    const ars::S0GroundTruth gt = ars::LoadOrComputeS0GroundTruth(cfg, data);
    ars::S0Index s0 = ars::LoadOrBuildS0Index(cfg, data);
    std::cerr << "[ars] ground truth "
              << (gt.from_cache ? "loaded" : "computed") << ", index "
              << (s0.from_cache ? "loaded" : "built") << " (" << s0.index.Size()
              << " elements)\n";

    // Step 2: per-query recall/effort curve over the fixed ef grid.
    const std::vector<std::size_t> grid = ars::GeometricEfGrid(
        cfg.ef_grid_min, cfg.ef_grid_max, cfg.ef_grid_ratio);
    const std::size_t n_ef = grid.size();
    std::vector<CurvePoint> curve(nq * n_ef);  // [query][grid index]
    const auto t_search = Clock::now();
    for (std::size_t e = 0; e < n_ef; ++e) {
      const auto results = s0.index.SearchBatch(data.queries, cfg.k, grid[e],
                                                cfg.search_threads);
      for (std::size_t q = 0; q < nq; ++q) {
        CurvePoint& p = curve[q * n_ef + e];
        p.recall = ars::RecallAtK(gt.knn.ids.Row(q), results[q].labels, cfg.k);
        p.recall_tie_aware = ars::TieAwareRecallAtK(gt.knn.dists.Row(q),
                                                    results[q].dists, cfg.k);
        p.distance_computations = results[q].distance_computations;
      }
    }
    const double search_seconds = SecondsSince(t_search);
    std::cerr << "[ars] searched " << n_ef << " ef values (" << grid.front()
              << ".." << grid.back() << ") in " << search_seconds << " s\n";

    // Step 3: labels.
    const bool tie_aware = cfg.recall_definition == "tie_aware";
    std::vector<ars::OracleLabel> labels(nq);
    std::size_t reached = 0;
    for (std::size_t q = 0; q < nq; ++q) {
      std::vector<double> r(n_ef);
      for (std::size_t e = 0; e < n_ef; ++e) {
        const CurvePoint& p = curve[q * n_ef + e];
        r[e] = tie_aware ? p.recall_tie_aware : p.recall;
      }
      labels[q] = ars::DeriveOracleLabel(r, cfg.target_recall);
      reached += labels[q].reached ? 1 : 0;
    }

    {
      std::ofstream csv(out_dir / "oracle_curves.csv");
      csv << "query_id,split,ef,recall,recall_tie_aware,distance_"
             "computations\n";
      csv.precision(10);
      for (std::size_t q = 0; q < nq; ++q) {
        for (std::size_t e = 0; e < n_ef; ++e) {
          const CurvePoint& p = curve[q * n_ef + e];
          csv << q << "," << ars::SplitName(split[q]) << "," << grid[e] << ","
              << p.recall << "," << p.recall_tie_aware << ","
              << p.distance_computations << "\n";
        }
      }
      if (!csv) {
        throw std::runtime_error("failed writing oracle_curves.csv");
      }
    }
    {
      std::ofstream csv(out_dir / "oracle_labels.csv");
      csv << "experiment_id,state,query_id,split,k,target_recall,"
             "recall_definition,reached,oracle_ef,oracle_ef_first_reach,"
             "oracle_distance_computations,oracle_recall,"
             "oracle_recall_tie_aware,max_ef,recall_at_max_ef,"
             "recall_tie_aware_at_max_ef\n";
      csv.precision(10);
      for (std::size_t q = 0; q < nq; ++q) {
        const ars::OracleLabel& l = labels[q];
        const CurvePoint& top = curve[q * n_ef + n_ef - 1];
        csv << exp_id << ",S0," << q << "," << ars::SplitName(split[q]) << ","
            << cfg.k << "," << cfg.target_recall << "," << cfg.recall_definition
            << "," << (l.reached ? 1 : 0) << ",";
        if (l.reached) {
          const CurvePoint& p = curve[q * n_ef + l.index];
          csv << grid[l.index] << "," << grid[l.first_reach_index] << ","
              << p.distance_computations << "," << p.recall << ","
              << p.recall_tie_aware;
        } else {
          csv << ",,,,";  // censored: no ef in the grid reaches the target
        }
        csv << "," << grid.back() << "," << top.recall << ","
            << top.recall_tie_aware << "\n";
      }
      if (!csv) {
        throw std::runtime_error("failed writing oracle_labels.csv");
      }
    }

    const ars::BuildInfo bi = ars::GetBuildInfo();
    const ars::GitInfo git = ars::GetGitInfo(bi.source_dir);
    const std::string metadata =
        ars::JsonObject()
            .Str("experiment_id", exp_id)
            .Str("experiment_name", cfg.experiment_name)
            .Str("phase", "2_oracle")
            .Str("config_path", config_path)
            .Str("config_yaml", cfg.raw_yaml)
            .Raw("git", ars::JsonObject()
                            .Str("commit", git.commit)
                            .Bool("dirty", git.dirty)
                            .Render())
            .Raw("build", ars::JsonObject()
                              .Str("compiler_id", bi.compiler_id)
                              .Str("compiler_version", bi.compiler_version)
                              .Str("build_type", bi.build_type)
                              .Str("cxx_flags", bi.cxx_flags)
                              .Int("cxx_standard", bi.cxx_standard)
                              .Int("openmp_version", bi.openmp_version)
                              .Str("hnswlib_commit", bi.hnswlib_commit)
                              .Render())
            .Raw("seeds", ars::JsonObject()
                              .Int("hnsw_level_seed",
                                   static_cast<std::int64_t>(cfg.index.seed))
                              .Int("split_seed",
                                   static_cast<std::int64_t>(cfg.split_seed))
                              .Render())
            .Raw("split",
                 ars::JsonObject()
                     .Str("method",
                          "Fisher-Yates over query ids 0..n-1 with "
                          "std::mt19937_64(seed) and rejection-sampled bounded "
                          "ints; first test_size permutation positions = test")
                     .Int("num_queries", static_cast<std::int64_t>(nq))
                     .Int("test_size", static_cast<std::int64_t>(n_test))
                     .Int("train_size", static_cast<std::int64_t>(nq - n_test))
                     .Bool("fixed_file", cfg.split_fixed_file)
                     .Str("path", cfg.split_path)
                     .Bool("created_this_run", split_created)
                     .Str("fnv1a64_of_csv", ars::Hex64(ars::Fnv1a64(split_csv)))
                     .Render())
            .Raw("oracle",
                 ars::JsonObject()
                     .Int("k", static_cast<std::int64_t>(cfg.k))
                     .Num("target_recall", cfg.target_recall)
                     .Str("recall_definition", cfg.recall_definition)
                     .Str("label_rule",
                          "smallest grid ef from which recall >= target holds "
                          "at every larger grid ef; censored if not reached "
                          "at max grid ef")
                     .Raw("ef_grid", JsonSizeArray(grid))
                     .Int("queries_reached", static_cast<std::int64_t>(reached))
                     .Int("queries_censored",
                          static_cast<std::int64_t>(nq - reached))
                     .Int("search_threads", cfg.search_threads)
                     .Num("search_seconds", search_seconds)
                     .Render())
            .Raw("ground_truth", ars::JsonObject()
                                     .Str("state", "S0")
                                     .Str("cache_path", gt.cache_path)
                                     .Bool("from_cache", gt.from_cache)
                                     .Render())
            .Raw("index", ars::JsonObject()
                              .Str("cache_path", s0.cache_path)
                              .Bool("from_cache", s0.from_cache)
                              .Int("build_threads", cfg.build_threads)
                              .Render())
            .Int("omp_max_threads", omp_get_max_threads())
            .Render();
    std::ofstream(out_dir / "metadata.json") << metadata << "\n";
    std::cerr << "[ars] oracle reached target for " << reached << "/" << nq
              << " queries; wrote " << out_dir << "\n";
    std::cout << out_dir.string() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "[ars] error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
