// Phase 1: static S0 baseline. Builds (or loads) the HNSW index over the base
// set, computes (or loads) exact ground truth, then runs a fixed-efSearch
// sweep over every query, single-threaded, writing one row per
// (query, ef) to results.csv plus a metadata.json alongside it.
//
// Usage: ars_static_sweep <config.yaml>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ars/build_info.h"
#include "ars/config.h"
#include "ars/hnsw_index.h"
#include "ars/metrics.h"
#include "ars/run_metadata.h"
#include "ars/s0_setup.h"
#include "ars/vecs_io.h"

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct QueryOutcome {
  double recall = 0.0;
  double recall_tie_aware = 0.0;
  std::uint64_t distance_computations = 0;
  std::vector<double> latency_us;  // one per timing repeat
};

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <config.yaml>\n";
    return 2;
  }
  try {
    const std::string config_path = argv[1];
    const ars::StaticSweepConfig cfg = ars::LoadStaticSweepConfig(config_path);
    const std::string exp_id = ars::MakeExperimentId(cfg.raw_yaml);
    const fs::path out_dir = fs::path(cfg.output_dir) / exp_id;
    fs::create_directories(out_dir);
    fs::create_directories(cfg.cache_dir);
    std::ofstream(out_dir / "config.yaml") << cfg.raw_yaml;
    std::cerr << "[ars] experiment " << exp_id << " -> " << out_dir << "\n";

    const ars::S0Data data = ars::LoadS0Data(cfg);
    const ars::FloatMatrix& base = data.base;
    const ars::FloatMatrix& queries = data.queries;
    std::cerr << "[ars] base " << base.rows << "x" << base.dim << ", queries "
              << queries.rows << "x" << queries.dim << "\n";

    const ars::S0GroundTruth gt = ars::LoadOrComputeS0GroundTruth(cfg, data);
    std::cerr << "[ars] ground truth (k=" << cfg.gt_k << ") "
              << (gt.from_cache ? "loaded" : "computed") << " in " << gt.seconds
              << " s\n";

    ars::S0Index built = ars::LoadOrBuildS0Index(cfg, data);
    std::cerr << "[ars] index " << (built.from_cache ? "loaded" : "built")
              << " in " << built.seconds << " s (" << built.index.Size()
              << " elements)\n";

    // Sweep. Single-threaded on purpose: latency is only comparable that way
    // (design doc section 13), and Search() sets an index-wide ef. Repeats are
    // the outer loop; recall and distance counts must be identical across
    // repeats (search is deterministic), which is checked below.
    const std::size_t nq = queries.rows;
    const std::size_t n_ef = cfg.ef_values.size();
    std::vector<QueryOutcome> outcomes(n_ef * nq);
    const auto t_sweep = Clock::now();
    for (int rep = 0; rep < cfg.timing_repeats; ++rep) {
      for (std::size_t e = 0; e < n_ef; ++e) {
        const std::size_t ef = cfg.ef_values[e];
        for (std::size_t w = 0; w < std::min(cfg.warmup_queries, nq); ++w) {
          (void)built.index.Search(queries.Row(w), cfg.k, ef);
        }
        for (std::size_t q = 0; q < nq; ++q) {
          const auto t0 = Clock::now();
          const ars::SearchResult r =
              built.index.Search(queries.Row(q), cfg.k, ef);
          const double us =
              std::chrono::duration<double, std::micro>(Clock::now() - t0)
                  .count();
          QueryOutcome& o = outcomes[e * nq + q];
          const double recall =
              ars::RecallAtK(gt.knn.ids.Row(q), r.labels, cfg.k);
          if (rep == 0) {
            o.recall = recall;
            o.recall_tie_aware =
                ars::TieAwareRecallAtK(gt.knn.dists.Row(q), r.dists, cfg.k);
            o.distance_computations = r.distance_computations;
          } else if (recall != o.recall ||
                     r.distance_computations != o.distance_computations) {
            throw std::runtime_error(
                "non-deterministic search at ef=" + std::to_string(ef) +
                " query=" + std::to_string(q));
          }
          o.latency_us.push_back(us);
        }
      }
      std::cerr << "[ars] repeat " << rep + 1 << "/" << cfg.timing_repeats
                << " done (" << SecondsSince(t_sweep) << " s)\n";
    }

    {
      std::ofstream csv(out_dir / "results.csv");
      csv << "experiment_id,state,policy,ef,query_id,k,recall,"
             "recall_tie_aware,distance_computations,latency_us\n";
      csv.precision(10);
      for (std::size_t e = 0; e < n_ef; ++e) {
        for (std::size_t q = 0; q < nq; ++q) {
          const QueryOutcome& o = outcomes[e * nq + q];
          csv << exp_id << ",S0,fixed_ef," << cfg.ef_values[e] << "," << q
              << "," << cfg.k << "," << o.recall << "," << o.recall_tie_aware
              << "," << o.distance_computations << "," << Median(o.latency_us)
              << "\n";
        }
      }
      if (!csv) {
        throw std::runtime_error("failed writing results.csv");
      }
    }

    const ars::BuildInfo bi = ars::GetBuildInfo();
    const ars::GitInfo git = ars::GetGitInfo(bi.source_dir);
    const std::string metadata =
        ars::JsonObject()
            .Str("experiment_id", exp_id)
            .Str("experiment_name", cfg.experiment_name)
            .Str("phase", "1_static_baseline")
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
                              .Render())
            .Raw("data",
                 ars::JsonObject()
                     .Int("base_rows", static_cast<std::int64_t>(base.rows))
                     .Int("query_rows", static_cast<std::int64_t>(nq))
                     .Int("dim", static_cast<std::int64_t>(base.dim))
                     .Str("query_selection",
                          "first max_queries rows in file order")
                     .Render())
            .Raw("ground_truth", ars::JsonObject()
                                     .Str("state", "S0")
                                     .Str("cache_path", gt.cache_path)
                                     .Bool("from_cache", gt.from_cache)
                                     .Num("seconds", gt.seconds)
                                     .Render())
            .Raw("index", ars::JsonObject()
                              .Str("cache_path", built.cache_path)
                              .Bool("from_cache", built.from_cache)
                              .Num("seconds", built.seconds)
                              .Int("build_threads", cfg.build_threads)
                              .Render())
            .Int("omp_max_threads", omp_get_max_threads())
            .Num("sweep_seconds", SecondsSince(t_sweep))
            .Render();
    std::ofstream(out_dir / "metadata.json") << metadata << "\n";
    std::cerr << "[ars] wrote " << (out_dir / "results.csv") << "\n";
    std::cout << out_dir.string() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "[ars] error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
