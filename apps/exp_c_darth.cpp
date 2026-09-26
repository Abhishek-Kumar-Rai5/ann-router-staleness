// DARTH re-implementation driver (Experiment C). Modes:
//   verify         plain stop-condition search must match searchKnn exactly
//   trace          collect training traces (train split only)
//   eval           run the frozen DARTH policy (test split only)
//   predict_check  compare C++ model predictions with the Python ones
// Usage: exp_c_darth <mode> <config.yaml>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ars/build_info.h"
#include "ars/darth.h"
#include "ars/hnsw_index.h"
#include "ars/metrics.h"
#include "ars/query_split.h"
#include "ars/run_metadata.h"
#include "ars/vecs_io.h"

namespace fs = std::filesystem;
namespace darth = ars::darth;

namespace {

using Clock = std::chrono::steady_clock;

struct Config {
  std::string raw;
  std::string experiment_name;
  std::string index_path;
  std::string queries;
  std::string split_path;
  std::string gt_prefix;
  std::size_t k = 10;
  std::size_t ef_cap = 0;
  std::vector<std::size_t> verify_efs;
  std::string trace_dir;
  std::string policy_path;
  std::string output_dir;
  int threads = 1;
};

Config LoadConfig(const std::string& path) {
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  Config c;
  c.raw = ss.str();
  const YAML::Node y = YAML::Load(c.raw);
  c.experiment_name = y["experiment_name"].as<std::string>();
  c.index_path = y["index_path"].as<std::string>();
  c.queries = y["queries"].as<std::string>();
  c.split_path = y["split_path"].as<std::string>();
  c.gt_prefix = y["gt_prefix"].as<std::string>();
  c.k = y["k"].as<std::size_t>();
  c.ef_cap = y["ef_cap"].as<std::size_t>();
  c.verify_efs = y["verify_efs"].as<std::vector<std::size_t>>();
  c.trace_dir = y["trace_dir"].as<std::string>();
  c.policy_path = y["policy_path"].as<std::string>();
  c.output_dir = y["output_dir"].as<std::string>();
  c.threads = y["threads"].as<int>();
  return c;
}

std::vector<std::size_t> Rows(const std::vector<ars::Split>& split,
                              ars::Split want) {
  std::vector<std::size_t> r;
  for (std::size_t i = 0; i < split.size(); ++i) {
    if (split[i] == want) {
      r.push_back(i);
    }
  }
  return r;
}

bool SameResult(const ars::SearchResult& a, const ars::SearchResult& b) {
  return a.labels == b.labels && a.dists == b.dists &&
         a.distance_computations == b.distance_computations;
}

struct Ctx {
  Config cfg;
  std::string exp_id;
  fs::path out;
  ars::HnswIndex index;
  ars::FloatMatrix queries;
  ars::IdMatrix gt_ids;
  ars::FloatMatrix gt_dists;
  std::vector<ars::Split> split;
};

darth::Params PlainParams(const Ctx& c, std::size_t ef) {
  darth::Params p;
  p.k = c.cfg.k;
  p.ef = ef;
  p.no_deletions = c.index.DeletedCount() == 0;
  return p;
}

std::string Verify(Ctx& c) {
  std::string per_ef = "[";
  bool all_ok = true;
  for (std::size_t e = 0; e < c.cfg.verify_efs.size(); ++e) {
    const std::size_t ef = c.cfg.verify_efs[e];
    const auto ref = c.index.SearchBatch(c.queries, c.cfg.k, ef, c.cfg.threads);
    std::vector<std::uint8_t> ok(c.queries.rows, 0);
    std::vector<std::uint8_t> kind(c.queries.rows, 0);
    std::vector<std::uint64_t> dc(c.queries.rows, 0);
    const auto n = static_cast<std::int64_t>(c.queries.rows);
#pragma omp parallel for schedule(dynamic, 16) num_threads(c.cfg.threads)
    for (std::int64_t q = 0; q < n; ++q) {
      darth::StopCondition sc(PlainParams(c, ef), darth::Mode::kPlain, nullptr,
                              nullptr);
      const auto r =
          c.index.SearchWithStopCondition(c.queries.Row(q), c.cfg.k, sc);
      ok[q] = SameResult(r, ref[q]) && !sc.stopped_early() ? 1 : 0;
      dc[q] = r.distance_computations;
      if (ok[q] == 0) {
        auto sa = r.labels;
        auto sb = ref[q].labels;
        std::sort(sa.begin(), sa.end());
        std::sort(sb.begin(), sb.end());
        kind[q] = r.distance_computations != ref[q].distance_computations ? 1
                  : r.dists != ref[q].dists                               ? 2
                  : sa != sb                                              ? 3
                                                                          : 4;
      }
    }
    std::size_t mism = 0;
    std::uint64_t total = 0;
    std::array<std::int64_t, 5> by_kind{};
    for (std::size_t q = 0; q < ok.size(); ++q) {
      mism += ok[q] == 0 ? 1 : 0;
      total += dc[q];
      ++by_kind[kind[q]];
    }
    std::cerr << "[exp_c] ef=" << ef << " count=" << by_kind[1]
              << " dists=" << by_kind[2] << " idset=" << by_kind[3]
              << " order_only=" << by_kind[4] << "\n";
    all_ok = all_ok && mism == 0;
    per_ef += std::string(e ? ", " : "") +
              ars::JsonObject()
                  .Int("ef", static_cast<std::int64_t>(ef))
                  .Int("queries", n)
                  .Int("mismatches", static_cast<std::int64_t>(mism))
                  .Int("total_distance_computations",
                       static_cast<std::int64_t>(total))
                  .Render();
    std::cerr << "[exp_c] verify ef=" << ef << " mismatches=" << mism << "\n";
  }
  per_ef += "]";
  return ars::JsonObject()
      .Bool("all_identical", all_ok)
      .Raw("per_ef", per_ef)
      .Render();
}

std::string Trace(Ctx& c) {
  const auto rows = Rows(c.split, ars::Split::kTrain);
  const auto ref =
      c.index.SearchBatch(c.queries, c.cfg.k, c.cfg.ef_cap, c.cfg.threads);
  std::vector<std::vector<darth::Observation>> obs(rows.size());
  std::vector<ars::SearchResult> res(rows.size());
  std::vector<int> base_dists(rows.size(), 0);
  const auto n = static_cast<std::int64_t>(rows.size());
#pragma omp parallel for schedule(dynamic, 16) num_threads(c.cfg.threads)
  for (std::int64_t i = 0; i < n; ++i) {
    const std::size_t q = rows[i];
    darth::StopCondition sc(PlainParams(c, c.cfg.ef_cap), darth::Mode::kTrace,
                            nullptr, c.gt_ids.Row(q));
    res[i] = c.index.SearchWithStopCondition(c.queries.Row(q), c.cfg.k, sc);
    obs[i] = sc.observations();
    base_dists[i] = sc.dists();
  }
  std::size_t trace_mismatch = 0;
  std::size_t total_obs = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    trace_mismatch += SameResult(res[i], ref[rows[i]]) ? 0 : 1;
    total_obs += obs[i].size();
  }
  if (trace_mismatch != 0) {
    throw std::runtime_error("STOP: trace-mode search differs from searchKnn");
  }
  fs::create_directories(c.cfg.trace_dir);
  const fs::path bin = fs::path(c.cfg.trace_dir) / "trace_train.f64";
  std::ofstream b(bin, std::ios::binary);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto qid = static_cast<double>(rows[i]);
    for (const auto& o : obs[i]) {
      b.write(reinterpret_cast<const char*>(&qid), sizeof(double));
      b.write(reinterpret_cast<const char*>(o.x.data()),
              sizeof(double) * darth::kNumFeatures);
      b.write(reinterpret_cast<const char*>(&o.recall), sizeof(double));
    }
  }
  b.close();
  std::ofstream qs(c.out / "trace_queries.csv");
  qs << "query_id,observations,base_dists,distance_computations,final_recall_id"
        ",final_recall_tie_aware\n";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const std::size_t q = rows[i];
    qs << q << "," << obs[i].size() << "," << base_dists[i] << ","
       << res[i].distance_computations << ","
       << ars::RecallAtK(c.gt_ids.Row(q), res[i].labels, c.cfg.k) << ","
       << ars::TieAwareRecallAtK(c.gt_dists.Row(q), res[i].dists, c.cfg.k)
       << "\n";
  }
  std::string cols = "[\"query_id\"";
  for (const char* f : darth::kFeatureNames) {
    cols += std::string(", \"") + f + "\"";
  }
  cols += ", \"recall_id\"]";
  return ars::JsonObject()
      .Str("trace_file", bin.string())
      .Raw("columns", cols)
      .Str("dtype", "float64 row-major")
      .Int("queries", n)
      .Int("observations", static_cast<std::int64_t>(total_obs))
      .Int("search_mismatches_vs_searchKnn",
           static_cast<std::int64_t>(trace_mismatch))
      .Render();
}

std::string Eval(Ctx& c) {
  const YAML::Node pol = YAML::LoadFile(c.cfg.policy_path);
  darth::Params p = PlainParams(c, pol["ef_cap"].as<std::size_t>());
  p.target_recall = pol["target_recall"].as<double>();
  p.ipi = pol["ipi"].as<int>();
  p.mpi = pol["mpi"].as<int>();
  if (p.ef != c.cfg.ef_cap || pol["k"].as<std::size_t>() != c.cfg.k) {
    throw std::runtime_error("policy / config mismatch (ef_cap or k)");
  }
  const auto names = pol["feature_order"].as<std::vector<std::string>>();
  for (std::size_t i = 0; i < darth::kNumFeatures; ++i) {
    if (names.at(i) != darth::kFeatureNames[i]) {
      throw std::runtime_error("policy feature order != frozen C1.9 order");
    }
  }
  const darth::Predictor model(pol["model_path"].as<std::string>());
  const auto rows = Rows(c.split, ars::Split::kTest);

  for (const std::size_t q : rows) {
    (void)c.index.Search(c.queries.Row(q), c.cfg.k, c.cfg.ef_cap);
  }
  std::ofstream out(c.out / "darth_eval_rows.csv");
  out << std::setprecision(17);
  out << "query_id,distance_computations,base_dists,dists_at_decision,"
         "stopped_early,predictor_calls,predictor_inferences,"
         "predictor_seconds,last_predicted,recall_id,recall_tie_aware,"
         "wall_seconds,plain_ef50_seconds,plain_ef53_seconds,"
         "plain_ef50_dc,plain_ef53_dc,labels\n";
  for (const std::size_t q : rows) {
    darth::StopCondition sc(p, darth::Mode::kPredict, &model, nullptr);
    const auto t0 = Clock::now();
    const auto r =
        c.index.SearchWithStopCondition(c.queries.Row(q), c.cfg.k, sc);
    const double wall =
        std::chrono::duration<double>(Clock::now() - t0).count();
    auto timed = [&](std::size_t ef, ars::SearchResult* rr) {
      const auto s0 = Clock::now();
      *rr = c.index.Search(c.queries.Row(q), c.cfg.k, ef);
      return std::chrono::duration<double>(Clock::now() - s0).count();
    };
    ars::SearchResult r50;
    ars::SearchResult r53;
    const double w50 = timed(50, &r50);
    const double w53 = timed(53, &r53);
    out << q << "," << r.distance_computations << "," << sc.dists() << ","
        << sc.dists_at_decision() << "," << (sc.stopped_early() ? 1 : 0) << ","
        << sc.predictor_calls() << "," << sc.predictor_inferences() << ","
        << sc.predictor_seconds() << "," << sc.last_predicted() << ","
        << ars::RecallAtK(c.gt_ids.Row(q), r.labels, c.cfg.k) << ","
        << ars::TieAwareRecallAtK(c.gt_dists.Row(q), r.dists, c.cfg.k) << ","
        << wall << "," << w50 << "," << w53 << "," << r50.distance_computations
        << "," << r53.distance_computations << ",";
    for (std::size_t j = 0; j < r.labels.size(); ++j) {
      out << (j ? " " : "") << r.labels[j];
    }
    out << "\n";
  }
  return ars::JsonObject()
      .Int("test_queries", static_cast<std::int64_t>(rows.size()))
      .Str("policy_path", c.cfg.policy_path)
      .Int("ipi", p.ipi)
      .Int("mpi", p.mpi)
      .Num("target_recall", p.target_recall)
      .Int("ef_cap", static_cast<std::int64_t>(p.ef))
      .Render();
}

std::string PredictCheck(Ctx& c) {
  const YAML::Node pol = YAML::LoadFile(c.cfg.policy_path);
  const darth::Predictor model(pol["model_path"].as<std::string>());
  std::ifstream b(fs::path(c.cfg.trace_dir) / "trace_train.f64",
                  std::ios::binary);
  constexpr std::size_t kCols = darth::kNumFeatures + 2;
  std::array<double, kCols> row{};
  std::ofstream out(c.out / "cpp_predictions.csv");
  out << std::setprecision(17) << "row,prediction\n";
  std::size_t i = 0;
  std::size_t n = 0;
  while (b.read(reinterpret_cast<char*>(row.data()), sizeof(row))) {
    // A fixed subsample is enough to confirm C++ and Python load the model and
    // order the features the same way.
    if (i % 997 == 0) {
      darth::Features x{};
      std::copy(row.begin() + 1, row.begin() + 1 + darth::kNumFeatures,
                x.begin());
      out << i << "," << model.Predict(x) << "\n";
      ++n;
    }
    ++i;
  }
  return ars::JsonObject()
      .Int("rows_read", static_cast<std::int64_t>(i))
      .Int("rows_predicted", static_cast<std::int64_t>(n))
      .Render();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " verify|trace|eval|predict_check <config.yaml>\n";
    return 2;
  }
  try {
    const std::string mode = argv[1];
    Config cfg = LoadConfig(argv[2]);
    const std::string exp_id = ars::MakeExperimentId(mode + "\n" + cfg.raw);
    const fs::path out = fs::path(cfg.output_dir) / exp_id;
    fs::create_directories(out);
    std::ofstream(out / "config.yaml") << cfg.raw;
    Ctx c{cfg,
          exp_id,
          out,
          ars::HnswIndex::Load(cfg.index_path, 128),
          ars::ReadFvecs(cfg.queries),
          ars::ReadIvecs(cfg.gt_prefix + ".ivecs"),
          ars::ReadFvecs(cfg.gt_prefix + ".fvecs"),
          ars::ReadSplitCsv(cfg.split_path)};
    if (c.split.size() != c.queries.rows || c.gt_ids.rows != c.queries.rows) {
      throw std::runtime_error("split / ground truth / query count mismatch");
    }
    const auto t0 = Clock::now();
    std::string result;
    if (mode == "verify") {
      result = Verify(c);
    } else if (mode == "trace") {
      result = Trace(c);
    } else if (mode == "eval") {
      result = Eval(c);
    } else if (mode == "predict_check") {
      result = PredictCheck(c);
    } else {
      throw std::runtime_error("unknown mode " + mode);
    }
    const ars::BuildInfo bi = ars::GetBuildInfo();
    const ars::GitInfo git = ars::GetGitInfo(bi.source_dir);
    std::ofstream(out / "metadata.json")
        << ars::JsonObject()
               .Str("experiment_id", exp_id)
               .Str("experiment_name", cfg.experiment_name)
               .Str("phase", "C1_darth_" + mode)
               .Str("config_yaml", cfg.raw)
               .Raw("git", ars::JsonObject()
                               .Str("commit", git.commit)
                               .Bool("dirty", git.dirty)
                               .Render())
               .Str("hnswlib_commit", bi.hnswlib_commit)
               .Str("compiler", bi.compiler_id + " " + bi.compiler_version)
               .Str("cxx_flags", bi.cxx_flags)
               .Num("seconds",
                    std::chrono::duration<double>(Clock::now() - t0).count())
               .Raw("result", result)
               .Render()
        << "\n";
    std::cout << out.string() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "[exp_c] error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
