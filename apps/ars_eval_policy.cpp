// Phase 4 validation: re-run the real hnswlib search for every query at the
// efSearch a policy assigned to it, against the verified S0 ground truth.
// Used to check independently that curve-lookup based evaluation (Python)
// matches actual search results. Deterministic; no timing.
//
// Usage: ars_eval_policy <config.yaml>   (writes policy_eval.csv)

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "ars/build_info.h"
#include "ars/config.h"
#include "ars/hnsw_index.h"
#include "ars/metrics.h"
#include "ars/policy.h"
#include "ars/run_metadata.h"
#include "ars/s0_setup.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <config.yaml>\n";
    return 2;
  }
  try {
    const ars::PolicyEvalConfig cfg = ars::LoadPolicyEvalConfig(argv[1]);
    const std::string exp_id = ars::MakeExperimentId(cfg.raw_yaml);
    const fs::path out_dir = fs::path(cfg.output_dir) / exp_id;
    fs::create_directories(out_dir);
    std::ofstream(out_dir / "config.yaml") << cfg.raw_yaml;

    const ars::S0Data data = ars::LoadS0Data(cfg);
    const ars::S0GroundTruth gt = ars::LoadOrComputeS0GroundTruth(cfg, data);
    ars::S0Index s0 = ars::LoadOrBuildS0Index(cfg, data);
    const std::vector<ars::PolicyEntry> policy =
        ars::ReadPolicyCsv(cfg.policy_path);

    // Group queries by ef so each group is one race-free SearchBatch.
    std::map<std::size_t, std::vector<std::size_t>> by_ef;
    for (const auto& e : policy) {
      if (e.query_id >= data.queries.rows) {
        throw std::runtime_error("policy query id out of range");
      }
      by_ef[e.ef].push_back(e.query_id);
    }
    std::ofstream csv(out_dir / "policy_eval.csv");
    csv << "query_id,ef,recall,recall_tie_aware,distance_computations\n";
    csv.precision(10);
    for (const auto& [ef, ids] : by_ef) {
      ars::FloatMatrix sub{std::vector<float>(ids.size() * data.queries.dim),
                           ids.size(), data.queries.dim};
      for (std::size_t i = 0; i < ids.size(); ++i) {
        std::copy(data.queries.Row(ids[i]),
                  data.queries.Row(ids[i]) + data.queries.dim, sub.Row(i));
      }
      const auto res = s0.index.SearchBatch(sub, cfg.k, ef, cfg.threads);
      for (std::size_t i = 0; i < ids.size(); ++i) {
        const std::size_t q = ids[i];
        csv << q << "," << ef << ","
            << ars::RecallAtK(gt.knn.ids.Row(q), res[i].labels, cfg.k) << ","
            << ars::TieAwareRecallAtK(gt.knn.dists.Row(q), res[i].dists, cfg.k)
            << "," << res[i].distance_computations << "\n";
      }
    }
    if (!csv) {
      throw std::runtime_error("failed writing policy_eval.csv");
    }
    const ars::BuildInfo bi = ars::GetBuildInfo();
    const ars::GitInfo git = ars::GetGitInfo(bi.source_dir);
    std::ofstream(out_dir / "metadata.json")
        << ars::JsonObject()
               .Str("experiment_id", exp_id)
               .Str("phase", "4_policy_validation")
               .Str("config_path", argv[1])
               .Str("config_yaml", cfg.raw_yaml)
               .Raw("git", ars::JsonObject()
                               .Str("commit", git.commit)
                               .Bool("dirty", git.dirty)
                               .Render())
               .Raw("build", ars::JsonObject()
                                 .Str("compiler_version", bi.compiler_version)
                                 .Str("cxx_flags", bi.cxx_flags)
                                 .Str("hnswlib_commit", bi.hnswlib_commit)
                                 .Render())
               .Str("ground_truth_cache", gt.cache_path)
               .Str("index_cache", s0.cache_path)
               .Int("policy_rows", static_cast<std::int64_t>(policy.size()))
               .Render()
        << "\n";
    std::cerr << "[ars] evaluated " << policy.size() << " queries at "
              << by_ef.size() << " distinct ef values -> " << out_dir << "\n";
    std::cout << out_dir.string() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "[ars] error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
