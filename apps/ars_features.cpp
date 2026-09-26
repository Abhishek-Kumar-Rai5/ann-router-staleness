// Computes the live routing features for every query. It never reads ground
// truth or oracle labels; the split file is only used to tag rows.
// Usage: ars_features <config.yaml>

#include <omp.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ars/build_info.h"
#include "ars/config.h"
#include "ars/features.h"
#include "ars/query_split.h"
#include "ars/run_metadata.h"
#include "ars/s0_setup.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <config.yaml>\n";
    return 2;
  }
  try {
    const std::string config_path = argv[1];
    const ars::FeatureConfig cfg = ars::LoadFeatureConfig(config_path);
    const std::string exp_id = ars::MakeExperimentId(cfg.raw_yaml);
    const fs::path out_dir = fs::path(cfg.output_dir) / exp_id;
    fs::create_directories(out_dir);
    std::ofstream(out_dir / "config.yaml") << cfg.raw_yaml;
    std::cerr << "[ars] experiment " << exp_id << " -> " << out_dir << "\n";

    const ars::S0Data data = ars::LoadS0Data(cfg);
    const std::size_t nq = data.queries.rows;
    const std::vector<ars::Split> split = ars::ReadSplitCsv(cfg.split_path);
    if (split.size() != nq) {
      throw std::runtime_error(
          "split file has " + std::to_string(split.size()) +
          " rows but there are " + std::to_string(nq) + " queries");
    }

    ars::S0Index s0 = ars::LoadOrBuildS0Index(cfg, data);
    std::cerr << "[ars] index " << (s0.from_cache ? "loaded" : "built") << " ("
              << s0.index.Size() << " elements)\n";

    const std::vector<double> centroid =
        ars::ComputeCentroid(data.base, data.excluded);
    std::ostringstream centroid_text;
    centroid_text.precision(17);
    for (const double c : centroid) {
      centroid_text << c << ",";
    }

    const std::vector<ars::QueryFeatures> feats = ars::ExtractFeatures(
        s0.index, data.queries, centroid, cfg.params, cfg.threads);

    {
      std::ofstream csv(out_dir / "features.csv");
      csv << "experiment_id,state,query_id,split,knn_dist,centroid_dist,"
             "score_concentration,lid,probe_distance_computations,"
             "lid_probe_distance_computations,knn_dist_lid_probe,"
             "score_concentration_lid_probe\n";
      csv.precision(17);
      for (std::size_t q = 0; q < nq; ++q) {
        const ars::QueryFeatures& f = feats[q];
        csv << exp_id << ",S0," << q << "," << ars::SplitName(split[q]) << ","
            << f.knn_dist << "," << f.centroid_dist << ","
            << f.score_concentration << "," << f.lid << ","
            << f.probe_distance_computations << ","
            << f.lid_probe_distance_computations << "," << f.knn_dist_lid_probe
            << "," << f.score_concentration_lid_probe << "\n";
      }
      if (!csv) {
        throw std::runtime_error("failed writing features.csv");
      }
    }
    {
      std::ofstream c(out_dir / "centroid.csv");
      c.precision(17);
      for (const double x : centroid) {
        c << x << "\n";
      }
    }

    const ars::BuildInfo bi = ars::GetBuildInfo();
    const ars::GitInfo git = ars::GetGitInfo(bi.source_dir);
    const std::string metadata =
        ars::JsonObject()
            .Str("experiment_id", exp_id)
            .Str("experiment_name", cfg.experiment_name)
            .Str("phase", "3_features")
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
                              .Str("note",
                                   "feature extraction is deterministic"
                                   "; no randomness is used")
                              .Render())
            .Raw(
                "features",
                ars::JsonObject()
                    .Str("knn_dist",
                         "Euclidean d_k of probe(k=probe_k, "
                         "ef=probe_ef)")
                    .Str("centroid_dist",
                         "Euclidean ||q - mean(indexed base vectors)||")
                    .Str("score_concentration", "d_1 / d_k of the same probe")
                    .Str("lid",
                         "Levina-Bickel MLE over probe(k=lid_k, "
                         "ef=lid_ef): -1/mean_{i<k} ln(d_i/d_k)")
                    .Int("probe_k",
                         static_cast<std::int64_t>(cfg.params.probe_k))
                    .Int("probe_ef",
                         static_cast<std::int64_t>(cfg.params.probe_ef))
                    .Int("lid_k", static_cast<std::int64_t>(cfg.params.lid_k))
                    .Int("lid_ef", static_cast<std::int64_t>(cfg.params.lid_ef))
                    .Str("centroid_fnv1a64",
                         ars::Hex64(ars::Fnv1a64(centroid_text.str())))
                    .Int("threads", cfg.threads)
                    .Render())
            .Raw("split", ars::JsonObject()
                              .Str("path", cfg.split_path)
                              .Str("use", "row tagging only")
                              .Render())
            .Raw("index", ars::JsonObject()
                              .Str("cache_path", s0.cache_path)
                              .Bool("from_cache", s0.from_cache)
                              .Render())
            .Int("omp_max_threads", omp_get_max_threads())
            .Render();
    std::ofstream(out_dir / "metadata.json") << metadata << "\n";
    std::cerr << "[ars] wrote features for " << nq << " queries to " << out_dir
              << "\n";
    std::cout << out_dir.string() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "[ars] error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
