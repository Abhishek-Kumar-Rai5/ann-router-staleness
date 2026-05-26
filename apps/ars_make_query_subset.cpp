// Phase 4b: draw the fixed confirmation query set from a vector file.
//
// Selection = the same algorithm as the router train/test split
// (ars::MakeQuerySplit: Fisher-Yates over source rows 0..n-1 with
// std::mt19937_64(seed) and rejection-sampled bounded integers; the first
// n_select permutation positions are selected). Selected rows are written in
// ascending source-row order. Depends only on (source row count, n_select,
// seed) — never on any search result or label.
//
// Writes: the subset .fvecs, an id map (row,source_row), an all-"test" tagging
// file for evaluation-only runs, and metadata.json.
//
// Usage: ars_make_query_subset <config.yaml>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "ars/build_info.h"
#include "ars/config.h"
#include "ars/query_split.h"
#include "ars/run_metadata.h"
#include "ars/vecs_io.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <config.yaml>\n";
    return 2;
  }
  try {
    const ars::QuerySubsetConfig cfg = ars::LoadQuerySubsetConfig(argv[1]);
    const std::string exp_id = ars::MakeExperimentId(cfg.raw_yaml);
    const fs::path out_dir = fs::path(cfg.output_dir) / exp_id;
    fs::create_directories(out_dir);
    std::ofstream(out_dir / "config.yaml") << cfg.raw_yaml;

    for (const std::string& p :
         {cfg.output_fvecs, cfg.output_ids, cfg.output_tagging}) {
      if (fs::exists(p)) {
        throw std::runtime_error("refusing to overwrite existing " + p);
      }
    }
    const ars::FloatMatrix src = ars::ReadFvecs(cfg.source_path);
    const auto key = [](const ars::FloatMatrix& m, std::size_t r) {
      return std::string(reinterpret_cast<const char*>(m.Row(r)),
                         m.dim * sizeof(float));
    };
    // 1. Eligible population (ascending source row): exclude byte-exact copies
    //    of any vector in the exclusion files; optionally keep each distinct
    //    vector once (lowest source row).
    std::unordered_set<std::string> excluded;
    for (const std::string& p : cfg.exclude_exact_duplicates_of) {
      const ars::FloatMatrix ex = ars::ReadFvecs(p);
      for (std::size_t r = 0; r < ex.rows; ++r) {
        excluded.insert(key(ex, r));
      }
    }
    std::vector<std::size_t> population;
    std::unordered_set<std::string> seen;
    std::size_t n_excluded = 0;
    std::size_t n_dedup = 0;
    for (std::size_t r = 0; r < src.rows; ++r) {
      const std::string k = key(src, r);
      if (excluded.count(k) != 0) {
        ++n_excluded;
        continue;
      }
      if (cfg.deduplicate_population && !seen.insert(k).second) {
        ++n_dedup;
        continue;
      }
      population.push_back(r);
    }
    if (population.size() < cfg.n_select) {
      throw std::runtime_error("eligible population smaller than n_select");
    }
    // 2. Seeded Fisher-Yates over population POSITIONS 0..N-1.
    const std::vector<ars::Split> pick =
        ars::MakeQuerySplit(population.size(), cfg.n_select, cfg.seed);

    ars::FloatMatrix out{{}, 0, src.dim};
    std::string ids = "row,source_row,population_position\n";
    std::string pop_ids;
    for (std::size_t p = 0; p < population.size(); ++p) {
      pop_ids += std::to_string(population[p]) + "\n";
      if (pick[p] != ars::Split::kTest) {
        continue;
      }
      const std::size_t r = population[p];
      ids += std::to_string(out.rows) + "," + std::to_string(r) + "," +
             std::to_string(p) + "\n";
      out.data.insert(out.data.end(), src.Row(r), src.Row(r) + src.dim);
      ++out.rows;
    }
    if (out.rows != cfg.n_select) {
      throw std::runtime_error("selected row count mismatch");
    }
    for (const std::string& p :
         {cfg.output_fvecs, cfg.output_ids, cfg.output_tagging}) {
      fs::create_directories(fs::path(p).parent_path());
    }
    ars::WriteFvecs(cfg.output_fvecs, out);
    std::ofstream(cfg.output_ids) << ids;
    const std::vector<ars::Split> tags(out.rows, ars::Split::kTest);
    std::ofstream(cfg.output_tagging) << ars::SplitToCsv(tags);

    const ars::BuildInfo bi = ars::GetBuildInfo();
    const ars::GitInfo git = ars::GetGitInfo(bi.source_dir);
    std::ofstream(out_dir / "metadata.json")
        << ars::JsonObject()
               .Str("experiment_id", exp_id)
               .Str("phase", "4b_confirmation_set")
               .Str("config_yaml", cfg.raw_yaml)
               .Raw("git", ars::JsonObject()
                               .Str("commit", git.commit)
                               .Bool("dirty", git.dirty)
                               .Render())
               .Str("method",
                    "eligible population = source rows (ascending) minus "
                    "byte-exact copies of exclusion files, optionally one "
                    "row per distinct vector (lowest row); then "
                    "ars::MakeQuerySplit(population_size, n_select, seed) over "
                    "population positions (Fisher-Yates, std::mt19937_64, "
                    "rejection-sampled bounded ints); first n_select positions "
                    "selected; output in ascending source-row order")
               .Int("source_rows", static_cast<std::int64_t>(src.rows))
               .Int("excluded_exact_duplicates",
                    static_cast<std::int64_t>(n_excluded))
               .Int("removed_population_duplicates",
                    static_cast<std::int64_t>(n_dedup))
               .Int("eligible_population",
                    static_cast<std::int64_t>(population.size()))
               .Str("eligible_population_fnv1a64",
                    ars::Hex64(ars::Fnv1a64(pop_ids)))
               .Int("n_selected", static_cast<std::int64_t>(out.rows))
               .Int("seed", static_cast<std::int64_t>(cfg.seed))
               .Str("query_set_id", cfg.query_set_id)
               .Str("ids_fnv1a64", ars::Hex64(ars::Fnv1a64(ids)))
               .Render()
        << "\n";
    std::cerr << "[ars] selected " << out.rows << " of " << src.rows
              << " rows -> " << cfg.output_fvecs << "\n";
    std::cout << out_dir.string() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "[ars] error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
