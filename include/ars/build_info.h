#pragma once

#include <string>

namespace ars {

// Compile-time build metadata, logged into every run's metadata file
// (CLAUDE.md reproducibility rules).
struct BuildInfo {
  std::string compiler_id;
  std::string compiler_version;
  std::string build_type;
  std::string cxx_flags;
  long cxx_standard;
  int openmp_version;
  std::string hnswlib_commit;  // pinned submodule commit, set at configure time
  std::string source_dir;      // repo root, for runtime git commit lookup
};

BuildInfo GetBuildInfo();

}  // namespace ars
