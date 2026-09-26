#pragma once

#include <string>

namespace ars {

struct BuildInfo {
  std::string compiler_id;
  std::string compiler_version;
  std::string build_type;
  std::string cxx_flags;
  long cxx_standard;
  int openmp_version;
  std::string hnswlib_commit;
  std::string source_dir;
};

BuildInfo GetBuildInfo();

}  // namespace ars
