#include "ars/build_info.h"

#include <omp.h>

namespace ars {

BuildInfo GetBuildInfo() {
  return BuildInfo{
      .compiler_id = ARS_COMPILER_ID,
      .compiler_version = ARS_COMPILER_VERSION,
      .build_type = ARS_BUILD_TYPE,
      .cxx_flags = ARS_CXX_FLAGS,
      .cxx_standard = __cplusplus,
      .openmp_version = _OPENMP,
      .hnswlib_commit = ARS_HNSWLIB_COMMIT,
      .source_dir = ARS_SOURCE_DIR,
  };
}

}  // namespace ars
