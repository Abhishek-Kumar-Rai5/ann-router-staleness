#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ars/config.h"
#include "ars/ground_truth.h"
#include "ars/hnsw_index.h"
#include "ars/vecs_io.h"

namespace ars {

struct S0Data {
  FloatMatrix base;
  FloatMatrix queries;
  std::vector<std::uint8_t> excluded;
};
S0Data LoadS0Data(const S0Config& cfg);

struct S0GroundTruth {
  KnnResult knn;
  std::string cache_path;
  bool from_cache = false;
  double seconds = 0.0;
};
// Cached for S0 only. Evolved states must always compute their own ground
// truth.
S0GroundTruth LoadOrComputeS0GroundTruth(const S0Config& cfg,
                                         const S0Data& data);

struct S0Index {
  HnswIndex index;
  std::string cache_path;
  bool from_cache = false;
  double seconds = 0.0;
};
S0Index LoadOrBuildS0Index(const S0Config& cfg, const S0Data& data);

}  // namespace ars
