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

// Shared S0 setup used by every phase's driver: load the configured slice of
// the dataset, then load-or-compute the S0 ground truth and load-or-build the
// S0 index from data/cache. Cache keys depend only on S0Config fields, so all
// phases with the same dataset/index/ground_truth sections share one cache.

struct S0Data {
  FloatMatrix base;  // all labelled vectors (S0 base, then inserted ones)
  FloatMatrix queries;
  // excluded[label] = 1 for lazily deleted labels (empty = none deleted).
  std::vector<std::uint8_t> excluded;
};
S0Data LoadS0Data(const S0Config& cfg);

struct S0GroundTruth {
  KnnResult knn;
  std::string cache_path;  // prefix; .ivecs (ids) and .fvecs (distances)
  bool from_cache = false;
  double seconds = 0.0;
};
// Cached for S0 only. Evolved states must never reuse this (CLAUDE.md: ground
// truth is recomputed for every index state).
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
