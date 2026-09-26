#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ars/hnsw_index.h"
#include "ars/vecs_io.h"

namespace ars {

// Routing features only look at the query and the live index, never at
// ground truth, so they can't leak the answer.
struct FeatureParams {
  std::size_t probe_k = 0;
  std::size_t probe_ef = 0;
  std::size_t lid_k = 0;
  std::size_t lid_ef = 0;
};

struct QueryFeatures {
  double knn_dist = 0.0;
  double centroid_dist = 0.0;
  double score_concentration = 0.0;
  double lid = 0.0;
  double knn_dist_lid_probe = 0.0;
  double score_concentration_lid_probe = 0.0;
  std::uint64_t probe_distance_computations = 0;
  std::uint64_t lid_probe_distance_computations = 0;
};

std::vector<double> ComputeCentroid(const FloatMatrix& base);
std::vector<double> ComputeCentroid(const FloatMatrix& base,
                                    const std::vector<std::uint8_t>& excluded);

double CentroidDistance(const float* q, const std::vector<double>& centroid);

double KnnDistance(const std::vector<float>& sq_dists, std::size_t k);
double ScoreConcentration(const std::vector<float>& sq_dists, std::size_t k);
// LidMle returns 0 if any distance is zero and +inf if all distances are
// equal. We report these cases instead of clipping them.
double LidMle(const std::vector<float>& sq_dists, std::size_t k);

std::vector<QueryFeatures> ExtractFeatures(HnswIndex& index,
                                           const FloatMatrix& queries,
                                           const std::vector<double>& centroid,
                                           const FeatureParams& p,
                                           int num_threads);

}  // namespace ars
