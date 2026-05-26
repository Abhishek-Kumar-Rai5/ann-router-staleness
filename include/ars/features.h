#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ars/hnsw_index.h"
#include "ars/vecs_io.h"

namespace ars {

// Live, query-time routing features (design doc §9). Every feature is computed
// from the query and the current index only — a shallow hnswlib probe search
// plus the centroid of the indexed vectors. No ground truth, oracle label or
// query-set statistic is used, so features cannot leak the answer.
//
// All distances below are Euclidean (sqrt of hnswlib's squared L2), sorted
// ascending: d_1 <= ... <= d_k from a probe search with (k, ef).

struct FeatureParams {
  std::size_t probe_k = 0;  // core probe: knn_dist, score_concentration
  std::size_t probe_ef = 0;
  std::size_t lid_k = 0;  // separate, deeper probe for the LID ablation
  std::size_t lid_ef = 0;
};

struct QueryFeatures {
  double knn_dist = 0.0;             // d_k of the core probe
  double centroid_dist = 0.0;        // ||q - centroid||
  double score_concentration = 0.0;  // d_1 / d_k of the core probe
  double lid = 0.0;                  // MLE LID over the lid probe (ablation)
  // LID-ablation router only: the core features recomputed from the first
  // probe_k results of the LID probe (that router pays the LID probe instead
  // of the core probe). Not used by the primary router.
  double knn_dist_lid_probe = 0.0;
  double score_concentration_lid_probe = 0.0;
  std::uint64_t probe_distance_computations = 0;
  std::uint64_t lid_probe_distance_computations = 0;
};

// Mean of all rows, accumulated in double.
std::vector<double> ComputeCentroid(const FloatMatrix& base);
// Centroid of the rows with excluded[row] == 0 (empty mask = all rows).
std::vector<double> ComputeCentroid(const FloatMatrix& base,
                                    const std::vector<std::uint8_t>& excluded);

// Euclidean distance between a float vector and a double centroid.
double CentroidDistance(const float* q, const std::vector<double>& centroid);

// Pure functions on ascending squared-L2 distances (as returned by search).
// Each throws if fewer than k distances are given.
//   KnnDistance        = d_k.
//   ScoreConcentration = d_1 / d_k; defined as 1 when d_k == 0 (all k
//                        candidates coincide with the query).
//   LidMle             = -1 / mean_{i=1..k-1} ln(d_i / d_k) (Levina-Bickel
//                        MLE; the i = k term is identically 0 and excluded).
//                        Conventions: 0 if d_k == 0 or any d_i == 0 (zero
//                        distances drive the log-sum to -inf); +inf if all
//                        d_i are equal (log-sum 0). Both are reported by the
//                        validation rather than silently clipped.
double KnnDistance(const std::vector<float>& sq_dists, std::size_t k);
double ScoreConcentration(const std::vector<float>& sq_dists, std::size_t k);
double LidMle(const std::vector<float>& sq_dists, std::size_t k);

// Features for every query row, parallel over queries. Deterministic and
// independent of num_threads (SearchBatch guarantees per-query identity).
std::vector<QueryFeatures> ExtractFeatures(HnswIndex& index,
                                           const FloatMatrix& queries,
                                           const std::vector<double>& centroid,
                                           const FeatureParams& p,
                                           int num_threads);

}  // namespace ars
