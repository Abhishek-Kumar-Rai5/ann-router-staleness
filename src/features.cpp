#include "ars/features.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ars {
namespace {

void RequireAtLeast(const std::vector<float>& sq_dists, std::size_t k,
                    const char* who) {
  if (k == 0 || sq_dists.size() < k) {
    throw std::invalid_argument(std::string(who) +
                                ": need 0 < k <= number of distances");
  }
}

}  // namespace

std::vector<double> ComputeCentroid(const FloatMatrix& base) {
  return ComputeCentroid(base, {});
}

std::vector<double> ComputeCentroid(const FloatMatrix& base,
                                    const std::vector<std::uint8_t>& excluded) {
  if (!excluded.empty() && excluded.size() != base.rows) {
    throw std::invalid_argument("ComputeCentroid: mask size mismatch");
  }
  std::size_t n = 0;
  std::vector<double> c(base.dim, 0.0);
  for (std::size_t i = 0; i < base.rows; ++i) {
    if (!excluded.empty() && excluded[i] != 0) {
      continue;
    }
    ++n;
    const float* row = base.Row(i);
    for (std::size_t d = 0; d < base.dim; ++d) {
      c[d] += row[d];
    }
  }
  if (n == 0) {
    throw std::invalid_argument("ComputeCentroid: empty base");
  }
  for (double& x : c) {
    x /= static_cast<double>(n);
  }
  return c;
}

double CentroidDistance(const float* q, const std::vector<double>& centroid) {
  double s = 0.0;
  for (std::size_t d = 0; d < centroid.size(); ++d) {
    const double diff = static_cast<double>(q[d]) - centroid[d];
    s += diff * diff;
  }
  return std::sqrt(s);
}

double KnnDistance(const std::vector<float>& sq_dists, std::size_t k) {
  RequireAtLeast(sq_dists, k, "KnnDistance");
  return std::sqrt(static_cast<double>(sq_dists[k - 1]));
}

double ScoreConcentration(const std::vector<float>& sq_dists, std::size_t k) {
  RequireAtLeast(sq_dists, k, "ScoreConcentration");
  const double dk = std::sqrt(static_cast<double>(sq_dists[k - 1]));
  if (dk == 0.0) {
    return 1.0;
  }
  return std::sqrt(static_cast<double>(sq_dists[0])) / dk;
}

double LidMle(const std::vector<float>& sq_dists, std::size_t k) {
  RequireAtLeast(sq_dists, k, "LidMle");
  if (k < 2) {
    throw std::invalid_argument("LidMle: need k >= 2");
  }
  const double dk = std::sqrt(static_cast<double>(sq_dists[k - 1]));
  if (dk == 0.0) {
    return 0.0;
  }
  double log_sum = 0.0;
  for (std::size_t i = 0; i + 1 < k; ++i) {
    const double di = std::sqrt(static_cast<double>(sq_dists[i]));
    if (di == 0.0) {
      return 0.0;
    }
    log_sum += std::log(di / dk);
  }
  if (log_sum == 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return -static_cast<double>(k - 1) / log_sum;
}

std::vector<QueryFeatures> ExtractFeatures(HnswIndex& index,
                                           const FloatMatrix& queries,
                                           const std::vector<double>& centroid,
                                           const FeatureParams& p,
                                           int num_threads) {
  if (centroid.size() != queries.dim) {
    throw std::invalid_argument("ExtractFeatures: centroid dim mismatch");
  }
  const auto probe =
      index.SearchBatch(queries, p.probe_k, p.probe_ef, num_threads);
  const auto lid_probe =
      index.SearchBatch(queries, p.lid_k, p.lid_ef, num_threads);
  std::vector<QueryFeatures> out(queries.rows);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    QueryFeatures& f = out[q];
    f.knn_dist = KnnDistance(probe[q].dists, p.probe_k);
    f.score_concentration = ScoreConcentration(probe[q].dists, p.probe_k);
    f.centroid_dist = CentroidDistance(queries.Row(q), centroid);
    f.lid = LidMle(lid_probe[q].dists, p.lid_k);
    if (p.lid_k >= p.probe_k) {
      f.knn_dist_lid_probe = KnnDistance(lid_probe[q].dists, p.probe_k);
      f.score_concentration_lid_probe =
          ScoreConcentration(lid_probe[q].dists, p.probe_k);
    }
    f.probe_distance_computations = probe[q].distance_computations;
    f.lid_probe_distance_computations = lid_probe[q].distance_computations;
  }
  return out;
}

}  // namespace ars
