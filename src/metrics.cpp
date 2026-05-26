#include "ars/metrics.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace ars {

double RecallAtK(const std::int32_t* gt_ids,
                 const std::vector<std::size_t>& found, std::size_t k) {
  if (k == 0) {
    throw std::invalid_argument("RecallAtK: k must be > 0");
  }
  const std::unordered_set<std::int32_t> truth(gt_ids, gt_ids + k);
  const std::size_t n = std::min(k, found.size());
  std::size_t hits = 0;
  for (std::size_t i = 0; i < n; ++i) {
    hits += truth.count(static_cast<std::int32_t>(found[i]));
  }
  return static_cast<double>(hits) / static_cast<double>(k);
}

double TieAwareRecallAtK(const float* gt_dists,
                         const std::vector<float>& found_dists, std::size_t k) {
  if (k == 0) {
    throw std::invalid_argument("TieAwareRecallAtK: k must be > 0");
  }
  constexpr float kRelTol = 1e-6F;
  const float threshold = gt_dists[k - 1] * (1.0F + kRelTol);
  const std::size_t n = std::min(k, found_dists.size());
  std::size_t hits = 0;
  for (std::size_t i = 0; i < n; ++i) {
    hits += found_dists[i] <= threshold ? 1 : 0;
  }
  return static_cast<double>(hits) / static_cast<double>(k);
}

}  // namespace ars
