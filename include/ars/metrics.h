#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ars {

// Recall@k by id: fraction of the first k ground-truth ids that appear among
// the first k found labels. gt_ids must hold at least k entries.
double RecallAtK(const std::int32_t* gt_ids,
                 const std::vector<std::size_t>& found, std::size_t k);

// Distance-based recall@k (ann-benchmarks definition): fraction of the first
// k found results whose distance is <= the k-th ground-truth distance (with a
// small relative tolerance). Unlike RecallAtK it does not penalise returning a
// different member of a set of equidistant (e.g. duplicate) vectors.
double TieAwareRecallAtK(const float* gt_dists,
                         const std::vector<float>& found_dists, std::size_t k);

}  // namespace ars
