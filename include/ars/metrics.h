#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ars {

double RecallAtK(const std::int32_t* gt_ids,
                 const std::vector<std::size_t>& found, std::size_t k);

// Unlike RecallAtK, this doesn't penalise returning a different one of
// several equally distant (e.g. duplicate) vectors.
double TieAwareRecallAtK(const float* gt_dists,
                         const std::vector<float>& found_dists, std::size_t k);

}  // namespace ars
