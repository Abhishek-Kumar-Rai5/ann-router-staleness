#pragma once

#include <cstddef>
#include <vector>

namespace ars {

std::vector<std::size_t> GeometricEfGrid(std::size_t min, std::size_t max,
                                         double ratio);

struct OracleLabel {
  bool reached = false;
  // Smallest grid ef where recall meets the target and stays there at every
  // larger ef. first_reach_index ignores the larger ones; the two only differ
  // when recall isn't monotone in ef.
  std::size_t index = 0;
  std::size_t first_reach_index = 0;
};

// Recall values are multiples of 1/k, so the target check allows a 1e-9 slack.
OracleLabel DeriveOracleLabel(const std::vector<double>& recall_by_ef,
                              double target);

}  // namespace ars
