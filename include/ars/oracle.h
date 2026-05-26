#pragma once

#include <cstddef>
#include <vector>

namespace ars {

// Oracle search effort (design doc §5.3, §7.7): per query, the smallest
// efSearch reaching the target recall, found by direct search against the
// state's own ground truth over a fixed grid of ef values.

// Increasing ef grid: start at `min`, next = max(prev + 1, round(prev *
// ratio)), stop at `max` (always included). Integer steps at small ef where
// ratio * ef < ef + 1, geometric steps above.
std::vector<std::size_t> GeometricEfGrid(std::size_t min, std::size_t max,
                                         double ratio);

struct OracleLabel {
  // False if recall at the largest grid ef is below target (censored label).
  bool reached = false;
  // Grid index of the oracle ef: the smallest i such that recall >= target at
  // grid[i] AND at every larger grid ef ("stably reaches"). Valid if reached.
  std::size_t index = 0;
  // Smallest i with recall >= target at grid[i], ignoring larger ef. Differs
  // from `index` only where recall is non-monotone in ef. Valid if reached.
  std::size_t first_reach_index = 0;
};

// recall_by_ef[i] is the query's recall at grid[i]. Recall values are
// multiples of 1/k, so they are compared with a 1e-9 tolerance below target.
OracleLabel DeriveOracleLabel(const std::vector<double>& recall_by_ef,
                              double target);

}  // namespace ars
