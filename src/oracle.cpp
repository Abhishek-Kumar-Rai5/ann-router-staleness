#include "ars/oracle.h"

#include <cmath>
#include <stdexcept>

namespace ars {

std::vector<std::size_t> GeometricEfGrid(std::size_t min, std::size_t max,
                                         double ratio) {
  if (min == 0 || max < min || !(ratio > 1.0)) {
    throw std::invalid_argument(
        "GeometricEfGrid: need 0 < min <= max and ratio > 1");
  }
  std::vector<std::size_t> grid{min};
  while (grid.back() < max) {
    const std::size_t prev = grid.back();
    const auto scaled = static_cast<std::size_t>(
        std::llround(static_cast<double>(prev) * ratio));
    std::size_t next = scaled > prev + 1 ? scaled : prev + 1;
    if (next > max) {
      next = max;
    }
    grid.push_back(next);
  }
  return grid;
}

OracleLabel DeriveOracleLabel(const std::vector<double>& recall_by_ef,
                              double target) {
  if (recall_by_ef.empty()) {
    throw std::invalid_argument("DeriveOracleLabel: empty recall curve");
  }
  constexpr double kTol = 1e-9;
  const auto meets = [&](double r) { return r >= target - kTol; };

  OracleLabel label;
  const std::size_t n = recall_by_ef.size();
  // Never reached even at the largest ef: the label is censored.
  if (!meets(recall_by_ef[n - 1])) {
    return label;
  }
  label.reached = true;
  std::size_t i = n - 1;
  while (i > 0 && meets(recall_by_ef[i - 1])) {
    --i;
  }
  label.index = i;
  label.first_reach_index = 0;
  while (!meets(recall_by_ef[label.first_reach_index])) {
    ++label.first_reach_index;
  }
  return label;
}

}  // namespace ars
