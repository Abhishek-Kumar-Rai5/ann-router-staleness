#include "ars/ground_truth.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ars {
namespace {

constexpr std::size_t kQueryBlock = 32;
constexpr std::size_t kBaseBlock = 2048;

using Candidate = std::pair<float, std::int32_t>;

}  // namespace

float SquaredL2(const float* a, const float* b, std::size_t dim) {
  // Separate accumulators let the compiler vectorise without -ffast-math. On
  // integer data like SIFT every partial sum is exact, so the result is too.
  constexpr std::size_t kLanes = 16;
  std::array<float, kLanes> acc{};
  std::size_t d = 0;
  for (; d + kLanes <= dim; d += kLanes) {
    for (std::size_t l = 0; l < kLanes; ++l) {
      const float diff = a[d + l] - b[d + l];
      acc[l] += diff * diff;
    }
  }
  float sum = 0.0F;
  for (; d < dim; ++d) {
    const float diff = a[d] - b[d];
    sum += diff * diff;
  }
  for (const float lane : acc) {
    sum += lane;
  }
  return sum;
}

KnnResult BruteForceKnn(const FloatMatrix& base, const FloatMatrix& queries,
                        std::size_t k) {
  return BruteForceKnnExcluding(base, queries, k, {});
}

KnnResult BruteForceKnnExcluding(const FloatMatrix& base,
                                 const FloatMatrix& queries, std::size_t k,
                                 const std::vector<std::uint8_t>& excluded) {
  if (base.dim != queries.dim) {
    throw std::invalid_argument("BruteForceKnn: dim mismatch");
  }
  if (!excluded.empty() && excluded.size() != base.rows) {
    throw std::invalid_argument("BruteForceKnn: exclusion mask size mismatch");
  }
  std::size_t live = base.rows;
  for (const std::uint8_t e : excluded) {
    live -= e != 0 ? 1 : 0;
  }
  if (k == 0 || k > live) {
    throw std::invalid_argument("BruteForceKnn: need 0 < k <= live rows");
  }
  if (base.rows > static_cast<std::size_t>(INT32_MAX)) {
    throw std::invalid_argument("BruteForceKnn: base too large for int32 ids");
  }

  KnnResult out;
  out.ids.rows = out.dists.rows = queries.rows;
  out.ids.dim = out.dists.dim = k;
  out.ids.data.resize(queries.rows * k);
  out.dists.data.resize(queries.rows * k);

  const std::size_t dim = base.dim;
  const auto num_blocks =
      static_cast<std::int64_t>((queries.rows + kQueryBlock - 1) / kQueryBlock);

#pragma omp parallel for schedule(dynamic, 1)
  for (std::int64_t qb = 0; qb < num_blocks; ++qb) {
    const std::size_t q_begin = static_cast<std::size_t>(qb) * kQueryBlock;
    const std::size_t q_end = std::min(q_begin + kQueryBlock, queries.rows);

    // Comparing (distance, id) pairs keeps the result independent of the order
    // base vectors are visited in.
    std::vector<std::priority_queue<Candidate>> heaps(q_end - q_begin);

    for (std::size_t b_begin = 0; b_begin < base.rows; b_begin += kBaseBlock) {
      const std::size_t b_end = std::min(b_begin + kBaseBlock, base.rows);
      for (std::size_t q = q_begin; q < q_end; ++q) {
        auto& heap = heaps[q - q_begin];
        const float* qv = queries.Row(q);
        for (std::size_t b = b_begin; b < b_end; ++b) {
          if (!excluded.empty() && excluded[b] != 0) {
            continue;
          }
          const Candidate c{SquaredL2(qv, base.Row(b), dim),
                            static_cast<std::int32_t>(b)};
          if (heap.size() < k) {
            heap.push(c);
          } else if (c < heap.top()) {
            heap.pop();
            heap.push(c);
          }
        }
      }
    }

    for (std::size_t q = q_begin; q < q_end; ++q) {
      auto& heap = heaps[q - q_begin];
      for (std::size_t r = k; r-- > 0;) {
        out.ids.Row(q)[r] = heap.top().second;
        out.dists.Row(q)[r] = heap.top().first;
        heap.pop();
      }
    }
  }
  return out;
}

}  // namespace ars
