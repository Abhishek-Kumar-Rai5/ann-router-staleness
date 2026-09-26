#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ars/vecs_io.h"

namespace ars {

// Neighbours are sorted by (distance, id), so ties always go to the smaller
// id and the output doesn't depend on thread count.
struct KnnResult {
  IdMatrix ids;
  FloatMatrix dists;
};

// Written separately from hnswlib's distance code on purpose, so the ground
// truth doesn't share a possible bug with the index it is checking.
float SquaredL2(const float* a, const float* b, std::size_t dim);

// Recompute this for every index state; never reuse an older state's result.
KnnResult BruteForceKnn(const FloatMatrix& base, const FloatMatrix& queries,
                        std::size_t k);

KnnResult BruteForceKnnExcluding(const FloatMatrix& base,
                                 const FloatMatrix& queries, std::size_t k,
                                 const std::vector<std::uint8_t>& excluded);

}  // namespace ars
