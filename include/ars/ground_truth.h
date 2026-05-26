#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ars/vecs_io.h"

namespace ars {

// Exact k-nearest neighbours under squared L2. Row i of `ids`/`dists` holds
// query i's neighbours in ascending (distance, id) order, so ties are broken
// deterministically by the smaller base id regardless of thread count.
struct KnnResult {
  IdMatrix ids;
  FloatMatrix dists;
};

// Squared L2 distance. Deliberately independent of hnswlib's distance code so
// that ground truth does not share an implementation with the index under
// test.
float SquaredL2(const float* a, const float* b, std::size_t dim);

// Brute-force exact k-NN of every query against every base vector,
// parallelised over queries with OpenMP. Requires k <= base.rows and matching
// dims; throws otherwise. Must be recomputed for every index state.
KnnResult BruteForceKnn(const FloatMatrix& base, const FloatMatrix& queries,
                        std::size_t k);

// As above, but rows with excluded[row] != 0 (lazily deleted vectors of an
// evolved index state) are never returned. An empty mask excludes nothing and
// gives results identical to BruteForceKnn. Requires k <= live rows.
KnnResult BruteForceKnnExcluding(const FloatMatrix& base,
                                 const FloatMatrix& queries, std::size_t k,
                                 const std::vector<std::uint8_t>& excluded);

}  // namespace ars
