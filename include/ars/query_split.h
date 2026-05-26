#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace ars {

// Router train/test split of the fixed query set (design doc §7.3, §13). It
// depends only on (num_queries, test_size, seed) — never on any search or
// label — so no outcome can influence which queries end up in which split.
enum class Split : std::uint8_t { kTrain = 0, kTest = 1 };

const char* SplitName(Split s);

// Unbiased integer in [0, bound) from a raw mt19937_64 stream via rejection
// sampling. Used instead of std::uniform_int_distribution (whose algorithm is
// implementation-defined), so the split is identical on every platform; the
// mt19937_64 output sequence itself is fixed by the C++ standard.
std::uint64_t UniformBelow(std::mt19937_64& rng, std::uint64_t bound);

// Fisher-Yates shuffle of 0..n-1 (i from n-1 down to 1, j = UniformBelow(i+1))
// seeded with `seed`. Query ids at the first `test_size` positions of the
// permutation are kTest; all others are kTrain. Returns one entry per query.
std::vector<Split> MakeQuerySplit(std::size_t num_queries,
                                  std::size_t test_size, std::uint64_t seed);

// CSV with header "query_id,split" and one row per query in id order.
std::string SplitToCsv(const std::vector<Split>& split);
std::vector<Split> ReadSplitCsv(const std::string& path);

}  // namespace ars
