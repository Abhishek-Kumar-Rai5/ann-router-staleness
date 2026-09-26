#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace ars {

enum class Split : std::uint8_t { kTrain = 0, kTest = 1 };

const char* SplitName(Split s);

// We avoid std::uniform_int_distribution because its algorithm differs
// between standard libraries; this keeps the split identical everywhere.
std::uint64_t UniformBelow(std::mt19937_64& rng, std::uint64_t bound);

// Depends only on the seed and sizes, never on search results, so no
// outcome can leak into which queries land in train or test.
std::vector<Split> MakeQuerySplit(std::size_t num_queries,
                                  std::size_t test_size, std::uint64_t seed);

std::string SplitToCsv(const std::vector<Split>& split);
std::vector<Split> ReadSplitCsv(const std::string& path);

}  // namespace ars
