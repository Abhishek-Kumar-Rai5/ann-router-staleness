#include "ars/query_split.h"

#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ars {

const char* SplitName(Split s) { return s == Split::kTest ? "test" : "train"; }

std::uint64_t UniformBelow(std::mt19937_64& rng, std::uint64_t bound) {
  if (bound == 0) {
    throw std::invalid_argument("UniformBelow: bound must be > 0");
  }
  const std::uint64_t threshold = (0 - bound) % bound;
  while (true) {
    const std::uint64_t r = rng();
    if (r >= threshold) {
      return r % bound;
    }
  }
}

std::vector<Split> MakeQuerySplit(std::size_t num_queries,
                                  std::size_t test_size, std::uint64_t seed) {
  if (test_size == 0 || test_size >= num_queries) {
    throw std::invalid_argument(
        "MakeQuerySplit: need 0 < test_size < num_queries");
  }
  std::vector<std::size_t> perm(num_queries);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937_64 rng(seed);
  for (std::size_t i = num_queries - 1; i > 0; --i) {
    const auto j = static_cast<std::size_t>(UniformBelow(rng, i + 1));
    std::swap(perm[i], perm[j]);
  }
  std::vector<Split> split(num_queries, Split::kTrain);
  for (std::size_t p = 0; p < test_size; ++p) {
    split[perm[p]] = Split::kTest;
  }
  return split;
}

std::string SplitToCsv(const std::vector<Split>& split) {
  std::string out = "query_id,split\n";
  for (std::size_t q = 0; q < split.size(); ++q) {
    out += std::to_string(q) + "," + SplitName(split[q]) + "\n";
  }
  return out;
}

std::vector<Split> ReadSplitCsv(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open split file " + path);
  }
  std::string line;
  if (!std::getline(in, line) || line != "query_id,split") {
    throw std::runtime_error("bad split header in " + path);
  }
  std::vector<Split> split;
  while (std::getline(in, line)) {
    const auto comma = line.find(',');
    if (comma == std::string::npos ||
        std::stoul(line.substr(0, comma)) != split.size()) {
      throw std::runtime_error("bad or out-of-order split row in " + path);
    }
    const std::string name = line.substr(comma + 1);
    if (name != "train" && name != "test") {
      std::string msg = "unknown split '";
      msg += name;
      msg += "' in ";
      msg += path;
      throw std::runtime_error(msg);
    }
    split.push_back(name == "test" ? Split::kTest : Split::kTrain);
  }
  return split;
}

}  // namespace ars
