#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ars {

// A per-query search-effort policy: the efSearch assigned to each query.
struct PolicyEntry {
  std::size_t query_id = 0;
  std::size_t ef = 0;
};

// Reads a CSV with header "query_id,ef" (any row order, each query at most
// once, ef > 0). Throws on malformed input or duplicate query ids.
std::vector<PolicyEntry> ReadPolicyCsv(const std::string& path);

}  // namespace ars
