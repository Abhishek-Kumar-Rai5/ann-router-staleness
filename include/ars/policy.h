#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ars {

struct PolicyEntry {
  std::size_t query_id = 0;
  std::size_t ef = 0;
};

std::vector<PolicyEntry> ReadPolicyCsv(const std::string& path);

}  // namespace ars
