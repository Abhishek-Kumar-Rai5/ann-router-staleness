#include "ars/policy.h"

#include <fstream>
#include <set>
#include <stdexcept>

namespace ars {

std::vector<PolicyEntry> ReadPolicyCsv(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open policy file " + path);
  }
  std::string line;
  if (!std::getline(in, line) || line != "query_id,ef") {
    throw std::runtime_error("bad policy header in " + path);
  }
  std::vector<PolicyEntry> out;
  std::set<std::size_t> seen;
  while (std::getline(in, line)) {
    const auto comma = line.find(',');
    if (comma == std::string::npos) {
      std::string msg = "bad policy row '";
      msg += line;
      msg += "' in ";
      msg += path;
      throw std::runtime_error(msg);
    }
    const PolicyEntry e{std::stoul(line.substr(0, comma)),
                        std::stoul(line.substr(comma + 1))};
    if (e.ef == 0 || !seen.insert(e.query_id).second) {
      throw std::runtime_error("zero ef or duplicate query in " + path);
    }
    out.push_back(e);
  }
  return out;
}

}  // namespace ars
