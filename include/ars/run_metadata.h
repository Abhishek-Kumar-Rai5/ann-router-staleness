#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ars {

// 64-bit FNV-1a, used for experiment ids and cache keys (not for security).
std::uint64_t Fnv1a64(const std::string& s);
std::string Hex64(std::uint64_t v);

// "<hash of config text>_<UTC timestamp>", per design doc section 20.
std::string MakeExperimentId(const std::string& config_text);

struct GitInfo {
  std::string commit;  // "none" if the repository has no commits yet
  bool dirty = true;
};
GitInfo GetGitInfo(const std::string& repo_dir);

std::string JsonEscape(const std::string& s);

// Minimal flat-ish JSON object builder: values are inserted pre-rendered, so
// nested objects are built by passing another builder's Render() output.
class JsonObject {
 public:
  JsonObject& Str(const std::string& key, const std::string& value);
  JsonObject& Num(const std::string& key, double value);
  JsonObject& Int(const std::string& key, std::int64_t value);
  JsonObject& Bool(const std::string& key, bool value);
  JsonObject& Raw(const std::string& key, const std::string& json);
  [[nodiscard]] std::string Render() const;

 private:
  std::vector<std::pair<std::string, std::string>> fields_;
};

}  // namespace ars
