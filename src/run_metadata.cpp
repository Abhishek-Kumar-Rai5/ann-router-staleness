#include "ars/run_metadata.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <memory>
#include <sstream>

namespace ars {
namespace {

std::string RunCommand(const std::string& cmd) {
  std::array<char, 256> buf{};
  std::string out;
  std::unique_ptr<FILE, int (*)(FILE*)> pipe(popen(cmd.c_str(), "r"), pclose);
  if (!pipe) {
    return out;
  }
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe.get()) !=
         nullptr) {
    out += buf.data();
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
  return out;
}

}  // namespace

std::uint64_t Fnv1a64(const std::string& s) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

std::string Hex64(std::uint64_t v) {
  std::array<char, 17> buf{};
  std::snprintf(buf.data(), buf.size(), "%016llx",
                static_cast<unsigned long long>(v));
  return buf.data();
}

std::string MakeExperimentId(const std::string& config_text) {
  const std::time_t now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm utc{};
  gmtime_r(&now, &utc);
  std::array<char, 32> ts{};
  std::strftime(ts.data(), ts.size(), "%Y%m%dT%H%M%SZ", &utc);
  return Hex64(Fnv1a64(config_text)).substr(0, 12) + "_" + ts.data();
}

GitInfo GetGitInfo(const std::string& repo_dir) {
  GitInfo info;
  const std::string git = "git -C '" + repo_dir + "' ";
  info.commit = RunCommand(git + "rev-parse --verify -q HEAD 2>/dev/null");
  if (info.commit.empty()) {
    info.commit = "none";
  }
  info.dirty = !RunCommand(git + "status --porcelain 2>/dev/null").empty();
  return info;
}

std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (const char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          std::array<char, 8> buf{};
          std::snprintf(buf.data(), buf.size(), "\\u%04x", c);
          out += buf.data();
        } else {
          out += c;
        }
    }
  }
  return out;
}

JsonObject& JsonObject::Str(const std::string& key, const std::string& value) {
  fields_.emplace_back(key, "\"" + JsonEscape(value) + "\"");
  return *this;
}

JsonObject& JsonObject::Num(const std::string& key, double value) {
  std::ostringstream os;
  os.precision(17);
  os << value;
  fields_.emplace_back(key, os.str());
  return *this;
}

JsonObject& JsonObject::Int(const std::string& key, std::int64_t value) {
  fields_.emplace_back(key, std::to_string(value));
  return *this;
}

JsonObject& JsonObject::Bool(const std::string& key, bool value) {
  fields_.emplace_back(key, value ? "true" : "false");
  return *this;
}

JsonObject& JsonObject::Raw(const std::string& key, const std::string& json) {
  fields_.emplace_back(key, json);
  return *this;
}

std::string JsonObject::Render() const {
  std::string out = "{";
  for (std::size_t i = 0; i < fields_.size(); ++i) {
    out += (i == 0 ? "\"" : ", \"") + JsonEscape(fields_[i].first) +
           "\": " + fields_[i].second;
  }
  return out + "}";
}

}  // namespace ars
