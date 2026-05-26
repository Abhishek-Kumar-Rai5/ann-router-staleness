// Supporting modules: .fvecs/.ivecs I/O, recall metrics, YAML config parsing
// and run-metadata helpers.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ars/config.h"
#include "ars/metrics.h"
#include "ars/run_metadata.h"
#include "ars/vecs_io.h"

namespace {

std::string TempPath(const std::string& name) {
  return ::testing::TempDir() + "/" + name;
}

TEST(VecsIo, FvecsAndIvecsRoundTrip) {
  const ars::FloatMatrix f{{1.5F, -2, 3, 4, 5, 6.25F}, 3, 2};
  const ars::IdMatrix i{{7, 8, 9, 10, 11, 12}, 2, 3};
  ars::WriteFvecs(TempPath("rt.fvecs"), f);
  ars::WriteIvecs(TempPath("rt.ivecs"), i);
  const auto f2 = ars::ReadFvecs(TempPath("rt.fvecs"));
  const auto i2 = ars::ReadIvecs(TempPath("rt.ivecs"));
  EXPECT_EQ(f2.rows, 3U);
  EXPECT_EQ(f2.dim, 2U);
  EXPECT_EQ(f2.data, f.data);
  EXPECT_EQ(i2.rows, 2U);
  EXPECT_EQ(i2.dim, 3U);
  EXPECT_EQ(i2.data, i.data);
  // max_rows truncates to a prefix.
  const auto head = ars::ReadFvecs(TempPath("rt.fvecs"), 2);
  EXPECT_EQ(head.rows, 2U);
  EXPECT_EQ(head.data, (std::vector<float>{1.5F, -2, 3, 4}));
}

TEST(VecsIo, RejectsTruncatedFileAndMissingFile) {
  const ars::FloatMatrix f{{1, 2, 3, 4}, 2, 2};
  const std::string path = TempPath("trunc.fvecs");
  ars::WriteFvecs(path, f);
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 2);
  EXPECT_THROW(ars::ReadFvecs(path), std::runtime_error);
  EXPECT_THROW(ars::ReadFvecs(TempPath("does_not_exist.fvecs")),
               std::runtime_error);
}

TEST(Metrics, RecallAtKCountsIdOverlap) {
  const std::vector<std::int32_t> gt = {5, 3, 9, 1};
  EXPECT_DOUBLE_EQ(ars::RecallAtK(gt.data(), {5, 3, 9, 1}, 4), 1.0);
  EXPECT_DOUBLE_EQ(ars::RecallAtK(gt.data(), {1, 9, 3, 5}, 4), 1.0);
  EXPECT_DOUBLE_EQ(ars::RecallAtK(gt.data(), {5, 3, 0, 2}, 4), 0.5);
  EXPECT_DOUBLE_EQ(ars::RecallAtK(gt.data(), {5}, 4), 0.25);  // short result
  // Only the first k of each list count.
  EXPECT_DOUBLE_EQ(ars::RecallAtK(gt.data(), {0, 5, 3}, 2), 0.5);
}

TEST(Metrics, TieAwareRecallAcceptsEquidistantSubstitutes) {
  const std::vector<float> gt = {1.0F, 2.0F, 2.0F};
  EXPECT_DOUBLE_EQ(ars::TieAwareRecallAtK(gt.data(), {1, 2, 2}, 3), 1.0);
  EXPECT_DOUBLE_EQ(ars::TieAwareRecallAtK(gt.data(), {1, 2, 2.5F}, 3),
                   2.0 / 3.0);
  // With k=2 the threshold is gt[1]=2, so a different id at distance 2 counts.
  EXPECT_DOUBLE_EQ(ars::TieAwareRecallAtK(gt.data(), {1, 2}, 2), 1.0);
}

constexpr const char* kValidYaml = R"(experiment_name: t
dataset:
  base: b.fvecs
  queries: q.fvecs
  max_base: 100
  max_queries: 0
index:
  m: 8
  ef_construction: 50
  seed: 3
  build_threads: 1
ground_truth:
  k: 20
search:
  k: 10
  ef_values: [10, 20, 40]
  timing_repeats: 3
  warmup_queries: 5
paths:
  cache_dir: c
  output_dir: o
)";

std::string WriteYaml(const std::string& name, const std::string& text) {
  const std::string path = TempPath(name);
  std::ofstream(path) << text;
  return path;
}

TEST(Config, ParsesAllFields) {
  const auto c = ars::LoadStaticSweepConfig(WriteYaml("ok.yaml", kValidYaml));
  EXPECT_EQ(c.experiment_name, "t");
  EXPECT_EQ(c.max_base, 100U);
  EXPECT_EQ(c.index.m, 8U);
  EXPECT_EQ(c.index.seed, 3U);
  EXPECT_EQ(c.gt_k, 20U);
  EXPECT_EQ(c.ef_values, (std::vector<std::size_t>{10, 20, 40}));
  EXPECT_EQ(c.raw_yaml, kValidYaml);
}

TEST(Config, MissingKeyAndBadValuesThrow) {
  std::string missing = kValidYaml;
  missing.replace(missing.find("  seed: 3\n"), 10, "");
  EXPECT_THROW(ars::LoadStaticSweepConfig(WriteYaml("missing.yaml", missing)),
               std::runtime_error);
  std::string bad_k = kValidYaml;
  bad_k.replace(bad_k.find("  k: 20"), 7, "  k: 5");
  EXPECT_THROW(ars::LoadStaticSweepConfig(WriteYaml("badk.yaml", bad_k)),
               std::runtime_error);
}

TEST(RunMetadata, HashIsStableAndJsonIsEscaped) {
  EXPECT_EQ(ars::Fnv1a64("abc"), ars::Fnv1a64("abc"));
  EXPECT_NE(ars::Fnv1a64("abc"), ars::Fnv1a64("abd"));
  EXPECT_EQ(ars::Hex64(255), "00000000000000ff");
  EXPECT_EQ(ars::JsonEscape("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
  const std::string json =
      ars::JsonObject().Str("s", "x").Int("i", 3).Bool("b", false).Render();
  EXPECT_EQ(json, R"({"s": "x", "i": 3, "b": false})");
}

}  // namespace
