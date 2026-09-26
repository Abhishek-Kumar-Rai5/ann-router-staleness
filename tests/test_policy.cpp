#include <gtest/gtest.h>

#include <fstream>
#include <stdexcept>
#include <string>

#include "ars/config.h"
#include "ars/policy.h"

namespace {

std::string Write(const std::string& name, const std::string& text) {
  const std::string path = ::testing::TempDir() + "/" + name;
  std::ofstream(path) << text;
  return path;
}

TEST(Policy, ReadsRowsInAnyOrder) {
  const auto p =
      ars::ReadPolicyCsv(Write("p.csv", "query_id,ef\n7,48\n2,18\n"));
  ASSERT_EQ(p.size(), 2U);
  EXPECT_EQ(p[0].query_id, 7U);
  EXPECT_EQ(p[0].ef, 48U);
  EXPECT_EQ(p[1].query_id, 2U);
  EXPECT_EQ(p[1].ef, 18U);
}

TEST(Policy, RejectsBadHeaderDuplicatesZeroEfAndMissingFile) {
  EXPECT_THROW(ars::ReadPolicyCsv(Write("h.csv", "q,ef\n1,2\n")),
               std::runtime_error);
  EXPECT_THROW(ars::ReadPolicyCsv(Write("d.csv", "query_id,ef\n1,18\n1,48\n")),
               std::runtime_error);
  EXPECT_THROW(ars::ReadPolicyCsv(Write("z.csv", "query_id,ef\n1,0\n")),
               std::runtime_error);
  EXPECT_THROW(ars::ReadPolicyCsv(Write("n.csv", "query_id,ef\n1\n")),
               std::runtime_error);
  EXPECT_THROW(ars::ReadPolicyCsv(::testing::TempDir() + "/missing.csv"),
               std::runtime_error);
}

TEST(PolicyEvalConfig, ParsesAndValidates) {
  const std::string text = R"(experiment_name: v
dataset: {base: b, queries: q, max_base: 0, max_queries: 0}
index: {m: 16, ef_construction: 200, seed: 42, build_threads: 1}
ground_truth: {k: 100}
paths: {cache_dir: c, output_dir: o}
policy: {path: p.csv, k: 10, threads: 8}
)";
  const auto c = ars::LoadPolicyEvalConfig(Write("v.yaml", text));
  EXPECT_EQ(c.policy_path, "p.csv");
  EXPECT_EQ(c.k, 10U);
  EXPECT_EQ(c.threads, 8);
  std::string bad = text;
  bad.replace(bad.find("k: 10"), 5, "k: 0 ");
  EXPECT_THROW(ars::LoadPolicyEvalConfig(Write("vb.yaml", bad)),
               std::runtime_error);
}

}  // namespace
