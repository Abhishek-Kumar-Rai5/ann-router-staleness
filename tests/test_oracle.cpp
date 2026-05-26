// Phase 2: query split, ef grid, oracle-label derivation, and parallel batch
// search equivalence.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "ars/config.h"
#include "ars/ground_truth.h"
#include "ars/hnsw_index.h"
#include "ars/metrics.h"
#include "ars/oracle.h"
#include "ars/query_split.h"

namespace {

// ---------------------------------------------------------------- split ----

TEST(QuerySplit, Mt19937_64MatchesStandardMandatedValue) {
  // [rand.predef]: the 10000th output of a default-constructed mt19937_64 is
  // 9981545732273789042. Pins the engine the split depends on.
  std::mt19937_64 rng;
  rng.discard(9999);
  EXPECT_EQ(rng(), 9981545732273789042ULL);
}

TEST(QuerySplit, UniformBelowStaysInRangeAndCoversIt) {
  std::mt19937_64 rng(1);
  std::vector<int> counts(7, 0);
  for (int i = 0; i < 70000; ++i) {
    const auto v = ars::UniformBelow(rng, 7);
    ASSERT_LT(v, 7U);
    ++counts[v];
  }
  for (const int c : counts) {
    EXPECT_NEAR(c, 10000, 500);  // ~5 sigma
  }
  EXPECT_EQ(ars::UniformBelow(rng, 1), 0U);
  EXPECT_THROW(ars::UniformBelow(rng, 0), std::invalid_argument);
}

TEST(QuerySplit, PartitionHasExactSizes) {
  const auto split = ars::MakeQuerySplit(10000, 2000, 20261001);
  ASSERT_EQ(split.size(), 10000U);
  EXPECT_EQ(std::count(split.begin(), split.end(), ars::Split::kTest), 2000);
  EXPECT_EQ(std::count(split.begin(), split.end(), ars::Split::kTrain), 8000);
}

TEST(QuerySplit, DeterministicForSeedAndDifferentAcrossSeeds) {
  EXPECT_EQ(ars::MakeQuerySplit(5000, 1000, 7),
            ars::MakeQuerySplit(5000, 1000, 7));
  EXPECT_NE(ars::MakeQuerySplit(5000, 1000, 7),
            ars::MakeQuerySplit(5000, 1000, 8));
}

TEST(QuerySplit, GoldenValuesPinTheAlgorithm) {
  // Regression guard: any change to the shuffle, the bounded-int sampler or
  // the engine changes these ids, which would silently change the project's
  // fixed query split. Recorded when the split was first established.
  const auto split = ars::MakeQuerySplit(20, 5, 1);
  std::vector<std::size_t> test_ids;
  for (std::size_t q = 0; q < split.size(); ++q) {
    if (split[q] == ars::Split::kTest) {
      test_ids.push_back(q);
    }
  }
  EXPECT_EQ(test_ids, (std::vector<std::size_t>{1, 7, 10, 14, 17}));
}

TEST(QuerySplit, TestSetIsNotAContiguousBlock) {
  // A uniform random split should spread test ids over the whole id range.
  const auto split = ars::MakeQuerySplit(10000, 2000, 20261001);
  int first_half = 0;
  for (std::size_t q = 0; q < 5000; ++q) {
    first_half += split[q] == ars::Split::kTest ? 1 : 0;
  }
  EXPECT_NEAR(first_half, 1000, 150);
}

TEST(QuerySplit, CsvRoundTripAndRejectsBadFiles) {
  const auto split = ars::MakeQuerySplit(50, 10, 3);
  const std::string path = ::testing::TempDir() + "/split.csv";
  std::ofstream(path) << ars::SplitToCsv(split);
  EXPECT_EQ(ars::ReadSplitCsv(path), split);
  std::ofstream(path) << "query_id,split\n0,train\n2,test\n";  // gap in ids
  EXPECT_THROW(ars::ReadSplitCsv(path), std::runtime_error);
  std::ofstream(path) << "query_id,split\n0,validation\n";
  EXPECT_THROW(ars::ReadSplitCsv(path), std::runtime_error);
}

TEST(QuerySplit, RejectsDegenerateSizes) {
  EXPECT_THROW(ars::MakeQuerySplit(10, 0, 1), std::invalid_argument);
  EXPECT_THROW(ars::MakeQuerySplit(10, 10, 1), std::invalid_argument);
}

// -------------------------------------------------------------- ef grid ----

TEST(EfGrid, IntegerStepsThenGeometricAndEndsAtMax) {
  const auto g = ars::GeometricEfGrid(10, 4096, 1.05);
  EXPECT_EQ(g.front(), 10U);
  EXPECT_EQ(g.back(), 4096U);
  // 10..20 inclusive: ratio*ef < ef+1 there, so the step is +1.
  for (std::size_t i = 0; i <= 10; ++i) {
    EXPECT_EQ(g[i], 10 + i);
  }
  for (std::size_t i = 1; i < g.size(); ++i) {
    ASSERT_GT(g[i], g[i - 1]);
    // Relative step never exceeds the ratio (rounding aside), except the
    // final clamp to max, which can only shrink a step.
    EXPECT_LE(static_cast<double>(g[i]),
              std::max(g[i - 1] + 1.0, g[i - 1] * 1.05 + 0.5));
  }
}

TEST(EfGrid, DegenerateAndInvalid) {
  EXPECT_EQ(ars::GeometricEfGrid(16, 16, 2.0), (std::vector<std::size_t>{16}));
  EXPECT_EQ(ars::GeometricEfGrid(10, 45, 2.0),
            (std::vector<std::size_t>{10, 20, 40, 45}));
  EXPECT_THROW(ars::GeometricEfGrid(0, 10, 2.0), std::invalid_argument);
  EXPECT_THROW(ars::GeometricEfGrid(10, 5, 2.0), std::invalid_argument);
  EXPECT_THROW(ars::GeometricEfGrid(10, 20, 1.0), std::invalid_argument);
}

// ----------------------------------------------------------- oracle label --

TEST(OracleLabel, MonotoneCurvePicksFirstEfAtTarget) {
  const auto l = ars::DeriveOracleLabel({0.5, 0.8, 0.9, 1.0, 1.0}, 0.9);
  ASSERT_TRUE(l.reached);
  EXPECT_EQ(l.index, 2U);
  EXPECT_EQ(l.first_reach_index, 2U);
}

TEST(OracleLabel, TargetBetweenRecallStepsNeedsNextStep) {
  // With k = 10, recall is a multiple of 0.1: a 0.95 target needs 1.0.
  const auto l = ars::DeriveOracleLabel({0.8, 0.9, 0.9, 1.0}, 0.95);
  ASSERT_TRUE(l.reached);
  EXPECT_EQ(l.index, 3U);
}

TEST(OracleLabel, ExactBoundaryValueCountsAsReached) {
  // 9/10 computed in floating point must satisfy a 0.9 target.
  const auto l = ars::DeriveOracleLabel({0.0, 9.0 / 10.0}, 0.9);
  ASSERT_TRUE(l.reached);
  EXPECT_EQ(l.index, 1U);
}

TEST(OracleLabel, NonMonotoneCurveUsesStableReach) {
  // Reaches target at index 1, dips at 2, reaches again from 3 on: the
  // oracle is the stable point (3); first_reach records the transient (1).
  const auto l = ars::DeriveOracleLabel({0.8, 1.0, 0.9, 1.0, 1.0}, 1.0);
  ASSERT_TRUE(l.reached);
  EXPECT_EQ(l.index, 3U);
  EXPECT_EQ(l.first_reach_index, 1U);
}

TEST(OracleLabel, AlreadyAtTargetAtSmallestEf) {
  const auto l = ars::DeriveOracleLabel({1.0, 1.0, 1.0}, 0.95);
  ASSERT_TRUE(l.reached);
  EXPECT_EQ(l.index, 0U);
  EXPECT_EQ(l.first_reach_index, 0U);
}

TEST(OracleLabel, NotReachedAtMaxEfIsCensored) {
  EXPECT_FALSE(ars::DeriveOracleLabel({0.5, 0.9, 0.9}, 0.95).reached);
  // Reaching it transiently but ending below target is also censored.
  EXPECT_FALSE(ars::DeriveOracleLabel({0.5, 1.0, 0.9}, 0.95).reached);
  EXPECT_THROW(ars::DeriveOracleLabel({}, 0.9), std::invalid_argument);
}

// ------------------------------------------------- search batch / oracle ---

ars::FloatMatrix Gaussian(std::size_t rows, std::size_t dim,
                          std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  ars::FloatMatrix m{std::vector<float>(rows * dim), rows, dim};
  for (float& x : m.data) {
    x = normal(rng);
  }
  return m;
}

TEST(SearchBatch, IdenticalToSerialSearchForAnyThreadCount) {
  const auto base = Gaussian(4000, 16, 21);
  const auto queries = Gaussian(300, 16, 22);
  ars::HnswIndex index(16, base.rows,
                       {.m = 16, .ef_construction = 100, .seed = 5});
  index.Add(base, 0, 1);
  for (const std::size_t ef : {10U, 37U, 200U}) {
    std::vector<ars::SearchResult> serial;
    for (std::size_t q = 0; q < queries.rows; ++q) {
      serial.push_back(index.Search(queries.Row(q), 10, ef));
    }
    for (const int threads : {1, 4, 16}) {
      const auto batch = index.SearchBatch(queries, 10, ef, threads);
      ASSERT_EQ(batch.size(), serial.size());
      for (std::size_t q = 0; q < queries.rows; ++q) {
        EXPECT_EQ(batch[q].labels, serial[q].labels);
        EXPECT_EQ(batch[q].dists, serial[q].dists);
        EXPECT_EQ(batch[q].distance_computations,
                  serial[q].distance_computations);
      }
    }
  }
}

TEST(OracleEndToEnd, LabelIsMinimalEfOnRealSearchCurves) {
  // Small real index: derive labels from search curves over a grid and check
  // the defining property directly — target met at the label ef and at every
  // larger grid ef, and not met at the grid ef just below it.
  const auto base = Gaussian(3000, 16, 31);
  const auto queries = Gaussian(100, 16, 32);
  const auto gt = ars::BruteForceKnn(base, queries, 10);
  ars::HnswIndex index(16, base.rows,
                       {.m = 8, .ef_construction = 40, .seed = 9});
  index.Add(base, 0, 1);
  const auto grid = ars::GeometricEfGrid(10, 400, 1.2);
  std::vector<std::vector<double>> curves(queries.rows,
                                          std::vector<double>(grid.size()));
  for (std::size_t e = 0; e < grid.size(); ++e) {
    const auto res = index.SearchBatch(queries, 10, grid[e], 4);
    for (std::size_t q = 0; q < queries.rows; ++q) {
      curves[q][e] = ars::TieAwareRecallAtK(gt.dists.Row(q), res[q].dists, 10);
    }
  }
  std::set<std::size_t> distinct_labels;
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto l = ars::DeriveOracleLabel(curves[q], 1.0);
    if (!l.reached) {
      continue;
    }
    distinct_labels.insert(l.index);
    for (std::size_t e = l.index; e < grid.size(); ++e) {
      EXPECT_GE(curves[q][e], 1.0 - 1e-9);
    }
    if (l.index > 0) {
      EXPECT_LT(curves[q][l.index - 1], 1.0 - 1e-9);
    }
    EXPECT_LE(l.first_reach_index, l.index);
  }
  // A weak index (M=8, efC=40) should give queries of differing difficulty.
  EXPECT_GT(distinct_labels.size(), 1U);
}

// ---------------------------------------------------------------- config ---

TEST(OracleConfig, FixedFileSplitNeedsNoSeedOrSize) {
  const std::string text = R"(experiment_name: o
dataset: {base: b, queries: q, max_base: 0, max_queries: 0}
index: {m: 16, ef_construction: 200, seed: 43, build_threads: 1}
ground_truth: {k: 100}
paths: {cache_dir: c, output_dir: o}
split: {path: tags.csv, fixed_file: true}
oracle:
  k: 10
  target_recall: 0.95
  recall_definition: tie_aware
  ef_grid: {min: 10, max: 4096, ratio: 1.05}
  search_threads: 4
)";
  const std::string path = ::testing::TempDir() + "/oracle_fixed.yaml";
  std::ofstream(path) << text;
  const auto c = ars::LoadOracleConfig(path);
  EXPECT_TRUE(c.split_fixed_file);
  EXPECT_EQ(c.split_path, "tags.csv");
  EXPECT_EQ(c.index.seed, 43U);
}

TEST(QuerySubsetConfig, ParsesAllFields) {
  const std::string text = R"(experiment_name: s
source: src.fvecs
n_select: 7
seed: 20261005
query_set_id: qs
output_fvecs: o.fvecs
output_ids: o_ids.csv
output_tagging: o_tags.csv
output_dir: r
)";
  const std::string path = ::testing::TempDir() + "/subset.yaml";
  std::ofstream(path) << text;
  const auto c = ars::LoadQuerySubsetConfig(path);
  EXPECT_EQ(c.n_select, 7U);
  EXPECT_EQ(c.seed, 20261005U);
  EXPECT_EQ(c.query_set_id, "qs");
}

TEST(OracleConfig, ParsesAndValidates) {
  const std::string base = R"(experiment_name: o
dataset: {base: b, queries: q, max_base: 0, max_queries: 0}
index: {m: 16, ef_construction: 200, seed: 42, build_threads: 1}
ground_truth: {k: 100}
paths: {cache_dir: c, output_dir: o}
split: {seed: 11, test_size: 20, path: s.csv}
oracle:
  k: 10
  target_recall: 0.95
  recall_definition: tie_aware
  ef_grid: {min: 10, max: 100, ratio: 1.05}
  search_threads: 4
)";
  const std::string path = ::testing::TempDir() + "/oracle.yaml";
  std::ofstream(path) << base;
  const auto c = ars::LoadOracleConfig(path);
  EXPECT_EQ(c.split_seed, 11U);
  EXPECT_EQ(c.split_test_size, 20U);
  EXPECT_DOUBLE_EQ(c.target_recall, 0.95);
  EXPECT_EQ(c.recall_definition, "tie_aware");
  EXPECT_EQ(c.ef_grid_max, 100U);
  EXPECT_EQ(c.index.seed, 42U);  // shared S0 sections parsed too

  std::string bad = base;
  bad.replace(bad.find("tie_aware"), 9, "fuzzy");
  std::ofstream(path) << bad;
  EXPECT_THROW(ars::LoadOracleConfig(path), std::runtime_error);
  bad = base;
  bad.replace(bad.find("0.95"), 4, "1.50");
  std::ofstream(path) << bad;
  EXPECT_THROW(ars::LoadOracleConfig(path), std::runtime_error);
}

}  // namespace
