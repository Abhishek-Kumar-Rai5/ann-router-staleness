// Index management: search correctness against brute force, exact distance
// counting, save/load round trip, and seeded reproducibility.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "ars/ground_truth.h"
#include "ars/hnsw_index.h"
#include "ars/metrics.h"

namespace {

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

double MeanRecall(ars::HnswIndex& index, const ars::FloatMatrix& queries,
                  const ars::KnnResult& gt, std::size_t k, std::size_t ef) {
  double sum = 0.0;
  for (std::size_t q = 0; q < queries.rows; ++q) {
    sum += ars::RecallAtK(gt.ids.Row(q),
                          index.Search(queries.Row(q), k, ef).labels, k);
  }
  return sum / static_cast<double>(queries.rows);
}

constexpr ars::HnswParams kParams{.m = 16, .ef_construction = 200, .seed = 7};

TEST(CountingL2Space, MatchesReferenceDistanceAndCounts) {
  const auto data = Gaussian(2, 32, 1);
  ars::CountingL2Space space(32);
  auto fn = space.get_dist_func();
  ars::CountingL2Space::ThreadCount() = 0;
  const float d = fn(data.Row(0), data.Row(1), space.get_dist_func_param());
  EXPECT_NEAR(d, ars::SquaredL2(data.Row(0), data.Row(1), 32), 1e-4);
  (void)fn(data.Row(0), data.Row(0), space.get_dist_func_param());
  EXPECT_EQ(ars::CountingL2Space::ThreadCount(), 2U);
}

TEST(HnswIndex, ReturnsSortedResultsWithConsistentDistances) {
  const auto base = Gaussian(2000, 16, 2);
  const auto queries = Gaussian(20, 16, 3);
  ars::HnswIndex index(16, base.rows, kParams);
  index.Add(base, 0, 1);
  ASSERT_EQ(index.Size(), base.rows);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto r = index.Search(queries.Row(q), 10, 50);
    ASSERT_EQ(r.labels.size(), 10U);
    for (std::size_t i = 0; i < r.labels.size(); ++i) {
      EXPECT_NEAR(r.dists[i],
                  ars::SquaredL2(queries.Row(q), base.Row(r.labels[i]), 16),
                  1e-3);
      if (i > 0) {
        EXPECT_LE(r.dists[i - 1], r.dists[i]);
      }
    }
  }
}

TEST(HnswIndex, RecallReachesOneAtLargeEfAndIsMonotoneInEf) {
  const auto base = Gaussian(5000, 16, 4);
  const auto queries = Gaussian(200, 16, 5);
  const ars::KnnResult gt = ars::BruteForceKnn(base, queries, 10);
  ars::HnswIndex index(16, base.rows, kParams);
  index.Add(base, 0, 1);
  const double low = MeanRecall(index, queries, gt, 10, 10);
  const double high = MeanRecall(index, queries, gt, 10, 500);
  EXPECT_LT(low, high);
  EXPECT_GE(high, 0.999);
}

TEST(HnswIndex, DistanceCountIsPerQueryAndGrowsWithEf) {
  const auto base = Gaussian(5000, 16, 6);
  const auto queries = Gaussian(50, 16, 7);
  ars::HnswIndex index(16, base.rows, kParams);
  index.Add(base, 0, 1);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto small = index.Search(queries.Row(q), 10, 10);
    const auto large = index.Search(queries.Row(q), 10, 400);
    EXPECT_GT(small.distance_computations, 0U);
    EXPECT_LT(small.distance_computations, base.rows);
    EXPECT_GT(large.distance_computations, small.distance_computations);
    // Counter is reset per search: repeating gives the identical count.
    EXPECT_EQ(index.Search(queries.Row(q), 10, 10).distance_computations,
              small.distance_computations);
  }
}

TEST(HnswIndex, SingleThreadedBuildIsReproducibleForFixedSeed) {
  const auto base = Gaussian(3000, 16, 8);
  const auto queries = Gaussian(50, 16, 9);
  ars::HnswIndex a(16, base.rows, kParams);
  ars::HnswIndex b(16, base.rows, kParams);
  a.Add(base, 0, 1);
  b.Add(base, 0, 1);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto ra = a.Search(queries.Row(q), 10, 20);
    const auto rb = b.Search(queries.Row(q), 10, 20);
    EXPECT_EQ(ra.labels, rb.labels);
    EXPECT_EQ(ra.distance_computations, rb.distance_computations);
  }
}

TEST(HnswIndex, SaveLoadRoundTripGivesIdenticalSearches) {
  const auto base = Gaussian(2000, 16, 10);
  const auto queries = Gaussian(30, 16, 11);
  ars::HnswIndex index(16, base.rows, kParams);
  index.Add(base, 0, 4);
  const std::string path = ::testing::TempDir() + "/ars_hnsw_roundtrip.bin";
  index.Save(path);
  ars::HnswIndex loaded = ars::HnswIndex::Load(path, 16);
  std::filesystem::remove(path);
  ASSERT_EQ(loaded.Size(), index.Size());
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto r1 = index.Search(queries.Row(q), 10, 40);
    const auto r2 = loaded.Search(queries.Row(q), 10, 40);
    EXPECT_EQ(r1.labels, r2.labels);
    EXPECT_EQ(r1.dists, r2.dists);
    EXPECT_EQ(r1.distance_computations, r2.distance_computations);
  }
}

TEST(HnswIndex, LabelsHonourFirstLabelOffset) {
  const auto base = Gaussian(100, 8, 12);
  ars::HnswIndex index(8, base.rows, kParams);
  index.Add(base, 1000, 1);
  const auto r = index.Search(base.Row(17), 1, 50);
  ASSERT_EQ(r.labels.size(), 1U);
  EXPECT_EQ(r.labels[0], 1017U);
}

}  // namespace
