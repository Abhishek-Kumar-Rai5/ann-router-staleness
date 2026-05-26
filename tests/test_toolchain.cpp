// Phase 0 smoke tests: verify the toolchain, OpenMP, and that hnswlib
// compiles and links as a black box. No project logic is tested here.

#include <gtest/gtest.h>
#include <omp.h>

#include <cstddef>
#include <random>
#include <vector>

#include "ars/build_info.h"
#include "hnswlib/hnswlib.h"

namespace {

TEST(Toolchain, BuildInfoIsPopulated) {
  const ars::BuildInfo info = ars::GetBuildInfo();
  EXPECT_FALSE(info.compiler_id.empty());
  EXPECT_FALSE(info.compiler_version.empty());
  EXPECT_GE(info.cxx_standard, 202002L);
  EXPECT_GT(info.openmp_version, 0);
}

TEST(Toolchain, OpenMPRunsParallelRegion) {
  int sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < 1000; ++i) {
    sum += 1;
  }
  EXPECT_EQ(sum, 1000);
  EXPECT_GE(omp_get_max_threads(), 1);
}

// Each inserted point, queried exactly, should be its own nearest neighbour.
TEST(HnswlibSmoke, SelfQueryReturnsSelf) {
  constexpr int kDim = 16;
  constexpr std::size_t kN = 500;
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> data(kN * kDim);
  for (float& x : data) {
    x = dist(rng);
  }

  hnswlib::L2Space space(kDim);
  hnswlib::HierarchicalNSW<float> index(&space, kN, /*M=*/16,
                                        /*ef_construction=*/200,
                                        /*random_seed=*/42);
  for (std::size_t i = 0; i < kN; ++i) {
    index.addPoint(&data[i * kDim], i);
  }
  index.setEf(100);

  std::size_t hits = 0;
  for (std::size_t i = 0; i < kN; ++i) {
    auto result = index.searchKnn(&data[i * kDim], 1);
    ASSERT_EQ(result.size(), 1U);
    if (result.top().second == i) {
      ++hits;
    }
  }
  EXPECT_EQ(hits, kN);
}

TEST(HnswlibSmoke, MarkDeleteExcludesLabel) {
  constexpr int kDim = 4;
  hnswlib::L2Space space(kDim);
  hnswlib::HierarchicalNSW<float> index(&space, 10, 16, 100, 42);
  std::vector<float> a = {0, 0, 0, 0};
  std::vector<float> b = {1, 1, 1, 1};
  index.addPoint(a.data(), 0);
  index.addPoint(b.data(), 1);
  index.markDelete(0);
  auto result = index.searchKnn(a.data(), 1);
  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result.top().second, 1U);
}

}  // namespace
