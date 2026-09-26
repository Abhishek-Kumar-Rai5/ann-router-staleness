#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ars/ground_truth.h"

namespace {

ars::FloatMatrix RandomMatrix(std::size_t rows, std::size_t dim,
                              std::uint32_t seed, bool integer_valued) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> ints(0, 255);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  ars::FloatMatrix m{std::vector<float>(rows * dim), rows, dim};
  for (float& x : m.data) {
    x = integer_valued ? static_cast<float>(ints(rng)) : normal(rng);
  }
  return m;
}

std::vector<std::int32_t> NaiveKnn(const ars::FloatMatrix& base, const float* q,
                                   std::size_t k) {
  std::vector<std::pair<double, std::int32_t>> all;
  for (std::size_t b = 0; b < base.rows; ++b) {
    double d = 0.0;
    for (std::size_t j = 0; j < base.dim; ++j) {
      const double diff = static_cast<double>(q[j]) - base.Row(b)[j];
      d += diff * diff;
    }
    all.emplace_back(d, static_cast<std::int32_t>(b));
  }
  std::sort(all.begin(), all.end());
  std::vector<std::int32_t> ids;
  for (std::size_t i = 0; i < k; ++i) {
    ids.push_back(all[i].second);
  }
  return ids;
}

TEST(SquaredL2, MatchesHandComputedValue) {
  std::vector<float> a(19, 0.0F);
  std::vector<float> b(19, 0.0F);
  a[0] = 3.0F;
  b[0] = 0.0F;
  a[17] = 1.0F;
  b[17] = 5.0F;
  a[18] = -2.0F;
  b[18] = 1.0F;
  EXPECT_FLOAT_EQ(ars::SquaredL2(a.data(), b.data(), a.size()), 34.0F);
  EXPECT_FLOAT_EQ(ars::SquaredL2(a.data(), a.data(), a.size()), 0.0F);
}

TEST(BruteForceKnn, HandBuiltOneDimensionalCase) {
  const ars::FloatMatrix base{{0, 10, 3, 7, 1}, 5, 1};
  const ars::FloatMatrix q{{2}, 1, 1};
  const ars::KnnResult r = ars::BruteForceKnn(base, q, 4);
  EXPECT_EQ(r.ids.data, (std::vector<std::int32_t>{2, 4, 0, 3}));
  EXPECT_EQ(r.dists.data, (std::vector<float>{1, 1, 4, 25}));
}

TEST(BruteForceKnn, DuplicatePointsTieBreakBySmallerId) {
  const ars::FloatMatrix base{std::vector<float>(5 * 3, 1.0F), 5, 3};
  const ars::FloatMatrix q{{0, 0, 0}, 1, 3};
  const ars::KnnResult r = ars::BruteForceKnn(base, q, 3);
  EXPECT_EQ(r.ids.data, (std::vector<std::int32_t>{0, 1, 2}));
}

TEST(BruteForceKnn, MatchesNaiveSortOnIntegerData) {
  const auto base = RandomMatrix(5000, 24, 1, /*integer_valued=*/true);
  const auto queries = RandomMatrix(70, 24, 2, /*integer_valued=*/true);
  const std::size_t k = 50;
  const ars::KnnResult r = ars::BruteForceKnn(base, queries, k);
  ASSERT_EQ(r.ids.rows, queries.rows);
  ASSERT_EQ(r.ids.dim, k);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto expected = NaiveKnn(base, queries.Row(q), k);
    const std::vector<std::int32_t> got(r.ids.Row(q), r.ids.Row(q) + k);
    ASSERT_EQ(got, expected) << "query " << q;
    for (std::size_t i = 1; i < k; ++i) {
      ASSERT_LE(r.dists.Row(q)[i - 1], r.dists.Row(q)[i]);
    }
  }
}

// With real-valued data, float vs double rounding can swap near-ties, so this
// compares sets and distances rather than exact order.
TEST(BruteForceKnn, MatchesNaiveSortOnGaussianData) {
  const auto base = RandomMatrix(3000, 32, 3, /*integer_valued=*/false);
  const auto queries = RandomMatrix(40, 32, 4, /*integer_valued=*/false);
  const std::size_t k = 10;
  const ars::KnnResult r = ars::BruteForceKnn(base, queries, k);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    auto expected = NaiveKnn(base, queries.Row(q), k);
    std::vector<std::int32_t> got(r.ids.Row(q), r.ids.Row(q) + k);
    std::sort(expected.begin(), expected.end());
    std::sort(got.begin(), got.end());
    EXPECT_EQ(got, expected) << "query " << q;
    for (std::size_t i = 0; i < k; ++i) {
      const auto id = static_cast<std::size_t>(r.ids.Row(q)[i]);
      EXPECT_FLOAT_EQ(r.dists.Row(q)[i],
                      ars::SquaredL2(queries.Row(q), base.Row(id), base.dim));
    }
  }
}

TEST(BruteForceKnn, IndependentOfThreadCount) {
  const auto base = RandomMatrix(4000, 16, 5, /*integer_valued=*/true);
  const auto queries = RandomMatrix(100, 16, 6, /*integer_valued=*/true);
  const int saved = omp_get_max_threads();
  omp_set_num_threads(1);
  const ars::KnnResult single = ars::BruteForceKnn(base, queries, 20);
  omp_set_num_threads(std::max(4, saved));
  const ars::KnnResult multi = ars::BruteForceKnn(base, queries, 20);
  omp_set_num_threads(saved);
  EXPECT_EQ(single.ids.data, multi.ids.data);
  EXPECT_EQ(single.dists.data, multi.dists.data);
}

TEST(BruteForceKnn, SelfQueryFindsSelfAtDistanceZero) {
  const auto base = RandomMatrix(500, 8, 7, /*integer_valued=*/false);
  const ars::KnnResult r = ars::BruteForceKnn(base, base, 1);
  for (std::size_t i = 0; i < base.rows; ++i) {
    EXPECT_EQ(r.ids.Row(i)[0], static_cast<std::int32_t>(i));
    EXPECT_EQ(r.dists.Row(i)[0], 0.0F);
  }
}

TEST(BruteForceKnn, RejectsInvalidArguments) {
  const ars::FloatMatrix base{{0, 1, 2, 3}, 2, 2};
  const ars::FloatMatrix q2{{0, 0}, 1, 2};
  const ars::FloatMatrix q3{{0, 0, 0}, 1, 3};
  EXPECT_THROW(ars::BruteForceKnn(base, q3, 1), std::invalid_argument);
  EXPECT_THROW(ars::BruteForceKnn(base, q2, 0), std::invalid_argument);
  EXPECT_THROW(ars::BruteForceKnn(base, q2, 3), std::invalid_argument);
}

}  // namespace
