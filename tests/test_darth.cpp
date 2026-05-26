// Experiment C (design_doc.md C1.9): DARTH re-implementation — feature
// arithmetic, plain-mode equivalence with hnswlib searchKnn, trace labels.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "ars/darth.h"
#include "ars/ground_truth.h"
#include "ars/hnsw_index.h"
#include "ars/metrics.h"
#include "ars/vecs_io.h"

namespace {

namespace darth = ars::darth;

ars::FloatMatrix Gauss(std::size_t rows, std::size_t dim, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0F, 1.0F);
  ars::FloatMatrix m{std::vector<float>(rows * dim), rows, dim};
  for (float& x : m.data) {
    x = d(rng);
  }
  return m;
}

TEST(DarthFeatures, FrozenOrderAndArithmetic) {
  std::vector<float> kbest = {7, 3, 10, 1, 5, 9, 2, 8, 4, 6};
  const auto f = darth::MakeFeatures(4, 120, 17, 42.5F, kbest);
  EXPECT_EQ(darth::kFeatureNames[5], std::string("furthest_dist"));
  EXPECT_EQ(darth::kFeatureNames[6], std::string("avg_dist"));
  EXPECT_DOUBLE_EQ(f[0], 4);     // step
  EXPECT_DOUBLE_EQ(f[1], 120);   // dists
  EXPECT_DOUBLE_EQ(f[2], 17);    // inserts
  EXPECT_DOUBLE_EQ(f[3], 42.5);  // first_nn_dist
  EXPECT_DOUBLE_EQ(f[4], 1);     // nn_dist
  EXPECT_DOUBLE_EQ(f[5], 10);    // furthest_dist
  EXPECT_DOUBLE_EQ(f[6], 5.5);   // avg_dist
  EXPECT_DOUBLE_EQ(f[7], 8.25);  // population variance
  EXPECT_DOUBLE_EQ(f[8], 3);     // index floor(0.25*9) = 2
  EXPECT_DOUBLE_EQ(f[9], 5);     // index floor(0.50*9) = 4
  EXPECT_DOUBLE_EQ(f[10], 7);    // index floor(0.75*9) = 6
}

void ExpectPlainEqualsSearchKnn(ars::HnswIndex& idx,
                                const ars::FloatMatrix& q) {
  for (const std::size_t ef : {10U, 23U, 64U, 150U}) {
    for (std::size_t i = 0; i < q.rows; ++i) {
      darth::Params p;
      p.k = 10;
      p.ef = ef;
      p.no_deletions = idx.DeletedCount() == 0;
      darth::StopCondition sc(p, darth::Mode::kPlain, nullptr, nullptr);
      const auto a = idx.SearchWithStopCondition(q.Row(i), 10, sc);
      const auto b = idx.Search(q.Row(i), 10, ef);
      ASSERT_EQ(a.labels, b.labels) << "ef " << ef << " q " << i;
      ASSERT_EQ(a.dists, b.dists);
      ASSERT_EQ(a.distance_computations, b.distance_computations);
      EXPECT_FALSE(sc.stopped_early());
    }
  }
}

TEST(DarthPlainMode, ReproducesSearchKnnExactly) {
  const auto base = Gauss(4000, 16, 1);
  const auto q = Gauss(60, 16, 2);
  ars::HnswIndex idx(16, base.rows, {.m = 8, .ef_construction = 64, .seed = 7});
  idx.Add(base, 0, 1);
  ExpectPlainEqualsSearchKnn(idx, q);
}

TEST(DarthPlainMode, ReproducesSearchKnnWithLazyDeletions) {
  const auto base = Gauss(3000, 12, 3);
  const auto q = Gauss(50, 12, 4);
  ars::HnswIndex idx(12, base.rows, {.m = 8, .ef_construction = 64, .seed = 9});
  idx.Add(base, 0, 1);
  for (std::size_t l = 0; l < base.rows; l += 9) {
    idx.MarkDeleted(l);
  }
  ExpectPlainEqualsSearchKnn(idx, q);
}

TEST(DarthTraceMode, LabelsAreIdRecallAndSearchUnchanged) {
  const auto base = Gauss(3000, 16, 5);
  const auto q = Gauss(30, 16, 6);
  ars::HnswIndex idx(16, base.rows, {.m = 8, .ef_construction = 64, .seed = 3});
  idx.Add(base, 0, 1);
  const auto gt = ars::BruteForceKnn(base, q, 10);
  for (std::size_t i = 0; i < q.rows; ++i) {
    darth::Params p;
    p.k = 10;
    p.ef = 80;
    darth::StopCondition sc(p, darth::Mode::kTrace, nullptr, gt.ids.Row(i));
    const auto a = idx.SearchWithStopCondition(q.Row(i), 10, sc);
    const auto b = idx.Search(q.Row(i), 10, 80);
    ASSERT_EQ(a.labels, b.labels);
    ASSERT_EQ(a.distance_computations, b.distance_computations);
    const auto& obs = sc.observations();
    ASSERT_FALSE(obs.empty());
    EXPECT_LE(obs.size(), static_cast<std::size_t>(sc.dists()));
    // Last observation's label = id recall of the final k-best.
    EXPECT_DOUBLE_EQ(obs.back().recall,
                     ars::RecallAtK(gt.ids.Row(i), sc.KBestLabels(), 10));
    for (std::size_t j = 1; j < obs.size(); ++j) {
      EXPECT_GE(obs[j].x[1], obs[j - 1].x[1]);  // dists non-decreasing
      EXPECT_GE(obs[j].x[2], obs[j - 1].x[2]);  // inserts non-decreasing
      EXPECT_LE(obs[j].x[4], obs[j - 1].x[4]);  // nn_dist non-increasing
      EXPECT_LE(obs[j].x[5], obs[j - 1].x[5]);  // furthest non-increasing
      EXPECT_GE(obs[j].x[2], 10);
    }
  }
}

}  // namespace
