#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ars/config.h"
#include "ars/features.h"
#include "ars/ground_truth.h"
#include "ars/hnsw_index.h"

namespace {

std::vector<float> Sq(const std::vector<double>& d) {
  std::vector<float> out;
  for (const double x : d) {
    out.push_back(static_cast<float>(x * x));
  }
  return out;
}

TEST(Features, KnnDistanceAndConcentrationOnHandValues) {
  const auto sq = Sq({1.0, 2.0, 4.0, 8.0});
  EXPECT_DOUBLE_EQ(ars::KnnDistance(sq, 3), 4.0);
  EXPECT_DOUBLE_EQ(ars::KnnDistance(sq, 4), 8.0);
  EXPECT_DOUBLE_EQ(ars::ScoreConcentration(sq, 4), 1.0 / 8.0);
  EXPECT_DOUBLE_EQ(ars::ScoreConcentration(Sq({3, 3, 3}), 3), 1.0);
  EXPECT_DOUBLE_EQ(ars::ScoreConcentration(Sq({0, 0}), 2), 1.0);
  EXPECT_THROW(ars::KnnDistance(sq, 5), std::invalid_argument);
  EXPECT_THROW(ars::ScoreConcentration(sq, 0), std::invalid_argument);
}

TEST(Features, LidMleOnHandValues) {
  EXPECT_NEAR(ars::LidMle(Sq({1, 2, 4}), 3), 2.0 / std::log(8.0), 1e-6);
  EXPECT_NEAR(ars::LidMle(Sq({1, 2, 4, 100}), 3), 2.0 / std::log(8.0), 1e-6);
}

TEST(Features, LidMleRecoversKnownIntrinsicDimension) {
  for (const int dim : {2, 5, 12}) {
    std::vector<double> d;
    const int k = 2000;
    for (int i = 1; i <= k; ++i) {
      d.push_back(std::pow(static_cast<double>(i) / k, 1.0 / dim));
    }
    EXPECT_NEAR(ars::LidMle(Sq(d), k), dim, 0.05 * dim) << "D=" << dim;
  }
}

TEST(Features, LidMleEdgeConventions) {
  EXPECT_DOUBLE_EQ(ars::LidMle(Sq({0, 0, 0}), 3), 0.0);
  EXPECT_DOUBLE_EQ(ars::LidMle(Sq({0, 1, 2}), 3), 0.0);
  EXPECT_EQ(ars::LidMle(Sq({2, 2, 2}), 3),
            std::numeric_limits<double>::infinity());
  EXPECT_THROW(ars::LidMle(Sq({1}), 1), std::invalid_argument);
  EXPECT_THROW(ars::LidMle(Sq({1, 2}), 3), std::invalid_argument);
}

TEST(Features, CentroidAndCentroidDistance) {
  const ars::FloatMatrix base{{0, 0, 2, 4, 4, 8}, 3, 2};
  const auto c = ars::ComputeCentroid(base);
  ASSERT_EQ(c.size(), 2U);
  EXPECT_DOUBLE_EQ(c[0], 2.0);
  EXPECT_DOUBLE_EQ(c[1], 4.0);
  const std::vector<float> q = {5.0F, 8.0F};
  EXPECT_DOUBLE_EQ(ars::CentroidDistance(q.data(), c), 5.0);
  EXPECT_THROW(ars::ComputeCentroid(ars::FloatMatrix{}), std::invalid_argument);
}

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

TEST(Features, ExtractionIsDeterministicAndConsistentWithExactSearch) {
  const auto base = Gaussian(4000, 16, 41);
  const auto queries = Gaussian(150, 16, 42);
  ars::HnswIndex index(16, base.rows,
                       {.m = 16, .ef_construction = 100, .seed = 3});
  index.Add(base, 0, 1);
  const auto centroid = ars::ComputeCentroid(base);
  const ars::FeatureParams p{
      .probe_k = 10, .probe_ef = 10, .lid_k = 20, .lid_ef = 20};
  const auto f1 = ars::ExtractFeatures(index, queries, centroid, p, 1);
  const auto f8 = ars::ExtractFeatures(index, queries, centroid, p, 8);
  const auto gt = ars::BruteForceKnn(base, queries, 10);

  for (std::size_t q = 0; q < queries.rows; ++q) {
    EXPECT_EQ(f1[q].knn_dist, f8[q].knn_dist);
    EXPECT_EQ(f1[q].score_concentration, f8[q].score_concentration);
    EXPECT_EQ(f1[q].lid, f8[q].lid);
    EXPECT_EQ(f1[q].probe_distance_computations,
              f8[q].probe_distance_computations);
    EXPECT_GE(f1[q].knn_dist + 1e-5, std::sqrt(gt.dists.Row(q)[9]));
    EXPECT_GT(f1[q].score_concentration, 0.0);
    EXPECT_LE(f1[q].score_concentration, 1.0);
    EXPECT_TRUE(std::isfinite(f1[q].lid));
    EXPECT_GT(f1[q].lid, 0.0);
    EXPECT_GT(f1[q].probe_distance_computations, 0U);
    EXPECT_GE(f1[q].lid_probe_distance_computations,
              f1[q].probe_distance_computations);
    const auto lp = index.Search(queries.Row(q), 20, 20);
    EXPECT_EQ(f1[q].knn_dist_lid_probe, ars::KnnDistance(lp.dists, 10));
    EXPECT_EQ(f1[q].score_concentration_lid_probe,
              ars::ScoreConcentration(lp.dists, 10));
    const auto s10 = index.Search(queries.Row(q), 10, 20);
    EXPECT_EQ(f1[q].knn_dist_lid_probe, ars::KnnDistance(s10.dists, 10));
    EXPECT_EQ(f1[q].lid_probe_distance_computations, s10.distance_computations);
    EXPECT_NEAR(f1[q].centroid_dist,
                ars::CentroidDistance(queries.Row(q), centroid), 1e-12);
  }
}

TEST(FeatureConfig, ParsesAndValidates) {
  const std::string text = R"(experiment_name: f
dataset: {base: b, queries: q, max_base: 0, max_queries: 0}
index: {m: 16, ef_construction: 200, seed: 42, build_threads: 1}
ground_truth: {k: 100}
paths: {cache_dir: c, output_dir: o}
split: {path: s.csv}
features: {probe_k: 10, probe_ef: 10, lid_k: 20, lid_ef: 20, threads: 4}
)";
  const std::string path = ::testing::TempDir() + "/features.yaml";
  std::ofstream(path) << text;
  const auto c = ars::LoadFeatureConfig(path);
  EXPECT_EQ(c.params.probe_k, 10U);
  EXPECT_EQ(c.params.lid_ef, 20U);
  EXPECT_EQ(c.threads, 4);
  EXPECT_EQ(c.split_path, "s.csv");
  std::string bad = text;
  bad.replace(bad.find("probe_k: 10"), 11, "probe_k: 1");
  std::ofstream(path) << bad;
  EXPECT_THROW(ars::LoadFeatureConfig(path), std::runtime_error);
}

}  // namespace
