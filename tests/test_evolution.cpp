#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "ars/config.h"
#include "ars/features.h"
#include "ars/ground_truth.h"
#include "ars/hnsw_index.h"
#include "ars/s0_setup.h"
#include "ars/vecs_io.h"

namespace {

ars::FloatMatrix Ints(std::size_t rows, std::size_t dim, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> d(0, 100);
  ars::FloatMatrix m{std::vector<float>(rows * dim), rows, dim};
  for (float& x : m.data) {
    x = static_cast<float>(d(rng));
  }
  return m;
}

std::string Tmp(const std::string& n) { return ::testing::TempDir() + "/" + n; }

std::string Slurp(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

constexpr ars::HnswParams kP{.m = 16, .ef_construction = 100, .seed = 42};

TEST(MaskedGroundTruth, EmptyMaskIdenticalToUnmasked) {
  const auto base = Ints(3000, 16, 1);
  const auto q = Ints(50, 16, 2);
  const auto a = ars::BruteForceKnn(base, q, 20);
  const auto b = ars::BruteForceKnnExcluding(base, q, 20, {});
  EXPECT_EQ(a.ids.data, b.ids.data);
  EXPECT_EQ(a.dists.data, b.dists.data);
}

TEST(MaskedGroundTruth, ExcludedRowsNeverReturnedAndMatchesNaive) {
  const auto base = Ints(2000, 8, 3);
  const auto q = Ints(40, 8, 4);
  std::vector<std::uint8_t> ex(base.rows, 0);
  for (std::size_t i = 0; i < base.rows; i += 3) {
    ex[i] = 1;
  }
  const auto r = ars::BruteForceKnnExcluding(base, q, 10, ex);
  ars::FloatMatrix live{{}, 0, base.dim};
  std::vector<std::int32_t> label;
  for (std::size_t i = 0; i < base.rows; ++i) {
    if (ex[i] == 0) {
      live.data.insert(live.data.end(), base.Row(i), base.Row(i) + base.dim);
      ++live.rows;
      label.push_back(static_cast<std::int32_t>(i));
    }
  }
  const auto n = ars::BruteForceKnn(live, q, 10);
  for (std::size_t i = 0; i < r.ids.data.size(); ++i) {
    EXPECT_EQ(ex[r.ids.data[i]], 0);
    EXPECT_EQ(r.ids.data[i], label[n.ids.data[i]]);
    EXPECT_EQ(r.dists.data[i], n.dists.data[i]);
  }
  std::vector<std::uint8_t> bad(5, 0);
  EXPECT_THROW(ars::BruteForceKnnExcluding(base, q, 10, bad),
               std::invalid_argument);
  std::vector<std::uint8_t> all(base.rows, 1);
  EXPECT_THROW(ars::BruteForceKnnExcluding(base, q, 1, all),
               std::invalid_argument);
}

TEST(MaskedCentroid, ExcludesMaskedRows) {
  const ars::FloatMatrix m{{0, 0, 2, 2, 100, 100}, 3, 2};
  const auto c = ars::ComputeCentroid(m, {0, 0, 1});
  EXPECT_DOUBLE_EQ(c[0], 1.0);
  EXPECT_DOUBLE_EQ(c[1], 1.0);
  EXPECT_EQ(ars::ComputeCentroid(m, {}), ars::ComputeCentroid(m));
}

TEST(IndexEvolution, SaveBytesDeterministicAndResizeKeepsSearches) {
  const auto base = Ints(3000, 16, 5);
  const auto q = Ints(30, 16, 6);
  ars::HnswIndex a(16, base.rows, kP);
  ars::HnswIndex b(16, base.rows, kP);
  a.Add(base, 0, 1);
  b.Add(base, 0, 1);
  a.Save(Tmp("evo_a.bin"));
  b.Save(Tmp("evo_b.bin"));
  EXPECT_EQ(Slurp(Tmp("evo_a.bin")), Slurp(Tmp("evo_b.bin")));
  std::vector<std::vector<std::size_t>> before;
  for (std::size_t i = 0; i < q.rows; ++i) {
    before.push_back(a.Search(q.Row(i), 10, 50).labels);
  }
  a.Resize(base.rows + 500);
  for (std::size_t i = 0; i < q.rows; ++i) {
    EXPECT_EQ(a.Search(q.Row(i), 10, 50).labels, before[i]);
  }
  const auto ins = Ints(500, 16, 7);
  a.Add(ins, base.rows, 1);
  EXPECT_EQ(a.Size(), base.rows + 500);
}

TEST(IndexEvolution, LazyDeletePersistsAndIsNeverReturned) {
  const auto base = Ints(2000, 8, 8);
  ars::HnswIndex idx(8, base.rows, kP);
  idx.Add(base, 0, 1);
  std::set<std::size_t> del;
  for (std::size_t l = 0; l < base.rows; l += 7) {
    idx.MarkDeleted(l);
    del.insert(l);
  }
  EXPECT_EQ(idx.DeletedCount(), del.size());
  idx.Save(Tmp("evo_del.bin"));
  ars::HnswIndex re = ars::HnswIndex::Load(Tmp("evo_del.bin"), 8);
  EXPECT_EQ(re.DeletedCount(), del.size());
  for (std::size_t i = 0; i < 50; ++i) {
    for (const std::size_t l : re.Search(base.Row(i * 3), 10, 100).labels) {
      EXPECT_EQ(del.count(l), 0U);
    }
  }
}

TEST(StateLoading, InsertedRowsAppendedAndDeletedMaskBuilt) {
  const auto base = Ints(100, 4, 9);
  const auto ins = Ints(30, 4, 10);
  const auto q = Ints(5, 4, 11);
  ars::WriteFvecs(Tmp("st_base.fvecs"), base);
  ars::WriteFvecs(Tmp("st_ins.fvecs"), ins);
  ars::WriteFvecs(Tmp("st_q.fvecs"), q);
  std::ofstream(Tmp("st_del.csv")) << "label\n3\n104\n";
  ars::S0Config c;
  c.base_path = Tmp("st_base.fvecs");
  c.query_path = Tmp("st_q.fvecs");
  c.state.present = true;
  c.state.name = "t";
  c.state.index_path = "unused";
  c.state.inserted_vectors = Tmp("st_ins.fvecs");
  c.state.inserted_count = 20;
  c.state.deleted_labels = Tmp("st_del.csv");
  const auto d = ars::LoadS0Data(c);
  ASSERT_EQ(d.base.rows, 120U);
  EXPECT_EQ(std::vector<float>(d.base.Row(100), d.base.Row(100) + 4),
            std::vector<float>(ins.Row(0), ins.Row(0) + 4));
  ASSERT_EQ(d.excluded.size(), 120U);
  EXPECT_EQ(d.excluded[3], 1);
  EXPECT_EQ(d.excluded[104], 1);
  EXPECT_EQ(d.excluded[5], 0);
  std::ofstream(Tmp("st_del.csv")) << "label\n3\n3\n";
  EXPECT_THROW(ars::LoadS0Data(c), std::runtime_error);
}

TEST(EvolveConfig, ParsesAndValidates) {
  const std::string text = R"(experiment_name: e
dataset: {base: b, queries: q, max_base: 0, max_queries: 0}
index: {m: 16, ef_construction: 200, seed: 42, build_threads: 1}
ground_truth: {k: 100}
paths: {cache_dir: c, output_dir: o}
evolution:
  trajectory: insert
  name: id
  pool: p.fvecs
  order: o.csv
  counts: [10, 20, 40]
  state_dir: s
  verify_s0: v.bin
)";
  std::ofstream(Tmp("evo.yaml")) << text;
  const auto c = ars::LoadEvolveConfig(Tmp("evo.yaml"));
  EXPECT_EQ(c.counts, (std::vector<std::size_t>{10, 20, 40}));
  EXPECT_EQ(c.pool_path, "p.fvecs");
  std::string bad = text;
  bad.replace(bad.find("[10, 20, 40]"), 12, "[20, 10, 40]");
  std::ofstream(Tmp("evo_bad.yaml")) << bad;
  EXPECT_THROW(ars::LoadEvolveConfig(Tmp("evo_bad.yaml")), std::runtime_error);
}

}  // namespace
