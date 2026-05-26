#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ars/vecs_io.h"
#include "hnswlib/hnswlib.h"

namespace ars {

// L2 space that delegates to hnswlib's own (SIMD) L2 distance and counts
// every distance evaluation into a thread-local counter. hnswlib's built-in
// metric_distance_computations is a shared atomic that also counts neighbours
// skipped as already-visited, so it is neither per-query nor exact; this
// wrapper is both, without modifying hnswlib.
class CountingL2Space : public hnswlib::SpaceInterface<float> {
 public:
  explicit CountingL2Space(std::size_t dim);

  std::size_t get_data_size() override;
  hnswlib::DISTFUNC<float> get_dist_func() override;
  void* get_dist_func_param() override;

  // Distance evaluations performed by the calling thread since its last reset.
  static std::uint64_t& ThreadCount();

 private:
  struct Param {
    hnswlib::DISTFUNC<float> inner;
    void* inner_param;
  };
  static float CountingDistance(const void* a, const void* b, const void* p);

  hnswlib::L2Space inner_;
  Param param_{};
};

struct HnswParams {
  std::size_t m = 16;
  std::size_t ef_construction = 200;
  // Seeds hnswlib's level-assignment RNG. The build is only bit-reproducible
  // for a fixed seed when inserted single-threaded (see Add()).
  std::uint64_t seed = 0;
};

struct SearchResult {
  std::vector<std::size_t> labels;  // ascending distance
  std::vector<float> dists;
  std::uint64_t distance_computations = 0;
};

// Thin owner of an hnswlib index. hnswlib is used strictly as a black box:
// only its public insert / search / save / load API is called.
class HnswIndex {
 public:
  HnswIndex(std::size_t dim, std::size_t max_elements, const HnswParams& p);
  static HnswIndex Load(const std::string& path, std::size_t dim);

  // Inserts vectors.Row(i) with label first_label + i. num_threads > 1 is
  // faster but makes the graph depend on thread scheduling (non-reproducible).
  void Add(const FloatMatrix& vectors, std::size_t first_label,
           int num_threads);
  void Save(const std::string& path);

  // Top-k search with the given efSearch (hnswlib uses max(ef, k)). Sets the
  // index-wide ef, so calls must not run concurrently with different ef.
  SearchResult Search(const float* query, std::size_t k, std::size_t ef);

  // Searches every row of `queries` at one efSearch, parallel over queries.
  // ef is set once before the parallel region, so this is race-free; each
  // query's result (labels, distances, distance count) is identical to
  // Search() on it, independent of num_threads. No timing is recorded.
  std::vector<SearchResult> SearchBatch(const FloatMatrix& queries,
                                        std::size_t k, std::size_t ef,
                                        int num_threads);

  // Base-layer search controlled by `cond` via hnswlib's public
  // searchStopConditionClosest (Experiment C, design_doc.md C1). Returns the
  // first k results in ascending distance and the exact distance count of the
  // whole search (upper layers included), as Search() does. Thread-safe for
  // distinct `cond` objects.
  SearchResult SearchWithStopCondition(
      const float* query, std::size_t k,
      hnswlib::BaseSearchStopCondition<float>& cond) const;

  // Public hnswlib API only: grow capacity (no graph change, no RNG use);
  // lazily delete a label (searches skip it; the node stays in the graph).
  void Resize(std::size_t new_max_elements);
  void MarkDeleted(std::size_t label);
  [[nodiscard]] std::size_t DeletedCount() const;

  [[nodiscard]] std::size_t Size() const;
  [[nodiscard]] std::size_t Dim() const { return dim_; }

 private:
  HnswIndex() = default;
  // searchKnn is const and thread-safe in hnswlib; only setEf mutates.
  [[nodiscard]] SearchResult SearchAtCurrentEf(const float* query,
                                               std::size_t k) const;

  std::size_t dim_ = 0;
  std::unique_ptr<CountingL2Space> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_;
};

}  // namespace ars
