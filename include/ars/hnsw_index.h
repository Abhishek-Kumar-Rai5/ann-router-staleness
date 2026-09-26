#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ars/vecs_io.h"
#include "hnswlib/hnswlib.h"

namespace ars {

// Wraps hnswlib's L2 distance so we can count distance calls per thread.
// hnswlib's built-in counter is shared across threads and also counts
// neighbours it skips, so it can't give an exact per-query number.
class CountingL2Space : public hnswlib::SpaceInterface<float> {
 public:
  explicit CountingL2Space(std::size_t dim);

  std::size_t get_data_size() override;
  hnswlib::DISTFUNC<float> get_dist_func() override;
  void* get_dist_func_param() override;

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
  // A fixed seed only gives a bit-identical graph if inserts are
  // single-threaded.
  std::uint64_t seed = 0;
};

struct SearchResult {
  std::vector<std::size_t> labels;
  std::vector<float> dists;
  std::uint64_t distance_computations = 0;
};

// hnswlib is treated as a black box: we only call its public API.
class HnswIndex {
 public:
  HnswIndex(std::size_t dim, std::size_t max_elements, const HnswParams& p);
  static HnswIndex Load(const std::string& path, std::size_t dim);

  // More than one thread is faster, but then the graph depends on thread
  // timing and is no longer reproducible.
  void Add(const FloatMatrix& vectors, std::size_t first_label,
           int num_threads);
  void Save(const std::string& path);

  // This sets ef on the whole index, so don't run it concurrently with a
  // different ef.
  SearchResult Search(const float* query, std::size_t k, std::size_t ef);

  // Gives exactly the same per-query results as calling Search() in a loop,
  // whatever the thread count.
  std::vector<SearchResult> SearchBatch(const FloatMatrix& queries,
                                        std::size_t k, std::size_t ef,
                                        int num_threads);

  SearchResult SearchWithStopCondition(
      const float* query, std::size_t k,
      hnswlib::BaseSearchStopCondition<float>& cond) const;

  void Resize(std::size_t new_max_elements);
  // Lazy delete: the node stays in the graph, searches just skip it.
  void MarkDeleted(std::size_t label);
  [[nodiscard]] std::size_t DeletedCount() const;

  [[nodiscard]] std::size_t Size() const;
  [[nodiscard]] std::size_t Dim() const { return dim_; }

 private:
  HnswIndex() = default;
  [[nodiscard]] SearchResult SearchAtCurrentEf(const float* query,
                                               std::size_t k) const;

  std::size_t dim_ = 0;
  std::unique_ptr<CountingL2Space> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_;
};

}  // namespace ars
