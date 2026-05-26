#include "ars/hnsw_index.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ars {

CountingL2Space::CountingL2Space(std::size_t dim) : inner_(dim) {
  param_.inner = inner_.get_dist_func();
  param_.inner_param = inner_.get_dist_func_param();
}

std::size_t CountingL2Space::get_data_size() { return inner_.get_data_size(); }

hnswlib::DISTFUNC<float> CountingL2Space::get_dist_func() {
  return &CountingDistance;
}

void* CountingL2Space::get_dist_func_param() { return &param_; }

std::uint64_t& CountingL2Space::ThreadCount() {
  thread_local std::uint64_t count = 0;
  return count;
}

float CountingL2Space::CountingDistance(const void* a, const void* b,
                                        const void* p) {
  ++ThreadCount();
  const auto* param = static_cast<const Param*>(p);
  return param->inner(a, b, param->inner_param);
}

HnswIndex::HnswIndex(std::size_t dim, std::size_t max_elements,
                     const HnswParams& p)
    : dim_(dim), space_(std::make_unique<CountingL2Space>(dim)) {
  hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      space_.get(), max_elements, p.m, p.ef_construction, p.seed);
}

HnswIndex HnswIndex::Load(const std::string& path, std::size_t dim) {
  HnswIndex index;
  index.dim_ = dim;
  index.space_ = std::make_unique<CountingL2Space>(dim);
  index.hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      index.space_.get(), path);
  return index;
}

void HnswIndex::Add(const FloatMatrix& vectors, std::size_t first_label,
                    int num_threads) {
  if (vectors.dim != dim_) {
    throw std::invalid_argument("HnswIndex::Add: dim mismatch");
  }
  const auto n = static_cast<std::int64_t>(vectors.rows);
  if (num_threads <= 1) {
    for (std::int64_t i = 0; i < n; ++i) {
      hnsw_->addPoint(vectors.Row(i), first_label + i);
    }
    return;
  }
#pragma omp parallel for schedule(dynamic, 256) num_threads(num_threads)
  for (std::int64_t i = 0; i < n; ++i) {
    hnsw_->addPoint(vectors.Row(i), first_label + i);
  }
}

void HnswIndex::Save(const std::string& path) { hnsw_->saveIndex(path); }

SearchResult HnswIndex::Search(const float* query, std::size_t k,
                               std::size_t ef) {
  hnsw_->setEf(ef);
  return SearchAtCurrentEf(query, k);
}

std::vector<SearchResult> HnswIndex::SearchBatch(const FloatMatrix& queries,
                                                 std::size_t k, std::size_t ef,
                                                 int num_threads) {
  if (queries.dim != dim_) {
    throw std::invalid_argument("HnswIndex::SearchBatch: dim mismatch");
  }
  hnsw_->setEf(ef);
  std::vector<SearchResult> out(queries.rows);
  const auto n = static_cast<std::int64_t>(queries.rows);
#pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads)
  for (std::int64_t q = 0; q < n; ++q) {
    out[q] = SearchAtCurrentEf(queries.Row(q), k);
  }
  return out;
}

SearchResult HnswIndex::SearchAtCurrentEf(const float* query,
                                          std::size_t k) const {
  std::uint64_t& counter = CountingL2Space::ThreadCount();
  counter = 0;
  auto heap = hnsw_->searchKnn(query, k);

  SearchResult r;
  r.distance_computations = counter;
  r.labels.resize(heap.size());
  r.dists.resize(heap.size());
  // hnswlib returns a max-heap; fill from the back to get ascending order.
  for (std::size_t i = heap.size(); i-- > 0;) {
    r.dists[i] = heap.top().first;
    r.labels[i] = heap.top().second;
    heap.pop();
  }
  return r;
}

SearchResult HnswIndex::SearchWithStopCondition(
    const float* query, std::size_t k,
    hnswlib::BaseSearchStopCondition<float>& cond) const {
  std::uint64_t& counter = CountingL2Space::ThreadCount();
  counter = 0;
  auto res = hnsw_->searchStopConditionClosest(query, cond);
  // Mirror searchKnn + SearchAtCurrentEf exactly: hnswlib returns the result
  // heap in pop order (res[0] popped last), and searchKnn keeps the k entries
  // popped last; those k are then ordered by (distance, label). This only
  // matters for equidistant (duplicate) vectors.
  const std::size_t n = std::min(k, res.size());
  std::sort(res.begin(), res.begin() + static_cast<std::ptrdiff_t>(n));
  SearchResult r;
  r.distance_computations = counter;
  r.labels.resize(n);
  r.dists.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    r.dists[i] = res[i].first;
    r.labels[i] = res[i].second;
  }
  return r;
}

void HnswIndex::Resize(std::size_t new_max_elements) {
  hnsw_->resizeIndex(new_max_elements);
}

void HnswIndex::MarkDeleted(std::size_t label) { hnsw_->markDelete(label); }

std::size_t HnswIndex::DeletedCount() const { return hnsw_->num_deleted_; }

std::size_t HnswIndex::Size() const { return hnsw_->getCurrentElementCount(); }

}  // namespace ars
