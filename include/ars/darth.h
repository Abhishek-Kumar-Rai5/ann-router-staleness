#pragma once

// Experiment C (design_doc.md Addendum C1): an independent re-implementation of
// the DARTH early-termination mechanism (Chatzakis et al., SIGMOD'26; see
// THIRD_PARTY.md) on hnswlib's public stop-condition API. hnswlib itself is not
// modified. Kept separate from the Phase 4b, A1 and oracle code.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "hnswlib/hnswlib.h"

namespace ars::darth {

// Model input order, frozen in design_doc.md C1.9 (= DARTH training order).
inline constexpr std::size_t kNumFeatures = 11;
inline constexpr std::array<const char*, kNumFeatures> kFeatureNames = {
    "step",          "dists",         "inserts",      "first_nn_dist",
    "nn_dist",       "furthest_dist", "avg_dist",     "variance",
    "percentile_25", "percentile_50", "percentile_75"};
using Features = std::array<double, kNumFeatures>;

// Features from the search counters and the current k-best distances
// (any order; must hold exactly k values).
Features MakeFeatures(int step, int dists, int inserts, float first_nn_dist,
                      std::vector<float> kbest);

// LightGBM booster loaded from a text model (C API, single-row prediction).
class Predictor {
 public:
  explicit Predictor(const std::string& model_path);
  ~Predictor();
  Predictor(const Predictor&) = delete;
  Predictor& operator=(const Predictor&) = delete;
  Predictor(Predictor&&) = delete;
  Predictor& operator=(Predictor&&) = delete;
  [[nodiscard]] double Predict(const Features& x) const;
  [[nodiscard]] int NumFeatures() const;

 private:
  void* booster_ = nullptr;
};

enum class Mode {
  kPlain,    // no prediction: must reproduce hnswlib searchKnn exactly
  kTrace,    // plain search + one observation after every distance
  kPredict,  // DARTH early termination
};

struct Params {
  std::size_t k = 10;
  std::size_t ef = 0;          // hnswlib beam (search cap)
  bool no_deletions = true;    // emulate searchKnn's no-deletion stop rule
  double target_recall = 0.0;  // kPredict
  int ipi = 0;                 // kPredict: initial / maximum interval
  int mpi = 0;                 // kPredict: minimum interval
};

struct Observation {
  Features x;
  double recall;  // recall@k by id of the current k-best
};

// Per-query stop condition; construct a fresh one for every query.
class StopCondition : public hnswlib::BaseSearchStopCondition<float> {
 public:
  StopCondition(const Params& p, Mode mode, const Predictor* predictor,
                const std::int32_t* gt_ids);  // gt_ids: kTrace only

  void add_point_to_result(hnswlib::labeltype label, const void* datapoint,
                           float dist) override;
  void remove_point_from_result(hnswlib::labeltype label, const void* datapoint,
                                float dist) override;
  bool should_stop_search(float candidate_dist, float lower_bound) override;
  bool should_consider_candidate(float candidate_dist,
                                 float lower_bound) override;
  bool should_remove_extra() override;
  void filter_results(
      std::vector<std::pair<float, hnswlib::labeltype>>& candidates) override;

  [[nodiscard]] bool stopped_early() const { return stopped_; }
  [[nodiscard]] int dists() const { return dists_; }
  [[nodiscard]] int dists_at_decision() const { return dists_at_decision_; }
  [[nodiscard]] int predictor_calls() const { return calls_; }
  [[nodiscard]] int predictor_inferences() const { return inferences_; }
  [[nodiscard]] double predictor_seconds() const { return predictor_s_; }
  [[nodiscard]] double last_predicted() const { return last_pred_; }
  [[nodiscard]] const std::vector<Observation>& observations() const {
    return obs_;
  }
  [[nodiscard]] std::vector<hnswlib::labeltype> KBestLabels() const;

 private:
  [[nodiscard]] int Step() const { return pops_ > 0 ? pops_ - 1 : 0; }
  [[nodiscard]] Features CurrentFeatures() const;
  void RunDueCheck();

  Params p_;
  Mode mode_;
  const Predictor* predictor_;
  const std::int32_t* gt_ids_;

  std::size_t result_size_ = 0;  // size of hnswlib's result set
  std::vector<std::pair<float, hnswlib::labeltype>> kbest_;  // max-heap
  int inserts_ = 0;
  float first_nn_ = -1.0F;
  int dists_ = 0;
  int since_check_ = 0;
  int pops_ = 0;
  int pi_ = 0;
  bool due_ = false;
  bool stopped_ = false;
  int dists_at_decision_ = -1;
  int calls_ = 0;
  int inferences_ = 0;
  double predictor_s_ = 0.0;
  double last_pred_ = -1.0;
  std::vector<Observation> obs_;
};

}  // namespace ars::darth
