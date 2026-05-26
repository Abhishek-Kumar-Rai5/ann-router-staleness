#include "ars/darth.h"

#include <LightGBM/c_api.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace ars::darth {

Features MakeFeatures(int step, int dists, int inserts, float first_nn_dist,
                      std::vector<float> kbest) {
  const std::size_t k = kbest.size();
  if (k == 0) {
    throw std::invalid_argument("MakeFeatures: empty k-best");
  }
  std::sort(kbest.begin(), kbest.end());
  // Single-precision accumulation and population variance (C1.9).
  float sum = 0.0F;
  float sum_sq = 0.0F;
  for (const float d : kbest) {
    sum += d;
    sum_sq += d * d;
  }
  const float mean = sum / static_cast<float>(k);
  const float var = sum_sq / static_cast<float>(k) - mean * mean;
  auto pct = [&](float p) {
    return static_cast<double>(
        kbest[static_cast<std::size_t>(p * static_cast<float>(k - 1))]);
  };
  return {static_cast<double>(step),
          static_cast<double>(dists),
          static_cast<double>(inserts),
          static_cast<double>(first_nn_dist),
          static_cast<double>(kbest.front()),
          static_cast<double>(kbest.back()),
          static_cast<double>(mean),
          static_cast<double>(var),
          pct(0.25F),
          pct(0.50F),
          pct(0.75F)};
}

Predictor::Predictor(const std::string& model_path) {
  int iters = 0;
  BoosterHandle h = nullptr;
  if (LGBM_BoosterCreateFromModelfile(model_path.c_str(), &iters, &h) != 0) {
    throw std::runtime_error("cannot load LightGBM model " + model_path + ": " +
                             LGBM_GetLastError());
  }
  booster_ = h;
  if (NumFeatures() != static_cast<int>(kNumFeatures)) {
    throw std::runtime_error("model feature count != 11");
  }
}

Predictor::~Predictor() {
  if (booster_ != nullptr) {
    LGBM_BoosterFree(static_cast<BoosterHandle>(booster_));
  }
}

int Predictor::NumFeatures() const {
  int n = 0;
  LGBM_BoosterGetNumFeature(static_cast<BoosterHandle>(booster_), &n);
  return n;
}

double Predictor::Predict(const Features& x) const {
  double out = 0.0;
  int64_t len = 0;
  if (LGBM_BoosterPredictForMatSingleRow(
          static_cast<BoosterHandle>(booster_), x.data(), C_API_DTYPE_FLOAT64,
          static_cast<int32_t>(kNumFeatures), 1, C_API_PREDICT_NORMAL, 0, -1,
          "num_threads=1", &len, &out) != 0) {
    throw std::runtime_error(std::string("LightGBM predict failed: ") +
                             LGBM_GetLastError());
  }
  return out;
}

StopCondition::StopCondition(const Params& p, Mode mode,
                             const Predictor* predictor,
                             const std::int32_t* gt_ids)
    : p_(p), mode_(mode), predictor_(predictor), gt_ids_(gt_ids), pi_(p.ipi) {
  if (p_.ef < p_.k) {
    throw std::invalid_argument("StopCondition: ef < k");
  }
  if (mode_ == Mode::kPredict &&
      (predictor_ == nullptr || p_.ipi < 1 || p_.mpi < 1)) {
    throw std::invalid_argument("StopCondition: predict mode misconfigured");
  }
  if (mode_ == Mode::kTrace && gt_ids_ == nullptr) {
    throw std::invalid_argument("StopCondition: trace mode needs ground truth");
  }
  kbest_.reserve(p_.k + 1);
}

Features StopCondition::CurrentFeatures() const {
  std::vector<float> d;
  d.reserve(kbest_.size());
  for (const auto& e : kbest_) {
    d.push_back(e.first);
  }
  return MakeFeatures(Step(), dists_, inserts_, first_nn_, std::move(d));
}

std::vector<hnswlib::labeltype> StopCondition::KBestLabels() const {
  auto v = kbest_;
  std::sort(v.begin(), v.end());
  std::vector<hnswlib::labeltype> out;
  out.reserve(v.size());
  for (const auto& e : v) {
    out.push_back(e.second);
  }
  return out;
}

// The check (or observation) due after the most recent distance, evaluated
// after that point's possible insertion into the result.
void StopCondition::RunDueCheck() {
  if (!due_) {
    return;
  }
  due_ = false;
  const bool full = inserts_ >= static_cast<int>(p_.k);
  if (mode_ == Mode::kTrace) {
    if (full) {
      std::size_t hit = 0;
      for (const auto& e : kbest_) {
        for (std::size_t j = 0; j < p_.k; ++j) {
          if (static_cast<hnswlib::labeltype>(gt_ids_[j]) == e.second) {
            ++hit;
            break;
          }
        }
      }
      obs_.push_back({CurrentFeatures(),
                      static_cast<double>(hit) / static_cast<double>(p_.k)});
    }
    return;
  }
  // kPredict
  ++calls_;
  since_check_ = 0;
  if (!full) {
    last_pred_ = 0.0;  // fewer than k insertions: no inference, pi unchanged
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  double rp = predictor_->Predict(CurrentFeatures());
  predictor_s_ +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  ++inferences_;
  rp = std::clamp(rp, 0.0, 1.0);
  last_pred_ = rp;
  if (rp >= p_.target_recall) {
    stopped_ = true;
    dists_at_decision_ = dists_;
    return;
  }
  pi_ = static_cast<int>(p_.mpi + (p_.ipi - p_.mpi) * (p_.target_recall - rp));
  pi_ = std::max(pi_, 1);
}

void StopCondition::add_point_to_result(hnswlib::labeltype label,
                                        const void* /*datapoint*/, float dist) {
  ++result_size_;
  if (first_nn_ < 0.0F) {
    first_nn_ = dist;
  }
  if (kbest_.size() < p_.k || dist < kbest_.front().first) {
    ++inserts_;
    kbest_.emplace_back(dist, label);
    std::push_heap(kbest_.begin(), kbest_.end());
    if (kbest_.size() > p_.k) {
      std::pop_heap(kbest_.begin(), kbest_.end());
      kbest_.pop_back();
    }
  }
  RunDueCheck();
}

void StopCondition::remove_point_from_result(hnswlib::labeltype /*label*/,
                                             const void* /*datapoint*/,
                                             float /*dist*/) {
  --result_size_;
}

bool StopCondition::should_stop_search(float candidate_dist,
                                       float lower_bound) {
  RunDueCheck();
  if (stopped_) {
    return true;
  }
  // hnswlib's own rules: searchKnn without deletions stops on the distance
  // bound alone; the general path also requires a full result set.
  const bool stop = p_.no_deletions
                        ? candidate_dist > lower_bound
                        : candidate_dist > lower_bound && result_size_ == p_.ef;
  if (!stop) {
    ++pops_;
  }
  return stop;
}

bool StopCondition::should_consider_candidate(float candidate_dist,
                                              float lower_bound) {
  RunDueCheck();  // previous distance's check, if its point was not added
  ++dists_;
  if (stopped_) {
    return false;  // result frozen at the decision point
  }
  ++since_check_;
  if (mode_ == Mode::kTrace ||
      (mode_ == Mode::kPredict && since_check_ == pi_)) {
    due_ = true;
  }
  const bool consider = result_size_ < p_.ef || lower_bound > candidate_dist;
  if (!consider) {
    RunDueCheck();  // nothing will be added for this distance
  }
  return consider;
}

bool StopCondition::should_remove_extra() { return result_size_ > p_.ef; }

void StopCondition::filter_results(
    std::vector<std::pair<float, hnswlib::labeltype>>& /*candidates*/) {
  RunDueCheck();  // the last distance of the search
}

}  // namespace ars::darth
