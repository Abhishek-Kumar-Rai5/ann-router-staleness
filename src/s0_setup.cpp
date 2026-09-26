#include "ars/s0_setup.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "ars/run_metadata.h"

namespace fs = std::filesystem;

namespace ars {
namespace {

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::size_t RowLimit(std::size_t configured) {
  return configured == 0 ? kAllRows : configured;
}

std::string FileIdentity(const std::string& path, std::size_t rows) {
  return path + "|" + std::to_string(fs::file_size(path)) + "|" +
         std::to_string(rows);
}

}  // namespace

S0Data LoadS0Data(const S0Config& cfg) {
  S0Data d{ReadFvecs(cfg.base_path, RowLimit(cfg.max_base)),
           ReadFvecs(cfg.query_path, RowLimit(cfg.max_queries)),
           {}};
  if (!cfg.state.present) {
    return d;
  }
  if (cfg.state.inserted_count > 0) {
    const FloatMatrix ins =
        ReadFvecs(cfg.state.inserted_vectors, cfg.state.inserted_count);
    if (ins.rows != cfg.state.inserted_count || ins.dim != d.base.dim) {
      throw std::runtime_error("state: inserted vectors shorter than count");
    }
    d.base.data.insert(d.base.data.end(), ins.data.begin(), ins.data.end());
    d.base.rows += ins.rows;
  }
  if (!cfg.state.deleted_labels.empty()) {
    d.excluded.assign(d.base.rows, 0);
    std::ifstream in(cfg.state.deleted_labels);
    std::string line;
    if (!in || !std::getline(in, line) || line != "label") {
      throw std::runtime_error("state: bad deleted_labels file");
    }
    while (std::getline(in, line)) {
      const std::size_t l = std::stoul(line);
      if (l >= d.base.rows || d.excluded[l] != 0) {
        throw std::runtime_error("state: deleted label out of range/duplicate");
      }
      d.excluded[l] = 1;
    }
  }
  return d;
}

static std::string StateIdentity(const S0Config& cfg) {
  if (!cfg.state.present) {
    return "S0";
  }
  std::string id = "state:" + cfg.state.name + "|" +
                   FileIdentity(cfg.state.index_path, 0) +
                   "|ins=" + std::to_string(cfg.state.inserted_count);
  if (cfg.state.inserted_count > 0) {
    id += "|" + FileIdentity(cfg.state.inserted_vectors, 0);
  }
  if (!cfg.state.deleted_labels.empty()) {
    id += "|" + FileIdentity(cfg.state.deleted_labels, 0);
  }
  return id;
}

S0GroundTruth LoadOrComputeS0GroundTruth(const S0Config& cfg,
                                         const S0Data& data) {
  // The S0 key format hasn't changed since Phase 1, so old cache files are
  // still valid. Evolved states get their own key.
  const std::string sid = StateIdentity(cfg);
  const std::string key =
      Hex64(Fnv1a64((sid == "S0" ? std::string("gt|S0|") : "gt|" + sid + "|") +
                    FileIdentity(cfg.base_path, data.base.rows) + "|" +
                    FileIdentity(cfg.query_path, data.queries.rows) +
                    "|k=" + std::to_string(cfg.gt_k)));
  S0GroundTruth gt;
  gt.cache_path = cfg.cache_dir + "/gt_" +
                  (cfg.state.present ? "state_" + cfg.state.name : "S0") + "_" +
                  key;
  const std::string ids_path = gt.cache_path + ".ivecs";
  const std::string dists_path = gt.cache_path + ".fvecs";
  const auto t0 = Clock::now();
  if (fs::exists(ids_path) && fs::exists(dists_path)) {
    gt.knn.ids = ReadIvecs(ids_path);
    gt.knn.dists = ReadFvecs(dists_path);
    gt.from_cache = true;
    if (gt.knn.ids.rows != data.queries.rows || gt.knn.ids.dim != cfg.gt_k) {
      throw std::runtime_error("cached ground truth has wrong shape: " +
                               ids_path);
    }
  } else {
    gt.knn = BruteForceKnnExcluding(data.base, data.queries, cfg.gt_k,
                                    data.excluded);
    WriteIvecs(ids_path, gt.knn.ids);
    WriteFvecs(dists_path, gt.knn.dists);
  }
  gt.seconds = SecondsSince(t0);
  return gt;
}

S0Index LoadOrBuildS0Index(const S0Config& cfg, const S0Data& data) {
  if (cfg.state.present) {
    const auto t0 = Clock::now();
    S0Index s{HnswIndex::Load(cfg.state.index_path, data.base.dim),
              cfg.state.index_path, true, 0.0};
    if (s.index.Size() != data.base.rows) {
      throw std::runtime_error("state index element count != labelled vectors");
    }
    std::size_t n_del = 0;
    for (const std::uint8_t e : data.excluded) {
      n_del += e;
    }
    if (s.index.DeletedCount() != n_del) {
      throw std::runtime_error("state index deleted count != deleted_labels");
    }
    s.seconds = SecondsSince(t0);
    return s;
  }
  const std::string key =
      Hex64(Fnv1a64("hnsw|S0|" + FileIdentity(cfg.base_path, data.base.rows) +
                    "|m=" + std::to_string(cfg.index.m) +
                    "|efc=" + std::to_string(cfg.index.ef_construction) +
                    "|seed=" + std::to_string(cfg.index.seed) +
                    "|threads=" + std::to_string(cfg.build_threads)));
  const std::string path = cfg.cache_dir + "/hnsw_S0_" + key + ".bin";
  const auto t0 = Clock::now();
  if (fs::exists(path)) {
    S0Index s{HnswIndex::Load(path, data.base.dim), path, true, 0.0};
    s.seconds = SecondsSince(t0);
    return s;
  }
  HnswIndex index(data.base.dim, data.base.rows, cfg.index);
  index.Add(data.base, /*first_label=*/0, cfg.build_threads);
  const double seconds = SecondsSince(t0);
  index.Save(path);
  return S0Index{std::move(index), path, false, seconds};
}

}  // namespace ars
