// Phase 5' (design_doc.md Addendum A1): produce nested evolved index states.
//
// insert: rebuild S0 in-process exactly as the cached S0 was built (same
//   config, single-threaded, same capacity), save it and require it to be
//   byte-identical to the cached S0 file, then resize (public API) and insert
//   pool vectors in the configured order; after each count, save the state.
//   Inserted labels continue after the base: N0, N0+1, ...
// delete: load the cached S0 and lazily delete base labels in the configured
//   order (markDelete); after each count, save the state and its deleted list.
// hnswlib's level RNG is never touched: insertion continues the seeded stream
// of the in-process build.
//
// Usage: ars_evolve <config.yaml>

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "ars/build_info.h"
#include "ars/config.h"
#include "ars/hnsw_index.h"
#include "ars/run_metadata.h"
#include "ars/s0_setup.h"
#include "ars/vecs_io.h"

namespace fs = std::filesystem;

namespace {

std::vector<std::size_t> ReadOrder(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  if (!in || !std::getline(in, line) || line != "row") {
    throw std::runtime_error("bad order file " + path);
  }
  std::vector<std::size_t> out;
  while (std::getline(in, line)) {
    out.push_back(std::stoul(line));
  }
  return out;
}

bool FilesIdentical(const std::string& a, const std::string& b) {
  std::ifstream fa(a, std::ios::binary);
  std::ifstream fb(b, std::ios::binary);
  if (!fa || !fb) {
    return false;
  }
  return std::equal(
      std::istreambuf_iterator<char>(fa), std::istreambuf_iterator<char>(),
      std::istreambuf_iterator<char>(fb), std::istreambuf_iterator<char>());
}

ars::FloatMatrix Rows(const ars::FloatMatrix& m,
                      const std::vector<std::size_t>& rows, std::size_t begin,
                      std::size_t end) {
  ars::FloatMatrix out{std::vector<float>((end - begin) * m.dim), end - begin,
                       m.dim};
  for (std::size_t i = begin; i < end; ++i) {
    if (rows[i] >= m.rows) {
      throw std::runtime_error("order row out of range");
    }
    std::copy(m.Row(rows[i]), m.Row(rows[i]) + m.dim, out.Row(i - begin));
  }
  return out;
}

ars::FloatMatrix Slice(const ars::FloatMatrix& m, std::size_t begin,
                       std::size_t end) {
  return ars::FloatMatrix{
      std::vector<float>(m.Row(begin), m.Row(begin) + (end - begin) * m.dim),
      end - begin, m.dim};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <config.yaml>\n";
    return 2;
  }
  try {
    const ars::EvolveConfig cfg = ars::LoadEvolveConfig(argv[1]);
    const std::string exp_id = ars::MakeExperimentId(cfg.raw_yaml);
    const fs::path out_dir = fs::path(cfg.output_dir) / exp_id;
    fs::create_directories(out_dir);
    fs::create_directories(cfg.state_dir);
    std::ofstream(out_dir / "config.yaml") << cfg.raw_yaml;
    const ars::S0Data data = ars::LoadS0Data(cfg);
    const std::size_t n0 = data.base.rows;
    const std::vector<std::size_t> order = ReadOrder(cfg.order_path);
    if (order.size() < cfg.counts.back()) {
      throw std::runtime_error("order file shorter than the largest count");
    }
    std::string states_json = "[";
    bool s0_identical = false;
    const auto t0 = std::chrono::steady_clock::now();

    if (cfg.trajectory == "insert") {
      ars::HnswIndex index(data.base.dim, n0, cfg.index);
      index.Add(data.base, 0, 1);
      const std::string snap = cfg.state_dir + "/S0_inprocess_check.bin";
      index.Save(snap);
      s0_identical = FilesIdentical(snap, cfg.verify_s0);
      fs::remove(snap);
      if (!s0_identical) {
        throw std::runtime_error(
            "STOP: in-process S0 is not byte-identical to " + cfg.verify_s0);
      }
      std::cerr << "[ars] in-process S0 byte-identical to " << cfg.verify_s0
                << "\n";
      index.Resize(n0 + cfg.counts.back());
      const ars::FloatMatrix pool = ars::ReadFvecs(cfg.pool_path);
      const ars::FloatMatrix ins = Rows(pool, order, 0, cfg.counts.back());
      const std::string ins_path =
          cfg.state_dir + "/" + cfg.trajectory_name + "_inserted.fvecs";
      ars::WriteFvecs(ins_path, ins);
      std::size_t prev = 0;
      for (const std::size_t c : cfg.counts) {
        const ars::FloatMatrix step = Slice(ins, prev, c);
        index.Add(step, n0 + prev, 1);
        const std::string p = cfg.state_dir + "/" + cfg.trajectory_name + "_" +
                              std::to_string(c) + ".bin";
        index.Save(p);
        states_json += std::string(prev == 0 ? "" : ", ") +
                       "{\"count\": " + std::to_string(c) + ", \"index\": \"" +
                       p + "\", \"inserted_vectors\": \"" + ins_path + "\"}";
        std::cerr << "[ars] " << cfg.trajectory_name << " state " << c
                  << " saved (" << index.Size() << " elements)\n";
        prev = c;
      }
    } else {
      ars::HnswIndex index = ars::HnswIndex::Load(cfg.verify_s0, data.base.dim);
      if (index.Size() != n0) {
        throw std::runtime_error("cached S0 element count mismatch");
      }
      std::size_t prev = 0;
      for (const std::size_t c : cfg.counts) {
        for (std::size_t i = prev; i < c; ++i) {
          if (order[i] >= n0) {
            throw std::runtime_error("deletion label out of range");
          }
          index.MarkDeleted(order[i]);
        }
        const std::string p =
            cfg.state_dir + "/" + cfg.trajectory_name + "_" + std::to_string(c);
        index.Save(p + ".bin");
        std::ofstream del(p + "_deleted.csv");
        del << "label\n";
        for (std::size_t i = 0; i < c; ++i) {
          del << order[i] << "\n";
        }
        states_json += std::string(prev == 0 ? "" : ", ") +
                       "{\"count\": " + std::to_string(c) + ", \"index\": \"" +
                       p + ".bin\", \"deleted_labels\": \"" + p +
                       "_deleted.csv\"}";
        std::cerr << "[ars] " << cfg.trajectory_name << " state " << c
                  << " saved (" << index.DeletedCount() << " deleted)\n";
        prev = c;
      }
    }
    states_json += "]";
    const ars::BuildInfo bi = ars::GetBuildInfo();
    const ars::GitInfo git = ars::GetGitInfo(bi.source_dir);
    std::ofstream(out_dir / "metadata.json")
        << ars::JsonObject()
               .Str("experiment_id", exp_id)
               .Str("phase", "5prime_state_production")
               .Str("config_yaml", cfg.raw_yaml)
               .Raw("git", ars::JsonObject()
                               .Str("commit", git.commit)
                               .Bool("dirty", git.dirty)
                               .Render())
               .Str("hnswlib_commit", bi.hnswlib_commit)
               .Str("cxx_flags", bi.cxx_flags)
               .Int("hnsw_level_seed",
                    static_cast<std::int64_t>(cfg.index.seed))
               .Str("trajectory", cfg.trajectory)
               .Bool("in_process_s0_byte_identical", s0_identical)
               .Int("n0", static_cast<std::int64_t>(n0))
               .Raw("states", states_json)
               .Num("seconds", std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t0)
                                   .count())
               .Render()
        << "\n";
    std::cout << out_dir.string() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "[ars] error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
