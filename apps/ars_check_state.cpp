// Phase 5' validation (read-only): for an evolved state, search every query at
// several efSearch values and count returned labels that are lazily deleted or
// outside the state's labelled vectors. Prints one JSON line.
//
// Usage: ars_check_state <features-style state config.yaml>

#include <exception>
#include <iostream>
#include <string>

#include "ars/config.h"
#include "ars/run_metadata.h"
#include "ars/s0_setup.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <config.yaml>\n";
    return 2;
  }
  try {
    const ars::FeatureConfig cfg = ars::LoadFeatureConfig(argv[1]);
    const ars::S0Data data = ars::LoadS0Data(cfg);
    ars::S0Index s = ars::LoadOrBuildS0Index(cfg, data);
    std::size_t returned = 0;
    std::size_t deleted_returned = 0;
    std::size_t out_of_range = 0;
    for (const std::size_t ef : {10U, 100U, 1000U}) {
      const auto res = s.index.SearchBatch(data.queries, 10, ef, cfg.threads);
      for (const auto& r : res) {
        for (const std::size_t l : r.labels) {
          ++returned;
          if (l >= data.base.rows) {
            ++out_of_range;
          } else if (!data.excluded.empty() && data.excluded[l] != 0) {
            ++deleted_returned;
          }
        }
      }
    }
    std::cout << ars::JsonObject()
                     .Str("state", cfg.state.present ? cfg.state.name : "S0")
                     .Int("labels_returned",
                          static_cast<std::int64_t>(returned))
                     .Int("deleted_returned",
                          static_cast<std::int64_t>(deleted_returned))
                     .Int("out_of_range_returned",
                          static_cast<std::int64_t>(out_of_range))
                     .Int("index_deleted_count",
                          static_cast<std::int64_t>(s.index.DeletedCount()))
                     .Render()
              << "\n";
  } catch (const std::exception& e) {
    std::cerr << "[ars] error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
