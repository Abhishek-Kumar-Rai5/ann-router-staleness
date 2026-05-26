include(GoogleTest)

add_executable(ars_tests
  test_evolution.cpp
  test_features.cpp
  test_ground_truth.cpp
  test_hnsw_index.cpp
  test_io_metrics_config.cpp
  test_oracle.cpp
  test_policy.cpp
  test_toolchain.cpp
)
target_link_libraries(ars_tests PRIVATE ars_core GTest::gtest_main)
target_compile_options(ars_tests PRIVATE -Wall -Wextra)
gtest_discover_tests(ars_tests)

if(ARS_HAVE_DARTH)
  add_executable(ars_darth_tests test_darth.cpp)
  target_link_libraries(ars_darth_tests PRIVATE ars_darth GTest::gtest_main)
  target_compile_options(ars_darth_tests PRIVATE -Wall -Wextra)
  gtest_discover_tests(ars_darth_tests)
endif()
