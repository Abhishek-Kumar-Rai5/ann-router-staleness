cmake_minimum_required(VERSION 3.25)  # add_subdirectory(... SYSTEM)
project(ann_router_staleness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ARS_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(ARS_ENABLE_CLANG_TIDY "Run clang-tidy on project sources" ON)
# -march=native enables hnswlib's AVX/AVX-512 distance kernels. The effective
# flags are logged into every run's metadata, so results stay attributable.
option(ARS_NATIVE_ARCH "Compile with -march=native" ON)
if(ARS_NATIVE_ARCH)
  string(APPEND CMAKE_CXX_FLAGS " -march=native")
endif()

# clang-tidy: on by default, but degrade gracefully if not installed.
if(ARS_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set(ARS_CLANG_TIDY "${CLANG_TIDY_EXE}")
    # Point clang-tidy at the libstdc++ of the GCC that actually builds the
    # project. Otherwise clang picks the newest /usr/lib/gcc/<triple>/<ver>,
    # which may lack headers (e.g. a gcc-14 runtime without libstdc++-14-dev).
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-libgcc-file-name
        OUTPUT_VARIABLE ARS_LIBGCC_FILE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      get_filename_component(ARS_GCC_INSTALL_DIR "${ARS_LIBGCC_FILE}" DIRECTORY)
      list(APPEND ARS_CLANG_TIDY
        "--extra-arg=--gcc-install-dir=${ARS_GCC_INSTALL_DIR}")
    # clang cannot parse GCC's omp.h; it needs its own (libomp-<ver>-dev).
    endif()
  else()
    message(WARNING "clang-tidy not found; static analysis disabled for this build")
  endif()
endif()

find_package(OpenMP REQUIRED)

# hnswlib: header-only, used strictly as a black box (CLAUDE.md scope rule).
add_library(hnswlib INTERFACE)
target_include_directories(hnswlib SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib)

# yaml-cpp: pinned submodule, parses the per-run YAML experiment configs.
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/yaml-cpp EXCLUDE_FROM_ALL SYSTEM)

execute_process(
  COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib rev-parse HEAD
  OUTPUT_VARIABLE ARS_HNSWLIB_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)

# Project core library.
add_library(ars_core
  src/build_info.cpp
  src/config.cpp
  src/features.cpp
  src/ground_truth.cpp
  src/hnsw_index.cpp
  src/metrics.cpp
  src/oracle.cpp
  src/policy.cpp
  src/query_split.cpp
  src/run_metadata.cpp
  src/s0_setup.cpp
  src/vecs_io.cpp
)
target_include_directories(ars_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ars_core PUBLIC hnswlib OpenMP::OpenMP_CXX
  PRIVATE yaml-cpp::yaml-cpp)
target_compile_options(ars_core PRIVATE -Wall -Wextra -Wpedantic)
string(TOUPPER "${CMAKE_BUILD_TYPE}" ARS_BUILD_TYPE_UPPER)
target_compile_definitions(ars_core PRIVATE
  ARS_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  ARS_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
  ARS_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  ARS_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${ARS_BUILD_TYPE_UPPER}}"
  ARS_HNSWLIB_COMMIT="${ARS_HNSWLIB_COMMIT}"
  ARS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
if(ARS_CLANG_TIDY)
  set_target_properties(ars_core PROPERTIES CXX_CLANG_TIDY "${ARS_CLANG_TIDY}")
endif()

# Experiment drivers (one per phase), each driven by a YAML config.
add_executable(ars_static_sweep apps/ars_static_sweep.cpp)
add_executable(ars_oracle apps/ars_oracle.cpp)
add_executable(ars_features apps/ars_features.cpp)
add_executable(ars_eval_policy apps/ars_eval_policy.cpp)
add_executable(ars_make_query_subset apps/ars_make_query_subset.cpp)
add_executable(ars_evolve apps/ars_evolve.cpp)
add_executable(ars_check_state apps/ars_check_state.cpp)

foreach(app ars_static_sweep ars_oracle ars_features ars_eval_policy
            ars_make_query_subset ars_evolve ars_check_state)
  target_link_libraries(${app} PRIVATE ars_core)
  target_compile_options(${app} PRIVATE -Wall -Wextra -Wpedantic)
  if(ARS_CLANG_TIDY)
    set_target_properties(${app} PROPERTIES CXX_CLANG_TIDY "${ARS_CLANG_TIDY}")
  endif()
endforeach()

# Experiment C (design_doc.md Addendum C1): independent DARTH re-implementation.
# Needs LightGBM v4.6.0 built as a static library in third_party/LightGBM:
#   cmake -S third_party/LightGBM -B third_party/LightGBM/build \
#         -DBUILD_STATIC_LIB=ON -DBUILD_CLI=OFF -DCMAKE_BUILD_TYPE=Release
#   cmake --build third_party/LightGBM/build -j
set(ARS_LIGHTGBM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/LightGBM)
if(EXISTS ${ARS_LIGHTGBM_DIR}/lib_lightgbm.a)
  add_library(lightgbm_static STATIC IMPORTED)
  set_target_properties(lightgbm_static PROPERTIES
    IMPORTED_LOCATION ${ARS_LIGHTGBM_DIR}/lib_lightgbm.a)
  target_include_directories(lightgbm_static SYSTEM INTERFACE
    ${ARS_LIGHTGBM_DIR}/include)
  target_link_libraries(lightgbm_static INTERFACE OpenMP::OpenMP_CXX)
  add_library(ars_darth src/darth.cpp)
  target_link_libraries(ars_darth PUBLIC ars_core PRIVATE lightgbm_static)
  target_compile_options(ars_darth PRIVATE -Wall -Wextra -Wpedantic)
  add_executable(exp_c_darth apps/exp_c_darth.cpp)
  target_link_libraries(exp_c_darth PRIVATE ars_darth yaml-cpp::yaml-cpp)
  target_compile_options(exp_c_darth PRIVATE -Wall -Wextra -Wpedantic)
  if(ARS_CLANG_TIDY)
    set_target_properties(ars_darth exp_c_darth PROPERTIES
      CXX_CLANG_TIDY "${ARS_CLANG_TIDY}")
  endif()
  set(ARS_HAVE_DARTH ON)
else()
  message(STATUS "LightGBM static lib not found; Experiment C targets disabled")
  set(ARS_HAVE_DARTH OFF)
endif()

if(ARS_BUILD_TESTS)
  enable_testing()
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
  add_subdirectory(tests)
endif()
