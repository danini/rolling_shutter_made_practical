# =====================================================================
# Rolling Shutter Relative Pose Estimation Made Practical
# Top-level CMake: builds the SupeRANSAC core library (from the pinned
# submodule + RS overlay) and the synthetic C++ benchmark.
#
# The Python module (pysuperansac, needed for the real-data example) is
# built separately with `pip install ./external/superansac`.
#
# Prerequisite: run `bash rs_overlay/apply_overlay.sh` first so the RS
# solver headers and their small edits are present inside external/superansac.
# =====================================================================
cmake_minimum_required(VERSION 3.16)
project(RollingShutterMadePractical LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Portability-sensitive optimizations (opt-in) --------------------
# -march=native and -ffast-math reproduce the paper's timings but are NOT
# portable across machines. Default OFF; enable for benchmarking on the
# build host. Note: fast-math changes NaN semantics (see README caveat).
option(RS_ENABLE_NATIVE_ARCH "Build with -march=native -mtune=native" OFF)
option(RS_ENABLE_FAST_MATH   "Build with -ffast-math (non-IEEE)"       OFF)

set(SUPERANSAC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/superansac)
if(NOT EXISTS ${SUPERANSAC_ROOT}/src/superansac.cpp)
  message(FATAL_ERROR
    "external/superansac is empty. Run:\n"
    "  git submodule update --init external/superansac\n"
    "  bash rs_overlay/apply_overlay.sh")
endif()
if(NOT EXISTS ${SUPERANSAC_ROOT}/include/estimators/solver_rs_essential_matrix_seven_ac_direct.h)
  message(FATAL_ERROR
    "RS overlay not applied. Run:  bash rs_overlay/apply_overlay.sh")
endif()

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)

# ---- SupeRANSAC core library -----------------------------------------
# Mirror superansac/src/CMakeLists.txt but with paths rooted at the
# submodule so we do not pull in its Python subdirectory here.
file(GLOB_RECURSE SUPERANSAC_SOURCES
  ${SUPERANSAC_ROOT}/include/*.cc
  ${SUPERANSAC_ROOT}/include/*.cpp)

add_library(superansac_core SHARED
  ${SUPERANSAC_ROOT}/src/superansac.cpp
  ${SUPERANSAC_SOURCES})

target_include_directories(superansac_core PUBLIC
  ${SUPERANSAC_ROOT}/include
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

target_link_libraries(superansac_core
  Eigen3::Eigen
  Boost::system
  ${OpenCV_LIBS})

target_compile_options(superansac_core PRIVATE
  $<$<CONFIG:Release>:-O3>)
target_compile_definitions(superansac_core PRIVATE
  $<$<CONFIG:Release>:NDEBUG EIGEN_NO_DEBUG>)

# ---- Synthetic benchmark ---------------------------------------------
add_executable(benchmark_rs_superansac benchmark/benchmark_rs_superansac.cpp)
# The benchmark uses includes of the form "superansac/include/...", so it
# needs the directory that *contains* external/superansac on the search path.
target_include_directories(benchmark_rs_superansac PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/external
  ${SUPERANSAC_ROOT}/include)
target_link_libraries(benchmark_rs_superansac superansac_core)
target_compile_options(benchmark_rs_superansac PRIVATE
  $<$<CONFIG:Release>:-O3>)
target_compile_definitions(benchmark_rs_superansac PRIVATE
  $<$<CONFIG:Release>:NDEBUG EIGEN_NO_DEBUG>)

# ---- Optional non-portable flags applied to both targets -------------
foreach(tgt superansac_core benchmark_rs_superansac)
  if(RS_ENABLE_NATIVE_ARCH)
    target_compile_options(${tgt} PRIVATE -march=native -mtune=native)
  endif()
  if(RS_ENABLE_FAST_MATH)
    target_compile_options(${tgt} PRIVATE -ffast-math -fno-math-errno -ffp-contract=fast)
  endif()
endforeach()

message(STATUS "RS_ENABLE_NATIVE_ARCH = ${RS_ENABLE_NATIVE_ARCH}")
message(STATUS "RS_ENABLE_FAST_MATH   = ${RS_ENABLE_FAST_MATH}")
