cmake_minimum_required(VERSION 3.25)

project(SciPP
  VERSION 1.5.0
  DESCRIPTION "Modern C++20 port of SciPy, built on NumPP"
  LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Standard / options
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Backend feature flags — all OFF by default (mobile/CPU-only is the baseline).
option(SCIPP_WITH_BLAS   "Enable BLAS backend (via NumPP)"   OFF)
option(SCIPP_WITH_LAPACK "Enable LAPACK backend (via NumPP)" OFF)
option(SCIPP_WITH_CUDA   "Enable CUDA backend (via NumPP)"   OFF)
option(SCIPP_WITH_OPENCL "Enable OpenCL backend (via NumPP)" OFF)
option(SCIPP_WITH_METAL  "Enable Metal backend (via NumPP)"  OFF)
option(SCIPP_BUILD_TESTS "Build the test suite"              ON)

# -----------------------------------------------------------------------------
# NumPP dependency — pinned, resolved via find_package(NumPP CONFIG).
# In production this comes from a Conan/vcpkg package; locally, build+install
# NumPP and point SciPP at the prefix (see scripts/bootstrap_numpp.sh).
# -----------------------------------------------------------------------------
set(SCIPP_NUMPP_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/.deps/numpp"
    CACHE PATH "Install prefix of a locally-built NumPP (find_package hint)")
set(NumPP_DIR "${SCIPP_NUMPP_PREFIX}/lib/cmake/NumPP" CACHE PATH "" FORCE)

# Resolve only the pinned install prefix, ignoring any stale build-tree package
# registered by NumPP's own build in ~/.cmake/packages.
find_package(NumPP 1.6.0 CONFIG REQUIRED
  NO_CMAKE_PACKAGE_REGISTRY
  NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)
message(STATUS "SciPP: found NumPP ${NumPP_VERSION}")

# GPU/accel flag propagation: a requested SciPP GPU backend requires a NumPP
# package built with the matching backend. Fail fast otherwise.
foreach(_b BLAS LAPACK CUDA OPENCL METAL)
  if(SCIPP_WITH_${_b} AND DEFINED NUMPP_WITH_${_b} AND NOT NUMPP_WITH_${_b})
    message(FATAL_ERROR
      "SCIPP_WITH_${_b}=ON but the resolved NumPP package was built with "
      "NUMPP_WITH_${_b}=OFF. Rebuild/point to a NumPP variant with that backend.")
  endif()
endforeach()

# -----------------------------------------------------------------------------
# Generated headers
# -----------------------------------------------------------------------------
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/scipp/config.hpp" @ONLY)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/scipp/version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/scipp/version.hpp" @ONLY)

add_subdirectory(src)

if(SCIPP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
