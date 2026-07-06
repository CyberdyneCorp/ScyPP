add_library(scipp
  special/gamma.cpp
  special/erf.cpp
  special/erf_extras.cpp
  special/expint.cpp
  special/bessel.cpp
  special/spherical_bessel.cpp
  special/sici.cpp
  special/airy.cpp
  special/elliptic.cpp
  special/orthogonal.cpp
  special/combinatorics.cpp
  special/misc.cpp
  special/hypergeometric.cpp
  special/reductions.cpp
  constants/constants.cpp
  linalg/basic.cpp
  linalg/decomp.cpp
  linalg/eigen.cpp
  linalg/matfuncs.cpp
  linalg/special.cpp
  fft/fft.cpp
  fft/realtransforms.cpp
  optimize/roots_scalar.cpp
  optimize/minimize_scalar.cpp
  optimize/minimize.cpp
  optimize/least_squares.cpp
  optimize/linprog.cpp
  optimize/nnls.cpp
  integrate/quadrature.cpp
  integrate/nested_quad.cpp
  integrate/solve_ivp.cpp
  integrate/differentiate.cpp
  integrate/solve_bvp.cpp
  interpolate/interp1d.cpp
  interpolate/splines.cpp
  interpolate/bspline.cpp
  interpolate/regular_grid.cpp
  interpolate/rbf.cpp
  interpolate/griddata.cpp
  stats/special_inc.cpp
  stats/distributions.cpp
  stats/descriptive.cpp
  stats/tests.cpp
  stats/kde.cpp
  stats/discrete.cpp
  stats/ranktests.cpp
  stats/normality.cpp
  signal/convolution.cpp
  signal/windows.cpp
  signal/waveforms.cpp
  signal/filter.cpp
  signal/design.cpp
  signal/spectral.cpp
  signal/filter2.cpp
  signal/resample.cpp
  signal/peaks.cpp
  signal/lti.cpp
  signal/dlti.cpp
  signal/spectral2.cpp
  signal/design2.cpp
  sparse/formats.cpp
  sparse/ops.cpp
  sparse/linalg.cpp
  sparse/precond.cpp
  sparse/factor.cpp
  sparse/eigen.cpp
  sparse/csgraph.cpp
  spatial/distance.cpp
  spatial/kdtree.cpp
  spatial/geometry.cpp
  spatial/rotation.cpp
  ndimage/filters.cpp
  ndimage/morphology.cpp
  ndimage/measurements.cpp
  ndimage/interpolation.cpp
  cluster/vq.cpp
  cluster/hierarchy.cpp
  odr/odr.cpp
  io/io.cpp)

add_library(scipp::scipp ALIAS scipp)

target_compile_features(scipp PUBLIC cxx_std_20)

target_include_directories(scipp PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(scipp PUBLIC numpp::numpp)

# A semver ABI version on the shared library so consumers can pin compatibility.
set_target_properties(scipp PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})

# -----------------------------------------------------------------------------
# Install / export (find_package(SciPP) support)
# -----------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS scipp EXPORT SciPPTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Ship both *.hpp and *.h so C headers reachable from the umbrella header install too.
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/scipp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
# Generated headers live in the build tree, not the source include/, so install them explicitly.
install(FILES ${PROJECT_BINARY_DIR}/generated/scipp/version.hpp
              ${PROJECT_BINARY_DIR}/generated/scipp/config.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/scipp)

install(EXPORT SciPPTargets NAMESPACE scipp:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SciPP)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/SciPPConfig.cmake.in
  ${PROJECT_BINARY_DIR}/SciPPConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SciPP)
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/SciPPConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${PROJECT_BINARY_DIR}/SciPPConfig.cmake ${PROJECT_BINARY_DIR}/SciPPConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SciPP)
