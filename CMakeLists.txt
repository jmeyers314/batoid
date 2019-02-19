cmake_minimum_required(VERSION 2.8.12)
project(_batoid)
# set(CMAKE_VERBOSE_MAKEFILE True)
cmake_policy(SET CMP0063 NEW)  # Compile the static lib with hidden visibility.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# for testing vectorization.
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-vectorize -fno-slp-vectorize")
if (DEFINED ENV{CMAKE_COVER})
    set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
    include(CodeCoverage)
    APPEND_COVERAGE_COMPILER_FLAGS()
endif()

include_directories(include eigen)
add_subdirectory(pybind11)
set(PYSRC_FILES
  pysrc/asphere.cpp
  pysrc/batoid.cpp
  pysrc/bicubic.cpp
  pysrc/coating.cpp
  pysrc/coordsys.cpp
  pysrc/medium.cpp
  pysrc/obscuration.cpp
  pysrc/paraboloid.cpp
  pysrc/plane.cpp
  pysrc/polynomialSurface.cpp
  pysrc/quadric.cpp
  pysrc/ray.cpp
  pysrc/rayVector.cpp
  pysrc/sphere.cpp
  pysrc/sum.cpp
  pysrc/surface.cpp
  pysrc/table.cpp
)
set(SRC_FILES
  src/asphere.cpp
  src/batoid.cpp
  src/bicubic.cpp
  src/coating.cpp
  src/coordsys.cpp
  src/medium.cpp
  src/obscuration.cpp
  src/paraboloid.cpp
  src/plane.cpp
  src/polynomialSurface.cpp
  src/quadric.cpp
  src/ray.cpp
  src/rayVector.cpp
  src/sphere.cpp
  src/sum.cpp
  src/surface.cpp
  src/table.cpp
  src/utils.cpp
)
set(CMAKE_POSITION_INDEPENDENT_CODE True)
add_library(batoid STATIC ${SRC_FILES})
pybind11_add_module(_batoid ${PYSRC_FILES})
target_link_libraries(_batoid PUBLIC batoid)
