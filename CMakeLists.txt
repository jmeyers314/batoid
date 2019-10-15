cmake_minimum_required(VERSION 3.9)
project(_batoid CXX)
set(CMAKE_VERBOSE_MAKEFILE True)
cmake_policy(SET CMP0063 NEW)  # Compile the static lib with hidden visibility.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-value")
set(CMAKE_POSITION_INDEPENDENT_CODE True)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

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
  #pysrc/rayVector2.cpp
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
  #src/rayVector2.cpp
  src/sphere.cpp
  src/sum.cpp
  src/surface.cpp
  src/table.cpp
  src/utils.cpp
)

add_library(batoid STATIC ${SRC_FILES})
pybind11_add_module(_batoid ${PYSRC_FILES})
target_link_libraries(_batoid PUBLIC batoid)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(batoid PUBLIC OpenMP::OpenMP_CXX)
endif()

