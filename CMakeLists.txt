cmake_minimum_required(VERSION 3.9)
project(_batoid CXX)
# set(CMAKE_VERBOSE_MAKEFILE True)
cmake_policy(SET CMP0063 NEW)  # Compile the static lib with hidden visibility.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE True)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

if (DEFINED ENV{CMAKE_COVER})
  set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
  include(CodeCoverage)
  APPEND_COVERAGE_COMPILER_FLAGS()
endif()

include_directories(include)
add_subdirectory(pybind11)

set(SRC_FILES
  src/asphere.cpp
  src/batoid.cpp
  src/bicubic.cpp
  src/coating.cpp
  src/dualView.cpp
  src/medium.cpp
  src/obscuration.cpp
  src/paraboloid.cpp
  src/plane.cpp
  src/polynomialSurface.cpp
  src/quadric.cpp
  src/rayVector.cpp
  src/sphere.cpp
  src/sum.cpp
  src/surface.cpp
  src/tilted.cpp
  src/table.cpp
)

set(PYSRC_FILES
  pysrc/asphere.cpp
  pysrc/batoid.cpp
  pysrc/bicubic.cpp
  pysrc/coating.cpp
  pysrc/medium.cpp
  pysrc/obscuration.cpp
  pysrc/paraboloid.cpp
  pysrc/plane.cpp
  pysrc/polynomialSurface.cpp
  pysrc/quadric.cpp
  pysrc/rayVector.cpp
  pysrc/sphere.cpp
  pysrc/sum.cpp
  pysrc/surface.cpp
  pysrc/tilted.cpp
  pysrc/table.cpp
)

include(CheckCXXCompilerFlag)
macro(ADD_CXX_FLAG_IF_AVAIL  TARG  FLAG  FLAG_NAME)
  if(NOT "${FLAG}" STREQUAL "")
    check_cxx_compiler_flag("${FLAG}" ${FLAG_NAME})
    if(${FLAG_NAME})
      target_compile_options(${TARG} INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
    endif()
  endif()
endmacro()

add_library(batoid-compile-flags INTERFACE)
add_cxx_flag_if_avail(batoid-compile-flags "-Wno-unused-value" CXX_W_NO_UNUSED_VALUE)

if (DEFINED ENV{BATOID_GPU})

  add_cxx_flag_if_avail(batoid-compile-flags "-Wno-openmp-mapping" CXX_W_NO_OPENMP_MAPPING)
  add_cxx_flag_if_avail(batoid-compile-flags "-fopenmp" CXX_FOPENMP)
  # add_cxx_flag_if_avail doesn't seem to work for fopenmp-targets, so add forcibly
  # add_cxx_flag_if_avail(batoid-compile-flags "-fopenmp-targets=nvptx64-nvidia-cuda" CXX_FOPENMP_TARGETS)
  target_compile_options(batoid-compile-flags INTERFACE
    # $<$<COMPILE_LANGUAGE:CXX>:-fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_70 -Wno-unknown-cuda-version -Wno-pass-failed>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp-targets=nvptx64-nvidia-cuda>
  )

  set(ALL_FILES ${SRC_FILES} ${PYSRC_FILES})

  add_definitions(-DBATOID_GPU)
  pybind11_add_module(_batoid SYSTEM ${ALL_FILES})
  # target_link_libraries(_batoid PUBLIC batoid-compile-flags "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_70 -Wno-unknown-cuda-version -Wno-pass-failed")
  target_link_libraries(_batoid PUBLIC batoid-compile-flags "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda")

  set_target_properties(_batoid
    PROPERTIES
    DEFINE_SYMBOL ""
  )

else()

  add_library(batoid ${SRC_FILES})
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(batoid batoid-compile-flags OpenMP::OpenMP_CXX)
  else()
    target_link_libraries(batoid batoid-compile-flags)
  endif()
  pybind11_add_module(_batoid SYSTEM ${PYSRC_FILES})
  target_link_libraries(_batoid PRIVATE batoid INTERFACE batoid-compile-flags)

endif()
