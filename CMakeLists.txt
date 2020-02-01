cmake_minimum_required(VERSION 3.9)
project(_batoid CXX)
# project(_batoid CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE True)
cmake_policy(SET CMP0063 NEW)  # Compile the static lib with hidden visibility.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
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
# GPU stuff
  pysrc/coordsys2.cpp
  pysrc/medium2.cpp
  pysrc/rayVector2.cpp
  pysrc/surface2.cpp
  pysrc/paraboloid2.cpp
  pysrc/plane2.cpp
  pysrc/sphere2.cpp
  pysrc/quadric2.cpp
  pysrc/asphere2.cpp
)
set(SRC_FILES
  src/asphere.cpp
  src/batoid.cpp
  src/bicubic.cpp
  src/coating.cpp
  src/coordsys.cpp
  src/dualView.cpp
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
# GPU stuff
  src/coordsys2.cpp
  src/medium2.cpp
  src/rayVector2.cpp
  src/surface2.cpp
  src/paraboloid2.cpp
  src/plane2.cpp
  src/sphere2.cpp
  src/quadric2.cpp
  src/asphere2.cpp
)

set(ALL_FILES ${SRC_FILES} ${PYSRC_FILES})

find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#   add_library(omp INTERFACE)
#   add_library(omp-offload INTERFACE)
#   target_link_libraries(omp-offload INTERFACE omp "-fopenmp-targets=nvptx64-nvidia-cuda")
#   # target_link_libraries(omp-offload INTERFACE omp)
#   target_link_libraries(omp INTERFACE "-fopenmp")
#   # target_link_libraries(omp INTERFACE ${OpenMP_CXX_LIBRARIES})
#   target_compile_options(omp INTERFACE
#     $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
#   )
#   check_cxx_compiler_flag("-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda" OMP_OFFLOAD)
#   if(OMP_OFFLOAD)
#     target_compile_options(omp-offload INTERFACE
#       $<$<COMPILE_LANGUAGE:CXX>:-fopenmp-targets=nvptx64-nvidia-cuda>
#     )
#     add_library(batoid-offload STATIC ${SRC_FILES})
#     target_link_libraries(batoid-offload batoid-compile-flags omp-offload)
#     # set_target_properties(batoid-offload PROPERTIES LINKER_LANGUAGE CUDA)
#     pybind11_add_module(_batoid ${PYSRC_FILES})
#     target_link_libraries(_batoid PUBLIC batoid-compile-flags batoid-offload "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda")
#   else()
#     add_library(batoid-omp STATIC ${SRC_FILES})
#     target_link_libraries(batoid-omp batoid-compile-flags omp)
#     pybind11_add_module(_batoid ${PYSRC_FILES})
#     target_link_libraries(_batoid PUBLIC batoid-compile-flags batoid-omp)
#   endif()
# else()
#   add_library(batoid STATIC ${SRC_FILES})
#   pybind11_add_module(_batoid ${PYSRC_FILES})
#   target_link_libraries(_batoid PUBLIC batoid-compile-flags batoid)
# endif()

target_compile_options(batoid-compile-flags INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)
target_compile_options(batoid-compile-flags INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-fopenmp-targets=nvptx64-nvidia-cuda>
)

# add_library(batoid STATIC ${SRC_FILES})
# target_link_libraries(batoid batoid-compile-flags "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda")
# pybind11_add_module(_batoid ${PYSRC_FILES})
# target_link_libraries(_batoid PUBLIC batoid-compile-flags batoid)

pybind11_add_module(_batoid ${ALL_FILES})
target_link_libraries(_batoid PUBLIC batoid-compile-flags "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda")
