
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


set(CONFIG_USE_HUNTER "ON" CACHE BOOL "Turn on to enable using the hunteger package manager")




if (CONFIG_USE_HUNTER)
include("cmake/modules/HunterGate.cmake")
HunterGate(
  URL "https://github.com/ruslo/hunter/archive/v0.19.161.tar.gz"
  SHA1 "9af482b6a54875bbbb221274f7290324dff29919"
)
endif (CONFIG_USE_HUNTER)



project(microbenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})
set(TOP_DIR "${PROJECT_SOURCE_DIR}")
set(SRC_DIR "${TOP_DIR}/src")

if (CONFIG_USE_HUNTER)
hunter_add_package(benchmark)
find_package(benchmark CONFIG REQUIRED)

hunter_add_package(nlohmann_json)
find_package(nlohmann_json CONFIG REQUIRED)
endif (CONFIG_USE_HUNTER)


find_package(CUDA REQUIRED)
find_package(OpenBLAS REQUIRED)


set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_VERBOSE_BUILD OFF)


add_executable(bench
  src/daxpy.cpp
  src/sgemm.cpp
  src/cudamemcpy.cpp
  src/main.cpp
)

include_directories(bench
  ${OpenBLAS_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(bench benchmark::benchmark)
target_link_libraries(bench nlohmann_json)
target_link_libraries(bench ${OpenBLAS_LIB})
target_link_libraries(bench ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})


# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(bench PUBLIC cxx_std_11)

 
if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET bench 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()
