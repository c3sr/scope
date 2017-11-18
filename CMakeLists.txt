
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


include("cmake/modules/HunterGate.cmake")
HunterGate(
  URL "https://github.com/ruslo/hunter/archive/v0.19.161.tar.gz"
  SHA1 "9af482b6a54875bbbb221274f7290324dff29919"
)



project(microbenchmark LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})
set(TOP_DIR "${PROJECT_SOURCE_DIR}")
set(SRC_DIR "${TOP_DIR}/src")

hunter_add_package(benchmark)
find_package(benchmark CONFIG REQUIRED)

hunter_add_package(nlohmann_json)
find_package(nlohmann_json CONFIG REQUIRED)


find_package(OpenBLAS REQUIRED)

add_executable(bench
  src/sgemm.cpp
  src/main.cpp
)

include_directories(bench ${OpenBLAS_INCLUDE_DIR})

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(bench PUBLIC cxx_std_11)


target_link_libraries(bench benchmark::benchmark)
target_link_libraries(bench nlohmann_json)
target_link_libraries(bench ${OpenBLAS_LIB})



