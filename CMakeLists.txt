
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if (APPLE)
  set(CMAKE_C_COMPILER clang-5.0)
  set(CMAKE_CXX_COMPILER clang++-5.0)
  set(CUDACXX /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_HOST_COMPILER clang-3.7)
endif(APPLE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CONFIG_USE_HUNTER "ON" CACHE BOOL "Turn on to enable using the hunteger package manager")


if (CONFIG_USE_HUNTER)
include("cmake/modules/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.19.214.tar.gz"
    SHA1 "8e82b91de713ac393e22d898d421668403172f90"
)
endif (CONFIG_USE_HUNTER)



project(microbenchmark LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  # /wd4005  macro-redefinition
  # /wd4068  unknown pragma
  # /wd4244  conversion from 'type1' to 'type2'
  # /wd4267  conversion from 'size_t' to 'type2'
  # /wd4800  force value to bool 'true' or 'false' (performance warning)
  add_compile_options(/wd4005 /wd4068 /wd4244 /wd4267 /wd4800)
  add_definitions(/DNOMINMAX /DWIN32_LEAN_AND_MEAN=1 /DCONSTEXPR=const /D_CRT_SECURE_NO_WARNINGS)
else()
	add_definitions(-DCONSTEXPR=constexpr)
	set(CXX_WARNINGS "-Wall -Wextra -Wno-unused-function -Wno-unknown-warning-option -Wpointer-arith -Wcast-align -fstrict-aliasing -Wno-unused-local-typedefs -Wno-misleading-indentation -Wno-maybe-uninitialized -Wno-int-in-bool-context -Wno-implicit-fallthrough")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_WARNINGS} -fvisibility-inlines-hidden -O3")
endif()


# CUDA flags
set(CMAKE_CUDA_STANDARD 11)
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_30,code=compute_30 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_30,code=sm_30 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_32,code=sm_32 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_35,code=sm_35 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_52,code=sm_52 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_53,code=sm_53 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=sm_60 ")
string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_62,code=sm_62 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70 ")


if (CONFIG_USE_HUNTER)
hunter_add_package(benchmark)
find_package(benchmark CONFIG REQUIRED)

hunter_add_package(nlohmann_json)
find_package(nlohmann_json CONFIG REQUIRED)

hunter_add_package(fmt)
find_package(fmt CONFIG REQUIRED)

hunter_add_package(spdlog)
find_package(spdlog CONFIG REQUIRED)

# hunter_add_package(cxxopts)
# find_package(cxxopts CONFIG REQUIRED)

hunter_add_package(cub)
find_package(cub CONFIG REQUIRED)

hunter_add_package(debug_assert)
find_package(debug_assert CONFIG REQUIRED)

# hunter_add_package(range-v3)
# find_package(range-v3 CONFIG REQUIRED)

# hunter_add_package(type_safe)
# find_package(type_safe CONFIG REQUIRED)

# hunter_add_package(yaml-cpp)
# find_package(yaml-cpp CONFIG REQUIRED)

# hunter_add_package(Protobuf)
# find_package(Protobuf CONFIG REQUIRED)

endif (CONFIG_USE_HUNTER)


find_package(CUDA REQUIRED)
find_package(OpenBLAS REQUIRED)


set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_VERBOSE_BUILD OFF)



set(TOP_DIR "${PROJECT_SOURCE_DIR}")
set(THIRDPARTY_DIR "${TOP_DIR}/third_party")
set(SRC_DIR "${TOP_DIR}/src")

# add_subdirectory(${THIRDPARTY_DIR}/googletest)
# add_subdirectory(${THIRDPARTY_DIR}/cctz)
# add_subdirectory(${THIRDPARTY_DIR}/abseil-cpp)
add_subdirectory(${THIRDPARTY_DIR}/cxxopts)

add_executable(bench
  ${SRC_DIR}/cblas_daxpy.cpp
  ${SRC_DIR}/cblas_sgemm.cpp
  ${SRC_DIR}/cuda_memcpy.cpp
  ${SRC_DIR}/cuda_launch.cu
  ${SRC_DIR}/cuda_stream.cu
  ${SRC_DIR}/cublas_sgemm.cpp
  ${SRC_DIR}/init.cpp
  ${SRC_DIR}/main.cpp
)

include_directories(bench
  ${SRC_DIR}
  ${THIRDPARTY_DIR}
  ${TOP_DIR}/include
  ${OpenBLAS_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

set_target_properties(bench PROPERTIES
	# CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(bench benchmark::benchmark)
target_link_libraries(bench nlohmann_json)
target_link_libraries(bench fmt::fmt)
target_link_libraries(bench spdlog::spdlog)
target_link_libraries(bench cub::cub)
# target_link_libraries(bench rangev3)
# target_link_libraries(bench type_safe)
# target_link_libraries(bench yaml-cpp::yaml-cpp)
# target_link_libraries(bench protobuf::libprotobuf)
target_link_libraries(bench cxxopts)
target_link_libraries(bench debug_assert)
# target_link_libraries(bench absl::base absl::synchronization absl::strings)
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
