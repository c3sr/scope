
cmake_minimum_required(VERSION 3.12.1 FATAL_ERROR)

set(CMAKE_BUILD_TYPE "Release")

  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
  set(CUDACXX /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_HOST_COMPILER gcc)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)



option(CONFIG_USE_HUNTER "Turn on to enable using the hunteger package manager" ON)
option(USE_CUDA_EVENTS "Turn on to use cuda events and manual timing, instead of cudaDeviceSynchronize" ON)
option(CUDA_MULTI_ARCH "Whether to generate CUDA code for multiple architectures" OFF)

if (CONFIG_USE_HUNTER)
include("cmake/modules/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.19.214.tar.gz"
    SHA1 "8e82b91de713ac393e22d898d421668403172f90"
)
endif (CONFIG_USE_HUNTER)



project(microbenchmark LANGUAGES CXX VERSION 0.1.0)


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

include(GetGitRevisionDescription)
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if (CONFIG_USE_HUNTER)

hunter_add_package(Sugar)
include(${SUGAR_ROOT}/cmake/Sugar)

hunter_add_package(benchmark)
find_package(benchmark CONFIG REQUIRED)

hunter_add_package(fmt)
find_package(fmt CONFIG REQUIRED)

hunter_add_package(spdlog)
find_package(spdlog CONFIG REQUIRED)

hunter_add_package(cub)
find_package(cub CONFIG REQUIRED)

endif (CONFIG_USE_HUNTER)


find_package(CUDA REQUIRED)
find_package(CuDNN REQUIRED)
find_package(OpenBLAS REQUIRED)
#find_package(OpenMP REQUIRED)
#find_package(Protobuf REQUIRED)


include(sugar_doxygen_generate)
include(sugar_include)

# Disable extended variants of C++ dialects
# i.e. don't choose gnu++17 over c++17
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wundef -fcolor-diagnostics -stdlib=libc++ -pthread"
    CACHE STRING "Toolchain C++ compiler flags." FORCE)

  set(CMAKE_EXE_LINKER_FLAGS "-stdlib=libc++ -lc++abi"
    CACHE STRING "Toolchain C++ linker flags." FORCE)
endif()


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
	#set(CXX_WARNINGS "-Wall -Wextra -Wno-unused-function -Wno-unknown-warning-option -Wpointer-arith -Wcast-align -fstrict-aliasing -Wno-unused-local-typedefs -Wno-misleading-indentation -Wno-maybe-uninitialized -Wno-int-in-bool-context -Wno-implicit-fallthrough -Wno-unused-variable -Wno-unused-command-line-argument")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=1 ")
endif()

if (APPLE)
  list (APPEND CMAKE_CXX_FLAGS_INIT "-fvisibility=hidden -fvisibility-inlines-hidden")
endif(APPLE)



set(TOP_DIR "${PROJECT_SOURCE_DIR}")
set(THIRDPARTY_DIR "${TOP_DIR}/third_party")
set(SRC_DIR "${TOP_DIR}/src")

sugar_include("./src")


add_executable(bench ${BENCHMARK_SOURCES})


include_directories(bench
  ${SRC_DIR}
  ${THIRDPARTY_DIR}
  ${THIRDPARTY_DIR}/cutlass
  ${THIRDPARTY_DIR}/leathers/Source
  ${TOP_DIR}/include
  ${OpenBLAS_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${CUDNN_INCLUDE_DIR}
)

add_definitions(-DPROJECT_NAME=microbench)
add_definitions(-DGENERATED_BENCHMARK_LAYER=1)


#target_compile_options(bench PRIVATE "SHELL:--param ggc-min-expand=3")
#target_compile_options(bench PRIVATE "SHELL:--param ggc-min-heapsize=32768")

target_link_libraries(bench PRIVATE benchmark::benchmark)
target_link_libraries(bench PRIVATE fmt::fmt)
target_link_libraries(bench PRIVATE spdlog::spdlog)
target_link_libraries(bench PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(bench PRIVATE ${CUDNN_LIBRARY})
target_link_libraries(bench PRIVATE ${OpenBLAS_LIB})
target_link_libraries(bench PRIVATE numa)



# Request that bench be built with -std=c++11
# As this is a public compile feature anything that links to
# bench will also build with -std=c++11
target_compile_features(bench PUBLIC cxx_std_11)


