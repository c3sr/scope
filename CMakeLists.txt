
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

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



project(microbenchmark LANGUAGES CXX CUDA VERSION 0.1.0)


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

include(cotire)
include(GetGitRevisionDescription)
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)


if (CONFIG_USE_HUNTER)

  hunter_add_package(Sugar)
  include(${SUGAR_ROOT}/cmake/Sugar)

  hunter_add_package(benchmark)
  hunter_add_package(cub)
  hunter_add_package(fmt)
  hunter_add_package(spdlog)

  find_package(benchmark CONFIG REQUIRED)
  find_package(cub CONFIG REQUIRED)
  find_package(fmt CONFIG REQUIRED)
  find_package(spdlog CONFIG REQUIRED)

# hunter_add_package(nlohmann_json)
# find_package(nlohmann_json CONFIG REQUIRED)

# hunter_add_package(cxxopts)
# find_package(cxxopts CONFIG REQUIRED)

#hunter_add_package(debug_assert)
#find_package(debug_assert CONFIG REQUIRED)

# hunter_add_package(range-v3)
# find_package(range-v3 CONFIG REQUIRED)

# hunter_add_package(type_safe)
# find_package(type_safe CONFIG REQUIRED)

# hunter_add_package(yaml-cpp)
# find_package(yaml-cpp CONFIG REQUIRED)

# hunter_add_package(Protobuf)
# find_package(Protobuf CONFIG REQUIRED)

else()
  # Add Sugar
  set(SUGAR_ROOT "/path/to/sugar/root/directory")
  list(APPEND CMAKE_MODULE_PATH "${SUGAR_ROOT}/cmake/core")
  
  message(STATUS "looking for CUB in: ${CUB_ROOT}")
  include_directories(${CUB_ROOT}/include)
  find_package(spdlog REQUIRED)
  find_package(benchmark REQUIRED)
  find_package(fmt REQUIRED)
endif (CONFIG_USE_HUNTER)


find_package(CUDA REQUIRED)
find_package(CuDNN REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(OpenMP REQUIRED)
#find_package(Protobuf REQUIRED)


include(sugar_doxygen_generate)
include(sugar_include)

# Disable extended variants of C++ dialects
# i.e. don't choose gnu++17 over c++17
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wundef -fcolor-diagnostics -stdlib=libc++ -pthread"
    CACHE STRING "Toolchain C++ compiler flags." FORCE)

  set(CMAKE_EXE_LINKER_FLAGS "-stdlib=libc++ -lc++abi -fuse-ld=lld"
    CACHE STRING "Toolchain C++ linker flags." FORCE)
endif()


if (OpenMP_FOUND)
  if (OpenMP_C_FLAGS)
    string(REPLACE ";" " " OpenMP_C_FLAGS ${OpenMP_C_FLAGS})
  endif()
  if (OpenMP_CXX_FLAGS)
    string(REPLACE ";" " " OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS})
  endif()
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

  foreach(flag IN ITEMS ${OpenMP_CXX_FLAGS})
  LIST(APPEND CMAKE_CUDA_FLAGS
    -Xcompiler
    ${flag}
  )
  endforeach(flag)

endif()


if (UNIX AND NOT APPLE)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER}
                  -fuse-ld=gold -Wl,--version
                  ERROR_QUIET OUTPUT_VARIABLE ld_version)
  if ("${ld_version}" MATCHES "GNU gold")
    message(STATUS "Found Gold linker, use faster linker")
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold ")
  endif()
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
	set(CXX_WARNINGS "-Wall -Wextra -Wno-unused-function -Wno-unknown-warning-option -Wpointer-arith -Wcast-align -fstrict-aliasing -Wno-unused-local-typedefs -Wno-misleading-indentation -Wno-maybe-uninitialized -Wno-int-in-bool-context -Wno-implicit-fallthrough -Wno-unused-variable -Wno-unused-command-line-argument")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=1 -Wfatal-errors ${CXX_WARNINGS}")
endif()

if (APPLE)
  list (APPEND CMAKE_CXX_FLAGS_INIT "-fvisibility=hidden -fvisibility-inlines-hidden")
endif(APPLE)


# CUDA flags
set(CMAKE_CUDA_STANDARD 11)

# CMake FindCUDA auto seems to add unsupported architectures somtimes, so we allow the user
# to override with NVCC_ARCH_FLAGS
if(CUDA_MULTI_ARCH)
  CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS All)
else()
  if (DEFINED NVCC_ARCH_FLAGS)
    message(STATUS "Manual cuda arch flags...")
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${NVCC_ARCH_FLAGS})
  else()
    message(STATUS "Automatic cuda arch flags...")
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)
  endif()
endif()

LIST(APPEND CMAKE_CUDA_FLAGS ${CUDA_ARCH_FLAGS}
					         -Wno-deprecated-gpu-targets
					         -lineinfo
					         --expt-extended-lambda
)

MESSAGE(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

MESSAGE(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

MESSAGE(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
			   " archs: ${CUDA_ARCH_FLAGS_readable})")

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_VERBOSE_BUILD OFF)



set(TOP_DIR "${PROJECT_SOURCE_DIR}")
set(THIRDPARTY_DIR "${TOP_DIR}/third_party")
set(SRC_DIR "${TOP_DIR}/src")

sugar_include("./src")


add_executable(bench ${BENCHMARK_SOURCES} ${BENCHMARK_CUDA_SOURCES})


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

if (USE_CUDA_EVENTS)
  add_definitions(-DUSE_CUDA_EVENTS=1)
endif(USE_CUDA_EVENTS)

add_definitions(-DPROJECT_NAME=microbench)
add_definitions(-DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR})
add_definitions(-DCUDA_VERSION_MINOR=${CUDA_VERSION_MINOR})

set_target_properties(bench PROPERTIES
	# CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(bench benchmark::benchmark)
# target_link_libraries(bench nlohmann_json)
target_link_libraries(bench fmt::fmt)
target_link_libraries(bench spdlog::spdlog)
target_link_libraries(bench cub::cub)
# target_link_libraries(bench rangev3)
# target_link_libraries(bench type_safe)
# target_link_libraries(bench yaml-cpp::yaml-cpp)
# target_link_libraries(bench ${PROTOBUF_LIBRARY} )
#target_link_libraries(bench debug_assert)
target_link_libraries(bench ${CUDNN_LIBRARY})
target_link_libraries(bench ${OpenBLAS_LIB})
target_link_libraries(bench ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(bench numa)
target_link_libraries(bench ${OpenMP_CXX_LIBRARIES})


# Request that bench be built with -std=c++11
# As this is a public compile feature anything that links to
# bench will also build with -std=c++11
target_compile_features(bench PUBLIC cxx_std_11)


cotire(bench)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET bench
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()
