
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CUDACXX /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER gcc)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(CONFIG_USE_HUNTER "Turn on to enable using the hunteger package manager" ON)
option(CUDA_MULTI_ARCH "Whether to generate CUDA code for multiple architectures" OFF)
option(ENABLE_COMM "Include Comm|Scope (github.com/rai-project/comm_scope)" ON)
option(ENABLE_MISC "Include Misc|Scope (github.com/rai-project/misc_scope)" OFF)
option(ENABLE_EXAMPLE "Include Example|Scope (github.com/rai-project/example_scope)" ON)
option(ENABLE_NCCL "Include Nccl|Scope (github.com/rai-project/nccl_scope)" OFF)

if (CONFIG_USE_HUNTER)
  include("cmake/modules/HunterGate.cmake")
  HunterGate(
      URL "https://github.com/ruslo/hunter/archive/v0.23.1.tar.gz"
      SHA1 "51d2d6be411251c8de18c4ca20ef778880cf4cce"
  )
endif (CONFIG_USE_HUNTER)

project(scope LANGUAGES C CXX CUDA VERSION 0.1.0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

include(cotire)
include(CMakePackageConfigHelpers)
include(CTest)
include(ExternalProject)
include(GetGitRevisionDescription)
include(GNUInstallDirs)

if (CONFIG_USE_HUNTER)
  hunter_add_package(sugar)
  find_package(sugar CONFIG REQUIRED)
  # include(${SUGAR_ROOT}/cmake/sugar)

  hunter_add_package(benchmark)
  hunter_add_package(fmt)
  hunter_add_package(spdlog)

  find_package(spdlog CONFIG REQUIRED)
else()
  # Add Sugar
  message(STATUS "looking for Sugar in: ${SUGAR_ROOT}")
  list(APPEND CMAKE_MODULE_PATH "${SUGAR_ROOT}/cmake/module")
endif (CONFIG_USE_HUNTER)


find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(CUDA REQUIRED)

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
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

MESSAGE(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
			   " archs: ${CUDA_ARCH_FLAGS_readable})")

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_VERBOSE_BUILD OFF)



set(TOP_DIR "${PROJECT_SOURCE_DIR}")
set(THIRDPARTY_DIR "${TOP_DIR}/third_party")
set(SRC_DIR "${TOP_DIR}/src")

sugar_include("./src")


add_executable(scope ${BENCHMARK_SOURCES} ${BENCHMARK_CUDA_SOURCES})

if (ENABLE_COMM)
  message(STATUS "Enabling Comm|Scope")
  add_subdirectory(${TOP_DIR}/comm_scope)
  target_link_libraries(scope comm_scope)
endif(ENABLE_COMM)

if (ENABLE_MISC)
  message(STATUS "Enabling Misc|Scope")
  add_subdirectory(${TOP_DIR}/misc_scope)
  target_link_libraries(scope misc_scope)
endif(ENABLE_MISC)

if (ENABLE_EXAMPLE)
  message(STATUS "Enabling Example|Scope")
  add_subdirectory(${TOP_DIR}/example_scope)
  target_link_libraries(scope example_scope)
endif(ENABLE_EXAMPLE)

if (ENABLE_NCCL)
  message(STATUS "Enabling NCCL|Scope")
  add_subdirectory(${TOP_DIR}/nccl_scope)
  target_link_libraries(scope nccl_scope)
endif(ENABLE_NCCL)

# once all common utilities in src/scope, remove ${SRC_DIR}
target_include_directories(scope PRIVATE
  ${SRC_DIR}
  ${TOP_DIR}/include
)

target_include_directories(scope SYSTEM PRIVATE
  ${THIRDPARTY_DIR}
  ${CUDA_INCLUDE_DIRS}
)

target_compile_definitions(scope PUBLIC -DPROJECT_NAME=scope)

set_target_properties(scope PROPERTIES
	# CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(scope benchmark::benchmark)
target_link_libraries(scope fmt::fmt)
target_link_libraries(scope spdlog::spdlog)
target_link_libraries(scope ${CUDA_LIBRARIES})

# Request that scope be built with -std=c++11
# As this is a public compile feature anything that links to
# scope will also build with -std=c++11
target_compile_features(scope PUBLIC cxx_std_11)

git_get_head_revision(GIT_REFSPEC GIT_HASH)
git_local_changes(GIT_LOCAL_CHANGES)
MESSAGE(STATUS GIT_REFSPEC=${GIT_REFSPEC})
MESSAGE(STATUS GIT_HASH=${GIT_HASH})
MESSAGE(STATUS GIT_LOCAL_CHANGES=${GIT_LOCAL_CHANGES})

# Convert CUDA flags from list
MESSAGE(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
MESSAGE(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

# Generate version file
message(STATUS "${PROJECT_SOURCE_DIR}/.../version.hpp.in -> ${PROJECT_SOURCE_DIR}/.../version.hpp")
configure_file (
    "${PROJECT_SOURCE_DIR}/src/scope/utils/version.hpp.in"
    "${PROJECT_SOURCE_DIR}/src/scope/utils/version.hpp"
)

cotire(scope)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET scope
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

