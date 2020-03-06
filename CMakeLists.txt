
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(CONFIG_USE_HUNTER "Turn on to enable using the hunteger package manager" ON)
option(CUDA_MULTI_ARCH "Whether to generate CUDA code for multiple architectures" OFF)

# https://github.com/hunter-packages/gate#usage-custom-config
if (CONFIG_USE_HUNTER)
  include("cmake/modules/HunterGate.cmake")
  HunterGate(
      URL "https://github.com/ruslo/hunter/archive/v0.23.1.tar.gz"
      SHA1 "51d2d6be411251c8de18c4ca20ef778880cf4cce"
      LOCAL 
  )
endif (CONFIG_USE_HUNTER)

project(scope LANGUAGES C CXX CUDA VERSION 1.3.2)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

include(CMakePackageConfigHelpers)
include(CTest)
include(GetGitRevisionDescription)
include(GNUInstallDirs)
include(ScopeAddLibrary)
include(ScopeFatal)
include(ScopeStatus)
include(ScopeWarning)
include(TargetIncludeScopeDirectories)
include(TargetLinkScopeLibraries)

set(SCOPE_TOP_DIR "${PROJECT_SOURCE_DIR}")
set(SCOPE_THIRDPARTY_DIR "${SCOPE_TOP_DIR}/third_party")
set(SCOPE_SRC_DIR "${SCOPE_TOP_DIR}/src")
set(SCOPE_SCOPES_DIR "${SCOPE_TOP_DIR}/scopes")

# Checkout submodules
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE_UPDATE "Update submodules during cmake" OFF)
    if(GIT_SUBMODULE_UPDATE)
        scope_status("Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            scope_fatal("git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(   NOT EXISTS "${SCOPE_SCOPES_DIR}/nccl_scope/CMakeLists.txt" 
   OR NOT EXISTS "${SCOPE_SCOPES_DIR}/comm_scope/CMakeLists.txt"
   OR NOT EXISTS "${SCOPE_SCOPES_DIR}/example_scope/CMakeLists.txt"
   OR NOT EXISTS "${SCOPE_SCOPES_DIR}/misc_scope/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE_UPDATE was turned off or failed. Please update submodules and try again.")
endif()

if (CONFIG_USE_HUNTER)
  hunter_add_package(sugar)
  find_package(sugar CONFIG REQUIRED)
  # include(${SUGAR_ROOT}/cmake/sugar)

  hunter_add_package(fmt)
  hunter_add_package(spdlog)
else()
  # Add Sugar
  scope_status("looking for Sugar in: ${SUGAR_ROOT}")
  list(APPEND CMAKE_MODULE_PATH "${SUGAR_ROOT}/cmake/module")
endif (CONFIG_USE_HUNTER)


find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

set(BENCHMARK_ENABLE_TESTING OFF) # remove dep on gtest
set(BENCHMARK_ENABLE_GTEST_TESTS OFF) # remove dep on gtest
add_subdirectory(third_party/benchmark)
# google benchmark doesn't seem to export benchmark::benchmark target as promised
add_library(benchmark::benchmark ALIAS benchmark)

include(sugar_doxygen_generate)
include(sugar_include)


# Find the  CUDA toolkit directory so we can link against CUDA libraries
# https://gitlab.kitware.com/cmake/cmake/issues/17816
set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}")
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
scope_status(${CUDA_TOOLKIT_ROOT_DIR})
find_library(CUDART_LIB cudart
             HINTS
             "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
             "${CUDA_TOOLKIT_ROOT_DIR}/lib"
             "${CUDA_TOOLKIT_ROOT_DIR}"
)
scope_status(${CUDART_LIB})


# create imported library for libcudart
add_library(CUDA::cudart IMPORTED INTERFACE)
set_target_properties(CUDA::cudart PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
  INTERFACE_LINK_LIBRARIES
    "${CUDART_LIB}")
  







if (APPLE)
  list (APPEND CMAKE_CXX_FLAGS_INIT "-fvisibility=hidden -fvisibility-inlines-hidden")
endif(APPLE)


# CUDA flags
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

sugar_include("./src")


add_executable(scope ${BENCHMARK_SOURCES} ${BENCHMARK_CUDA_SOURCES})

# use c++11 not g++11
set_target_properties(scope PROPERTIES CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  # /wd4005  macro-redefinition
  # /wd4068  unknown pragma
  # /wd4244  conversion from 'type1' to 'type2'
  # /wd4267  conversion from 'size_t' to 'type2'
  # /wd4800  force value to bool 'true' or 'false' (performance warning)
  target_compile_options(scope PPRIVATE /wd4005 /wd4068 /wd4244 /wd4267 /wd4800)
  target_compile_definitions(scope PRIVATE /DNOMINMAX /DWIN32_LEAN_AND_MEAN=1 /DCONSTEXPR=const /D_CRT_SECURE_NO_WARNINGS)
else()
  target_compile_definitions(scope PRIVATE -DCONSTEXPR=constexpr)
  target_compile_options(scope PRIVATE
    -Wall 
    -Wextra 
    -Wcast-qual 
    -Wcast-align 
    -Wstrict-aliasing 
    -Wpointer-arith 
    -Winit-self 
    -Wshadow 
    -Wswitch-enum 
    -Wredundant-decls 
    -Wfloat-equal 
    -Wundef 
    -Wvla
    -fmax-errors=1 
    -Wfatal-errors
  )
endif()


if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
  target_compile_options(scope PRIVATE
    -fcolor-diagnostics
    -pthread
  )
endif()

# CXX flags
target_compile_options(scope PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

# CUDA flags
target_compile_options( scope PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--Wno-deprecated-gpu-targets;--expt-extended-lambda;-Xcompiler=-Wall;-Xcompiler=-Wextra>
)

# Treat everything in scope/scopes as a scope
# We expect that those scopes will set SCOPE_NEW_TARGET to the scope target if they are enabled
file(GLOB children LIST_DIRECTORIES true "${SCOPE_SCOPES_DIR}/*" )
foreach( child ${children} )
  if(IS_DIRECTORY ${child} )
    unset(SCOPE_NEW_TARGET)
    add_subdirectory(${child})
    if (DEFINED SCOPE_NEW_TARGET)
      scope_status("enabling ${child}")

      target_compile_options(${SCOPE_NEW_TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
        -Wall;
        -Wextra;
        -Wcast-qual;
        -Wcast-align;
        -Wstrict-aliasing;
        -Wpointer-arith;
        -Winit-self;
        -Wshadow;
        -Wswitch-enum;
        -Wredundant-decls;
        -Wfloat-equal;
        -Wundef;
        -Wvla;
        -fmax-errors=1;
        -Wfatal-errors;
        >
      )

      # add CUDA options to target
      target_compile_options(${SCOPE_NEW_TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--Wno-deprecated-gpu-targets;--expt-extended-lambda;-Xcompiler=-Wall;-Xcompiler=-Wextra>
      )

      # link scope against the target
      target_link_libraries(scope PUBLIC ${SCOPE_NEW_TARGET})
    endif()
  endif()
endforeach()


# Add include directories
target_include_scope_directories(scope)
target_include_directories(scope PRIVATE
  ${PROJECT_BINARY_DIR}/src/
)

target_include_directories(scope SYSTEM INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(scope SYSTEM PUBLIC ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES})



target_compile_definitions(scope PUBLIC -DPROJECT_NAME=scope)

set_target_properties(scope PROPERTIES
	# CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(scope PUBLIC benchmark::benchmark)
target_link_libraries(scope PUBLIC fmt::fmt)
target_link_libraries(scope PUBLIC spdlog::spdlog)
target_link_libraries(scope PRIVATE CUDA::cudart)


# Request that scope be built with -std=c++11
# As this is a public compile feature anything that links to
# scope will also build with -std=c++11
target_compile_features(scope PUBLIC cxx_std_11)

git_get_head_revision(GIT_REFSPEC GIT_HASH)
git_local_changes(GIT_LOCAL_CHANGES)
scope_status(GIT_REFSPEC=${GIT_REFSPEC})
scope_status(GIT_HASH=${GIT_HASH})
scope_status(GIT_LOCAL_CHANGES=${GIT_LOCAL_CHANGES})

# Generate version file
scope_status("${PROJECT_SOURCE_DIR}/.../config.hpp.in -> ${PROJECT_BINARY_DIR}/.../config.hpp")
configure_file (
    "${PROJECT_SOURCE_DIR}/src/config.hpp.in"
    "${PROJECT_BINARY_DIR}/src/config.hpp"
)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET scope
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()


