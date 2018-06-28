include(FindPackageHandleStandardArgs)

SET(numa_INCLUDE_SEARCH_PATHS
      ${numa}
      /usr/include
      $ENV{numa}
      $ENV{numa_HOME}
      $ENV{numa_HOME}/include
)

SET(numa_LIBRARY_SEARCH_PATHS
      ${numa}
      /usr/lib
      $ENV{numa}
      $ENV{numa_HOME}
      $ENV{numa_HOME}/lib
)

find_path(numa_INCLUDE_DIR
  NAMES numa.h
  PATHS ${numa_INCLUDE_SEARCH_PATHS}
  DOC "NUMA include directory")

find_library(numa_LIBRARY
  NAMES numa
  HINTS ${numa_LIBRARY_SEARCH_PATHS}
  DOC "NUMA library")

if (NUMA_LIBRARY)
    get_filename_component(numa_LIBRARY_DIR ${numa_LIBRARY} PATH)
endif()

mark_as_advanced(numa_INCLUDE_DIR numa_LIBRARY_DIR numa_LIBRARY)

find_package_handle_standard_args(numa REQUIRED_VARS numa_INCLUDE_DIR numa_LIBRARY)