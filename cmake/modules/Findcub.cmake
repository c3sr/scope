include(FindPackageHandleStandardArgs)
option(cub "Path to CUB installation directory")


SET(cub_INCLUDE_SEARCH_PATHS
${cub}
$ENV{cub}
$ENV{cub_HOME}
)

FIND_PATH(cub_INCLUDE_DIR NAMES cub/cub.cuh PATHS ${cub_INCLUDE_SEARCH_PATHS})

find_package_handle_standard_args(cub DEFAULT_MSG
  cub_INCLUDE_DIR)

MARK_AS_ADVANCED(
  cub_INCLUDE_DIR
  cub_LIB
  cub
)
