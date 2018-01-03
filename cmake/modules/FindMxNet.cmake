################################################################################
# - Try to find MxNet (http://mxnet.incubator.apache.org/)
# Once done this will define
# MxNet_FOUND - system has MxNet
# MxNet_INCLUDE_DIRS - The MxNet include directories
# MxNet_LIB - The Mxnet .so path
#
# To develop C/C++ code with MxNet, its source code must be checked out and
# compiled locally. This module works best if an environment varialbe
# called MXNET_ROOT is defined, and points to the root of MxNet source
# code.
################################################################################

# https://github.com/julitopower/FindMxNet

file(TO_CMAKE_PATH "$ENV{MXNET_ROOT}" MXNET_ROOT)

SET(MXNET_INCLUDE_SEARCH_PATHS
  /usr/include/mxnet
  /usr/local/include/mxnet
  /opt/mxnet/include/mxnet
  /opt/staging/incubator-mxnet/include/mxnet
  /usr/local/opt/mxnet/include/mxnet
  ${MXNET_ROOT}/include/mxnet
)

SET(MXNET_LIB_SEARCH_PATHS
        /lib
        /lib/mxnet
        /lib64
        /usr/lib
        /usr/lib/mxnet
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/MXNET/lib
	/opt/stating/
	/opt/staging/incubator-mxnet/lib
        /usr/local/opt/openblas/lib
        ${PROJECT_SOURCE_DIR}/3rdparty/mxnet/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/mxnet/lib
        ${MXNET_ROOT}
        ${MXNET_ROOT}/lib
 )


FIND_PATH(MxNet_INCLUDE_DIR NAMES c_api.h PATHS ${MXNET_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(MxNet_LIB NAMES mxnet PATHS ${MXNET_LIB_SEARCH_PATHS})

SET(MxNet_FOUND ON)

#    Check include files
IF(NOT MxNet_INCLUDE_DIR)
    SET(MxNet_FOUND OFF)
    MESSAGE(STATUS "Could not find MxNet include. Turning MxNet_FOUND off")
ENDIF()

# Set all the include paths relative to the one found
set(MxNet_INCLUDE_DIRS 
  ${MxNet_INCLUDE_DIR}/../
  ${MxNet_INCLUDE_DIR}/../../cpp-package/include/
  ${MxNet_INCLUDE_DIR}/../../dmlc-core/include/
  ${MxNet_INCLUDE_DIR}/../../mshadow
  ${MxNet_INCLUDE_DIR}/../../nnvm/include
  ${MxNet_INCLUDE_DIR}/dlpack/include
)

#    Check libraries
IF(NOT MxNet_LIB)
    SET(MxNet_FOUND OFF)
    MESSAGE(STATUS "Could not find MxNet lib. Turning MxNet_FOUND off")
ENDIF()

IF (MxNet_FOUND)
  IF (NOT MxNet_FIND_QUIETLY)
    MESSAGE(STATUS "Found MxNet libraries: ${MxNet_LIB}")
    MESSAGE(STATUS "Found MxNet include: ${MxNet_INCLUDE_DIRS}")
  ENDIF (NOT MxNet_FIND_QUIETLY)
ELSE (MxNet_FOUND)
  IF (MxNet_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find MxNet")
  ENDIF (MxNet_FIND_REQUIRED)
ENDIF (MxNet_FOUND)

MARK_AS_ADVANCED(
    MxNet_INCLUDE_DIRS
    MxNet_LIB
)
