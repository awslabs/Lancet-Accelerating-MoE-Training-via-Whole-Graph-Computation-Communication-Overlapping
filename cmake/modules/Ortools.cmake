# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

##############################################################################
# Provide:
#  - RAF_ORTOOLS_LIBRARY
#  - RAF_ORTOOLS_INCLUDE

if (${RAF_USE_ORTOOLS} STREQUAL "OFF")
  message(STATUS "Build without ORTOOLS support")
  set(RAF_ORTOOLS_INCLUDE "")
  set(RAF_ORTOOLS_LIBRARY "")
else()
  if (NOT ${RAF_USE_ORTOOLS} STREQUAL "ON")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${RAF_USE_ORTOOLS})
  endif()
  find_package(ortools REQUIRED)
  set(RAF_ORTOOLS_INCLUDE ${ORTOOLS_INCLUDE_DIRS})
  message(STATUS "Found RAF_ORTOOLS_INCLUDE = ${ORTOOLS_INCLUDE_DIRS}")
  set(RAF_ORTOOLS_LIBRARY ${ORTOOLS_LIBRARIES})
  message(STATUS "Found RAF_ORTOOLS_LIBRARY = ${RAF_ORTOOLS_LIBRARY}")
endif()