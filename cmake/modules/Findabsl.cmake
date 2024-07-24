# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Find the absl libraries

find_path(ABSL_INCLUDE_DIRS
  NAMES absl
  HINTS
  $ENV{ABSL_DIR}/include
)

set(ABSL_LIBS
    absl_bad_any_cast_impl #ABSL_LNK
    absl_bad_optional_access
    absl_bad_variant_access
    absl_base
    absl_city
    absl_civil_time
    absl_debugging_internal
    absl_demangle_internal
    absl_examine_stack
    absl_failure_signal_handler
    absl_graphcycles_internal
    absl_hash
    absl_hashtablez_sampler
    absl_int128
    absl_leak_check
    absl_malloc_internal
    absl_raw_hash_set
    absl_spinlock_wait
    absl_stacktrace
    absl_str_format_internal
    absl_strings
    absl_strings_internal
    absl_symbolize
    absl_synchronization
    absl_throw_delegate
    absl_time
    absl_time_zone)

foreach(X ${ABSL_LIBS})
    find_library(LIB_${X} NAME ${X} HINTS $ENV{ABSL_DIR}/lib)
    message(STATUS "${X} lib found here : ${LIB_${X}}")
    set(ABSL_LIBRARIES ${ABSL_LIBRARIES} ${LIB_${X}})
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(absl DEFAULT_MSG ABSL_INCLUDE_DIRS ABSL_LIBRARIES)
