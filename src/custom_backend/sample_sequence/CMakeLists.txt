# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
cmake_minimum_required (VERSION 3.5)

#
# sample_sequence.so
#
configure_file(libsample_sequence.ldscript libsample_sequence.ldscript COPYONLY)

add_library(
  sample_sequence SHARED
  sample_sequence.cc
)
set_target_properties(
  sample_sequence
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libsample_sequence.ldscript
)
set_target_properties(
  sample_sequence
  PROPERTIES LINK_FLAGS "-Wl,--version-script libsample_sequence.ldscript"
)
target_link_libraries(
  sample_sequence
  PRIVATE custombackend
)

install(
  TARGETS sample_sequence
  LIBRARY DESTINATION lib
)
