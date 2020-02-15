# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
cmake_minimum_required (VERSION 3.5)

#
# mecab_model.so
#
configure_file(libmecab_model.ldscript libmecab_model.ldscript COPYONLY)

set(CMAKE_CXX_FLAGS, "${CMAKE_CXX_FLAGS} `mecab-config --cflags`")

add_library(
  mecab_model SHARED
  mecab_model.cc
)
set_target_properties(
  mecab_model
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libmecab_model.ldscript
)
set_target_properties(
  mecab_model
  PROPERTIES LINK_FLAGS "-Wl,--version-script libmecab_model.ldscript"
)
target_link_libraries(
  mecab_model
  PRIVATE custombackend
)
# TODO : get library path from "mecab-config" command
target_link_libraries(
  mecab_model
  PRIVATE "-L/usr/local/lib -lmecab -lstdc++"
)

install(
  TARGETS mecab_model
  LIBRARY DESTINATION lib
)
