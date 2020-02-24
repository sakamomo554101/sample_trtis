# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
cmake_minimum_required (VERSION 3.5)

# need opencv
find_package(OpenCV REQUIRED)

#
# face_recognition_model.so
#
configure_file(libface_recognition_model.ldscript libface_recognition_model.ldscript COPYONLY)
add_subdirectory(dlib)
target_compile_options(dlib PRIVATE -fno-aggressive-loop-optimizations -Wno-maybe-uninitialized)
set_target_properties(
  dlib 
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

add_library(
  face_recognition_model SHARED
  face_wrapper.cc
  face_recognition_model.cc
)
target_include_directories(face_recognition_model PRIVATE ${OpenCV_INCLUDE_DIRS})
set_target_properties(
  face_recognition_model
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libface_recognition_model.ldscript
)
set_target_properties(
  face_recognition_model
  PROPERTIES LINK_FLAGS "-Wl,--version-script libface_recognition_model.ldscript"
)
target_link_libraries(
  face_recognition_model
  PRIVATE custombackend
  PRIVATE ${OpenCV_LIBS}
)
target_link_libraries(
  face_recognition_model
  PRIVATE dlib
)

install(
  TARGETS face_recognition_model
  LIBRARY DESTINATION lib
)
