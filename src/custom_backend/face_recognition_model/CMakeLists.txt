# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
cmake_minimum_required (VERSION 3.5)

# need opencv
find_package(OpenCV REQUIRED)

#
# face_recognition_model.so
#
configure_file(libface_recognition_model.ldscript libface_recognition_model.ldscript COPYONLY)
# cuda
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter")  # need to rewrite CXX_FLAGS to build dlib with CUDA
#option(DLIB_USE_CUDA "" 0)  # TODO : DLIB_USE_CUDA should be set 0(CPU) or 1(GPU) according to the build options.
add_subdirectory(dlib)
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
