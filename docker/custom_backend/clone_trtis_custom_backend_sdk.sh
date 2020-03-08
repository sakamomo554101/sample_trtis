#!/usr/bin/env bash

TRITON_FOLDER_NAME=tensorrt-inference-server
mkdir -p src/backends/custom/
git clone https://github.com/NVIDIA/tensorrt-inference-server && \
    cp -rf ${TRITON_FOLDER_NAME}/VERSION ./ && \
    cp -rf ${TRITON_FOLDER_NAME}/build ./ && \
    cp ${TRITON_FOLDER_NAME}/src/backends/custom/custom.h src/backends/custom/ && \
    cp -rf ${TRITON_FOLDER_NAME}/src/custom src/ && \
    cp -rf ${TRITON_FOLDER_NAME}/src/core src/
