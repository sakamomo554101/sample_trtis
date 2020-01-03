#!/usr/bin/env bash

cd build
cmake -DTRTIS_ENABLE_GPU=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX:PATH=/workspace/install/custom-backend-sdk 
make -j16 trtis-custom-backends
