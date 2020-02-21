#!/usr/bin/env bash

# dlibのリポジトリを取ってくる(TODO : もっと良いやり方がありそう)
if [ ! -e "/workspace/custom_backend/face_recognition_model/dlib" ]; then
    mkdir /workspace/custom_backend/face_recognition_model/dlib
    git clone https://github.com/davisking/dlib /workspace/custom_backend/face_recognition_model/dlib
fi

cd build
cmake -DTRTIS_ENABLE_GPU=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX:PATH=/workspace/install/custom-backend-sdk 
make -j16 trtis-custom-backends
