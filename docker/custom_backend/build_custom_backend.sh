#!/usr/bin/env bash

ENABLE_GPU="OFF"
PARAM_COUNT=$#
OPTION=$1
if [ ${PARAM_COUNT} = 1 -a "${OPTION}" = "USE_GPU" ]; then
    ENABLE_GPU="ON"
fi

# dlibのリポジトリを取ってくる(TODO : もっと良いやり方がありそう)
if [ ! -e "/workspace/custom_backend/face_recognition_model/dlib" ]; then
    mkdir /workspace/custom_backend/face_recognition_model/dlib
    git clone https://github.com/davisking/dlib /workspace/custom_backend/face_recognition_model/dlib
fi

# TODO : get parameter
cd build
cmake -DTRTIS_ENABLE_GPU=${ENABLE_GPU} \
      -DTRTIS_ENABLE_METRICS_GPU=${ENABLE_GPU} \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX:PATH=/workspace/install/custom-backend-sdk 
make -j16 trtis-custom-backends
export VERSION=`cat /workspace/VERSION`
cd install && tar zcf /workspace/v$VERSION.custombackend.tar.gz custom-backend-sdk
