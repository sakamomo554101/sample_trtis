#!/usr/bin/env bash

echo "----------start setup script of sample_trtis------------"

# clone trtis repository
git clone https://github.com/NVIDIA/tensorrt-inference-server

# build base trtis docker images
cd tensorrt-inference-server
## build trtis server image if not existed
if [ ! "$(docker image ls -q tensorrtserver_build)" ]; then
    echo "tensorrtserver_build image is not existed. start to build..."
    docker build --pull -t tensorrtserver_build --target trtserver_build .
fi
## build trtis custom-backend image if not existed
if [ ! "$(docker image ls -q tensorrtserver_cbe)" ]; then
    echo "tensorrtserver_cbe image is not existed. start to build..."
    docker build -t tensorrtserver_cbe -f Dockerfile.custombackend .
fi
## build trtis client image
if [ ! "$(docker image ls -q tensorrtserver_client)" ]; then
    echo "tensorrtserver_client image is not existed. start to build..."
    docker build -t tensorrtserver_client -f Dockerfile.client .
fi
cd ../

# create the trtis client, custom-backend containers(see docker-compose.yml for more information)
docker-compose up --build -d trtis-client-container
docker-compose up --build -d trtis-custom-backend-build-container

# build and copy model files
bash ./build_custom_models.sh

# create the trtis server container
docker-compose up --build -d trtis-server-build-container

echo "----------end setup script of sample_trtis------------"
