#!/usr/bin/env bash

echo "----------start setup script of sample_trtis------------"

# check parameter
BUILD_SERVER=true
BUILD_CLIENT=true
FLG_NO_CACHE=false
FLG_NO_BUILD=false
for param in $@
    do
        if [ $param == "--no-cache" ]; then
            FLG_NO_CACHE=true
        fi
        if [ $param == "--no-build" ]; then
            FLG_NO_BUILD=true
        fi
        if [ $param == "--only-client" ]; then
            BUILD_SERVER=false
        fi
        if [ $param == "--only-server" ]; then
            BUILD_CLIENT=false
        fi
    done

if "${BUILD_SERVER}"; then
    echo "will build server and custom-backend containers"
fi
if "${BUILD_CLIENT}"; then
    echo "will build client container"
fi
if "${FLG_NO_BUILD}"; then
    echo "will not build docker image"
fi
if "${FLG_NO_CACHE}"; then
    echo "will not use docker cache when docker image is built"
fi

# clone trtis repository
git clone https://github.com/NVIDIA/tensorrt-inference-server

# build base trtis docker images
cd tensorrt-inference-server
if "${BUILD_SERVER}"; then
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
fi
if "${BUILD_CLIENT}"; then
    ## build trtis client image if not existed
    if [ ! "$(docker image ls -q tensorrtserver_client)" ]; then
        echo "tensorrtserver_client image is not existed. start to build..."
        docker build -t tensorrtserver_client -f Dockerfile.client .
    fi
fi
cd ../

# stop and delete container if existed
docker-compose down

# create the trtis client, custom-backend containers(see docker-compose.yml for more information)
## create client container
if "${BUILD_CLIENT}"; then
    echo "----------  start to build trtis-client-container  ------------"
    if "${FLG_NO_CACHE}"; then
        docker-compose build --no-cache trtis-client-container
        docker-compose up -d trtis-client-container
    elif "${FLG_NO_BUILD}"; then
        docker-compose up -d trtis-client-container
    else
        docker-compose up --build -d trtis-client-container
    fi
    docker exec -it trtis-client-container sh -c "cd /workspace && bash build_client_library.sh"
fi
## create server containers
if "${BUILD_SERVER}"; then
    ### create custom backend build container
    echo "----------  start to build trtis-custom-backend-container  ------------"
    if "${FLG_NO_CACHE}"; then
        docker-compose build --no-cache trtis-custom-backend-build-container
        docker-compose up -d trtis-custom-backend-build-container
    elif "${FLG_NO_BUILD}"; then
        docker-compose up -d trtis-custom-backend-build-container
    else
        docker-compose up --build -d trtis-custom-backend-build-container
    fi
    bash ./setup_custom_backend.sh

    ### create server container
    echo "----------  start to build trtis-server-container  ------------"
    if "${FLG_NO_CACHE}"; then
        docker-compose build --no-cache trtis-server-build-container
        docker-compose up -d trtis-server-build-container
    elif "${FLG_NO_BUILD}"; then
        docker-compose up -d trtis-server-build-container
    else
        docker-compose up --build -d trtis-server-build-container
    fi
fi

echo "----------end setup script of sample_trtis------------"
