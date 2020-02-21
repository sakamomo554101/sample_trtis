#!/usr/bin/env bash

echo "----------start setup script of sample_trtis------------"

# check parameter
while getopts "bc" opts
do
    case $opts in
        b) FLG_NO_BUILD="TRUE" ;;
        c) FLG_NO_CACHE="TRUE" ;;
    esac
done

# for debug
if [ ! -z "$FLG_NO_CACHE" ]; then
    echo "not use cache"
elif [ ! -z "$FLG_NO_BUILD" ]; then
    echo "no build"
else
    echo "use cache"
fi

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

# stop and delete container if existed
docker-compose down

# create the trtis client, custom-backend containers(see docker-compose.yml for more information)
if [ ! -z "$FLG_NO_BUILD" ]; then
    docker-compose up -d trtis-client-container
    docker-compose up -d trtis-custom-backend-build-container
else
    docker-compose up --build -d trtis-client-container
    docker-compose up --build -d trtis-custom-backend-build-container
fi

# build and copy model files
bash ./build_all_containers.sh

# create the trtis server container
if [ ! -z "$FLG_NO_CACHE" ]; then
    docker-compose build --no-cache trtis-server-build-container
    docker-compose up -d trtis-server-build-container
elif [ ! -z "$FLG_NO_BUILD" ]; then
    docker-compose up -d trtis-server-build-container
else
    docker-compose up --build -d trtis-server-build-container
fi

echo "----------end setup script of sample_trtis------------"
