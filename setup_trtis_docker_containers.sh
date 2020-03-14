#!/usr/bin/env bash

echo "----------start setup script of sample_trtis------------"

# check parameter
BUILD_SERVER=true
BUILD_CLIENT=true
FLG_NO_CACHE=false
FLG_NO_BUILD=false
FLG_USE_GPU=false
USE_SOURCE_IMAGE=false
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
        if [ $param == "--use-gpu" ]; then
            FLG_USE_GPU=true
        fi
        if [ $param == "--use-source-image" ]; then
            USE_SOURCE_IMAGE=true
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
if "${FLG_USE_GPU}"; then
    echo "will build each container with GPU"
else
    echo "will build each container without GPU"
fi
if "${USE_SOURCE_IMAGE}"; then
    echo "will build trtis docker image using source codes"
fi

# clone trtis repository
git clone https://github.com/NVIDIA/tensorrt-inference-server

# build base trtis docker images
cd tensorrt-inference-server
# TODO : if USE_SOURCE_IMAGE is true, trtis server image and custom backend image need to be built from trtis source codes.
#if "${BUILD_SERVER}"; then
    ## build trtis server image if not existed
#    if [ ! "$(docker image ls -q tensorrtserver_build)" ]; then
#        echo "tensorrtserver_build image is not existed. start to build..."
#        docker build --pull -t tensorrtserver_build --target trtserver_build .
#    fi
    ## build trtis custom-backend image if not existed
#    if [ ! "$(docker image ls -q tensorrtserver_cbe)" ]; then
#        echo "tensorrtserver_cbe image is not existed. start to build..."
#        docker build -t tensorrtserver_cbe -f Dockerfile.custombackend .
#    fi
#fi
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

# create the trtis client, custom-backend containers(see docker-compose.yml or docker-compose.gpu.yml for more information)
## get docker-compose file name
DOCKER_COMPOSE_FILE_NAME="docker-compose.yml"
SERVER_NAME="trtis-server-container"
CUSTOM_BACKEND_NAME="trtis-custom-backend-build-container"
CLIENT_NAME="trtis-client-container"
if "${FLG_USE_GPU}"; then
    DOCKER_COMPOSE_FILE_NAME="docker-compose.gpu.yml"
    SERVER_NAME="trtis-server-container-gpu"
    CUSTOM_BACKEND_NAME="trtis-custom-backend-build-container-gpu"
fi
## create client container
if "${BUILD_CLIENT}"; then
    echo "----------  start to build trtis-client-container  ------------"
    if "${FLG_NO_CACHE}"; then
        docker-compose build --no-cache ${CLIENT_NAME}
        docker-compose up -d ${CLIENT_NAME}
    elif "${FLG_NO_BUILD}"; then
        docker-compose up -d ${CLIENT_NAME}
    else
        docker-compose up --build -d ${CLIENT_NAME}
    fi 
    docker exec -it ${CLIENT_NAME} sh -c "cd /workspace && bash build_client_library.sh"
fi
## create server containers
if "${BUILD_SERVER}"; then
    ### create custom backend build container
    echo "----------  start to build trtis-custom-backend-container  ------------"
    if "${FLG_NO_CACHE}"; then
        docker-compose -f ${DOCKER_COMPOSE_FILE_NAME} build --no-cache ${CUSTOM_BACKEND_NAME}
        docker-compose -f ${DOCKER_COMPOSE_FILE_NAME} up -d ${CUSTOM_BACKEND_NAME}
    elif "${FLG_NO_BUILD}"; then
        docker-compose -f ${DOCKER_COMPOSE_FILE_NAME} up -d ${CUSTOM_BACKEND_NAME}
    else
        docker-compose -f ${DOCKER_COMPOSE_FILE_NAME} up --build -d ${CUSTOM_BACKEND_NAME}
    fi
    OPTION=""
    if "${FLG_USE_GPU}"; then
        OPTION="USE_GPU"
    fi
    bash ./setup_custom_backend.sh ${CUSTOM_BACKEND_NAME} ${OPTION}

    ### create server container
    echo "----------  start to build trtis-server-container  ------------"
    if "${FLG_NO_CACHE}"; then
        docker-compose -f ${DOCKER_COMPOSE_FILE_NAME} build --no-cache ${SERVER_NAME}
        docker-compose -f ${DOCKER_COMPOSE_FILE_NAME} up -d ${SERVER_NAME}
    elif "${FLG_NO_BUILD}"; then
        docker-compose -f ${DOCKER_COMPOSE_FILE_NAME} up -d ${SERVER_NAME}
    else
        docker-compose -f ${DOCKER_COMPOSE_FILE_NAME} up --build -d ${SERVER_NAME}
    fi

    ### create data uploader container
    echo "----------  start to build data-uploader-container  ------------"
    if "${FLG_NO_CACHE}"; then
        docker-compose -f ${DOCKER_COMPOSE_FILE_NAME} build --no-cache "data-uploader-container"
        docker-compose -f ${DOCKER_COMPOSE_FILE_NAME} up -d "data-uploader-container"
    elif "${FLG_NO_BUILD}"; then
        docker-compose -f ${DOCKER_COMPOSE_FILE_NAME} up -d "data-uploader-container"
    else
        docker-compose -f ${DOCKER_COMPOSE_FILE_NAME} up --build -d "data-uploader-container"
    fi
fi

echo "----------end setup script of sample_trtis------------"
