#!/usr/bin/env bash

CUSTOM_BACKEND_CONTAINER_NAME="trtis-custom-backend-build-container"

# create model folder if not existed
mkdir -p model_repository/sample_instance/1
if [ ! -e ./model_repository/sample_instance/config.pbtxt ]; then
    cp ./model_settings/sample_instance/config.pbtxt ./model_repository/sample_instance/
fi

# TODO : build the server codes

# build the client codes
echo "----------  start to build trtis-client-container  ------------"
docker exec -it trtis-client-container sh -c "cd /workspace && bash build_client_library.sh"

# build the custom-backend codes
echo "----------  start to build trtis-custom-backend-build-container  ------------"
docker exec -it $CUSTOM_BACKEND_CONTAINER_NAME sh -c "cd /workspace && bash build_custom_backend.sh"

# move so object from custom_backend_container
CONTAINER_ID=$(docker ps -a -f name=$CUSTOM_BACKEND_CONTAINER_NAME --format "{{.ID}}")
docker cp $CONTAINER_ID:/workspace/build/trtis-custom-backends/src/custom/sample_instance/libsample_instance.so \
          model_repository/sample_instance/1/