#!/usr/bin/env bash

CUSTOM_BACKEND_CONTAINER_NAME="trtis-custom-backend-build-container"

## build custom backend files
docker exec -it $CUSTOM_BACKEND_CONTAINER_NAME sh -c "cd /workspace && bash build_custom_backend.sh"

# move so object from custom_backend_container
CONTAINER_ID=$(docker ps -a -f name=$CUSTOM_BACKEND_CONTAINER_NAME --format "{{.ID}}")
docker cp $CONTAINER_ID:/workspace/build/trtis-custom-backends/src/custom/sample_instance/libsample_instance.so \
          model_repository/sample_instance/1/
docker cp $CONTAINER_ID:/workspace/build/trtis-custom-backends/src/custom/sequence/libsequence.so \
          model_repository/simple_sequence/1/
docker cp $CONTAINER_ID:/workspace/build/trtis-custom-backends/src/custom/sample_sequence/libsample_sequence.so \
          model_repository/sample_sequence/1/
docker cp $CONTAINER_ID:/workspace/build/trtis-custom-backends/src/custom/mecab_model/libmecab_model.so \
          model_repository/mecab_model/1/
docker cp $CONTAINER_ID:/workspace/build/trtis-custom-backends/src/custom/face_recognition_model/libface_recognition_model.so \
          model_repository/face_recognition_model/1/
