#!/usr/bin/env bash

echo "start setup_custom_backend.sh ...."

# check parameter
if [ $# != 1 -a $# != 2 ]; then
    echo "parameter count is not matched! please set the one or two parameters!"
    echo "1st : the custom backend container name"
    echo "2nd : gpu enable or not"
    exit 1
fi

CUSTOM_BACKEND_CONTAINER_NAME=$1
OPTION=""
if [ $# == 2 ]; then
    OPTION=$2
fi

# create model folder and copy model setting files.
mkdir -p model_repository/sample_instance/1
cp -f ./model_settings/sample_instance/config.pbtxt \
      ./model_repository/sample_instance/
mkdir -p model_repository/simple_sequence/1
cp -f ./tensorrt-inference-server/qa/L0_backend_release/simple_seq_models/simple_sequence/config.pbtxt \
      ./model_repository/simple_sequence/
mkdir -p model_repository/sample_sequence/1
cp -f ./model_settings/sample_sequence/config.pbtxt \
      ./model_repository/sample_sequence/
mkdir -p model_repository/mecab_model/1
cp -f ./model_settings/mecab_model/config.pbtxt \
      ./model_repository/mecab_model/
## create face recognition model folder and move config and model file.
mkdir -p model_repository/face_recognition_model/1
cp -f ./model_settings/face_recognition_model/config.pbtxt \
      ./model_repository/face_recognition_model/
if [ ! -e "model_repository/face_recognition_model/dlib_face_recognition_resnet_model_v1.dat" ]; then
    curl -O http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
    bzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2
    mv dlib_face_recognition_resnet_model_v1.dat model_repository/face_recognition_model/
fi
if [ ! -e "model_repository/face_recognition_model/shape_predictor_5_face_landmarks.dat" ]; then
    curl -O http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_5_face_landmarks.dat.bz2
    mv shape_predictor_5_face_landmarks.dat model_repository/face_recognition_model/
fi

# create the face dataset folder if needed
mkdir -p dataset/face/image

# build the custom-backend codes
## add the build target source into CMakeLists.txt(TODO : write target into CMakeLists.txt if needed.)
docker exec -it $CUSTOM_BACKEND_CONTAINER_NAME sh -c \
    "echo 'add_subdirectory(../../custom_backend/sample_instance src/custom/sample_instance)' >> /workspace/build/trtis-custom-backends/CMakeLists.txt"
docker exec -it $CUSTOM_BACKEND_CONTAINER_NAME sh -c \
    "echo 'add_subdirectory(../../custom_backend/sample_sequence src/custom/sample_sequence)' >> /workspace/build/trtis-custom-backends/CMakeLists.txt"
docker exec -it $CUSTOM_BACKEND_CONTAINER_NAME sh -c \
    "echo 'add_subdirectory(../../custom_backend/mecab_model src/custom/mecab_model)' >> /workspace/build/trtis-custom-backends/CMakeLists.txt"
docker exec -it $CUSTOM_BACKEND_CONTAINER_NAME sh -c \
    "echo 'add_subdirectory(../../custom_backend/face_recognition_model src/custom/face_recognition_model)' >> /workspace/build/trtis-custom-backends/CMakeLists.txt"

## build custom backend files
docker exec -it $CUSTOM_BACKEND_CONTAINER_NAME sh -c "cd /workspace && bash build_custom_backend.sh ${OPTION}"

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
