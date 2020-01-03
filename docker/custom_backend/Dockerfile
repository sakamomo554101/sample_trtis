ARG BASE_IMAGE=tensorrtserver_cbe

FROM ${BASE_IMAGE}

# set working directory
WORKDIR /workspace

# copy build script
ADD ./build_custom_backend.sh /workspace/

# add the build target source into CMakeLists.txt
RUN echo "add_subdirectory(../../custom_backend/sample_instance src/custom/sample_instance)" \
    >> /workspace/build/trtis-custom-backends/CMakeLists.txt
