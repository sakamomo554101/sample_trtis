ARG BASE_IMAGE=tensorrtserver_client

FROM ${BASE_IMAGE}

# set working directory
WORKDIR /workspace

# copy build script
ADD ./build_client_library.sh /workspace/

# add the build target source into CMakeLists.txt
# Now, nothing to do.
