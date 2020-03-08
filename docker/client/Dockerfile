ARG BASE_IMAGE=tensorrtserver_client

FROM ${BASE_IMAGE}

# set working directory
WORKDIR /workspace

# copy build script
ADD ./build_client_library.sh /workspace/

# add the build target source into CMakeLists.txt
# Now, nothing to do.

# set locale of japanese
RUN sed -i.org -e 's|ports.ubuntu.com|jp.archive.ubuntu.com|g' /etc/apt/sources.list
RUN apt-get update \
    && apt-get install -y locales \
    && locale-gen ja_JP.UTF-8 \
    && echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

# install python library
RUN python -m pip install streamlit opencv-python
