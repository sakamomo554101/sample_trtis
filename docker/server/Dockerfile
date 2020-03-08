#ARG BASE_IMAGE=tensorrtserver_build
ARG BASE_IMAGE=nvcr.io/nvidia/tensorrtserver:20.01-py3

FROM ${BASE_IMAGE}

# set working directory
WORKDIR /workspace

# set env to skip the any interactive 
ENV DEBIAN_FRONTEND=noninteractive

# update apt-get library
RUN sed -i.org -e 's|ports.ubuntu.com|jp.archive.ubuntu.com|g' /etc/apt/sources.list
RUN apt-get update -y --fix-missing

# install core packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            build-essential \
            libopencv-dev \
            libopencv-core-dev \
            libssl-dev \
            libtool \
            pkg-config \
            wget \
            libboost-dev

# copy mecab install script and build mecab
ADD ./install_script/install_mecab.sh /workspace/
RUN set -x && \
    bash ./install_mecab.sh

# setup enrtypoint
ENTRYPOINT ["/opt/tensorrtserver/nvidia_entrypoint.sh"]
