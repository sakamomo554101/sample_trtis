ARG BASE_IMAGE=tensorrtserver_cbe

FROM ${BASE_IMAGE}

# set working directory
WORKDIR /workspace

# set env to skip the any interactive 
ENV DEBIAN_FRONTEND=noninteractive

# update apt-get library
RUN apt-get update -y --fix-missing

# install core packages
RUN set -x && \
    apt-get install -y wget

# copy build script
ADD ./custom_backend/build_custom_backend.sh /workspace/

# copy mecab install script and build mecab
ADD ./install_script/install_mecab.sh /workspace/
RUN set -x && \
    bash ./install_mecab.sh
