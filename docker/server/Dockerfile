ARG BASE_IMAGE=tensorrtserver_build

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

# build opencv library
RUN set -x && \
    apt-get install -y libopencv-dev libopencv-core-dev

# copy mecab install script and build mecab
ADD ./install_script/install_mecab.sh /workspace/
RUN set -x && \
    bash ./install_mecab.sh

# setup enrtypoint
ENTRYPOINT ["/opt/tensorrtserver/nvidia_entrypoint.sh"]
