ARG BASE_IMAGE=tensorrtserver_build

FROM ${BASE_IMAGE}

# set working directory
WORKDIR /workspace

# set env to skip the any interactive 
ENV DEBIAN_FRONTEND=noninteractive

# update apt-get library
RUN apt-get update -y --fix-missing

# build opencv library
RUN apt-get install -y libopencv-dev libopencv-core-dev

# setup enrtypoint
ENTRYPOINT ["/opt/tensorrtserver/nvidia_entrypoint.sh"]
