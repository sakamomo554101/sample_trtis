#!/usr/bin/env bash

# update apt-get library
sed -i.org -e 's|ports.ubuntu.com|jp.archive.ubuntu.com|g' /etc/apt/sources.list
apt-get update -y --fix-missing

# install core packages
apt-get update && \
apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        cmake \
        git \
        libopencv-dev \
        libopencv-core-dev \
        libssl-dev \
        libtool \
        pkg-config \
        wget \
        libboost-dev
