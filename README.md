# Sample TensorRT Inference Server codes

## Introduction

Sample TensorRT Inference Server(=TRTIS) source codes.

## Environment

* Docker
* Docker-Compose
* git

## Container Architecture

This repository create the following container.

* Server Container
* Client Container
* Custom-Backend Container
    * This container build the source code of TRTIS CustomInstance.

## Setup

### Build & Run each containers

need to execute the following command.
```
$ bash setup_trtis_docker_containers.sh
```

After above command is executed, all containers will be created and run.

### Fetch models

If you use the default trtis models(ex. ResNet50, sequence..), need to execute the following command.
```
$ bash fetch_default_models.sh
```
