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

### Execute trtis-client script

* attach the trtis client container
```
$ docker exec -it trtis-client-container bash
```

* execute client script in trtis client container(ex. simple model)
```
$ python3 src/clients/python/simple_client.py
```

* execute custom client script in trtis client container(ex. sample_instance model)
```
$ python3 custom_client/python/sample_instance_client.py
```

## Other information

### setup docker container without building docker images

need to execute the following command.
```
$ bash setup_trtis_docker_containers.sh -b
```

### setup docker container without using docker cache

need to execute the following command.
```
$ bash setup_trtis_docker_containers.sh -c
```
