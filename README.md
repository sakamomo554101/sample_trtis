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
$ bash setup_trtis_docker_containers.sh --no-build
```

### setup docker container without using docker cache

need to execute the following command.
```
$ bash setup_trtis_docker_containers.sh --no-cache
```

### setup only client docker container

```
$ bash setup_trtis_docker_containers.sh --only-client
```

### setup only server docker container

```
$ bash setup_trtis_docker_containers.sh --only-server
```

### use face recognition function

need to prepare the following steps

#### prepare the dataset with face image

The following directory is need to create on the root folder of this repository.

"dataset/face/image"

You need to store the image files with face in above folder.

#### prepare the config csv file

You need to store the config file of csv format in following path.

"dataset/face/face.csv"

This csv file format is following.

* first column is "image file name"
* second column is "face name"


```
hoge.jpg, fuga
taro.jpg, hanako

...

```
