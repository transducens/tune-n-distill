#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Image name suffix expected."
    exit
fi

#TAG=transducens_agaliano_ft_kd:0.2
TAG=$1
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --tag=$TAG .