#!/bin/bash

TAG=
NAME=
MODELS=
EXPERIMENTS=
CORPORA=
DEVICE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --image-tag=*)
            TAG="${1#*=}"
            ;;
        --docker-name=*)
            NAME="${1#*=}"
            ;;
        --models-dir=*)
            MODELS="${1#*=}"
            ;;
        --experiments-dir=*)
            EXPERIMENTS="${1#*=}"
            ;;
        --corpora-dir=*)
            CORPORA="${1#*=}"
            ;;
        --gpu=*)
            DEVICE="${1#*=}"
            ;;
        *)
            echo "Error: Invalid argument: $1"
            exit 1
  esac
  shift
done

docker run --gpus '"device=$DEVICE"' --name $NAME -it --shm-size=4g -v $PWD/tune-n-distill:/app -v $MODELS:/models -v $EXPERIMENTS:/experiments -v $CORPORA:/corpora $TAG bash
