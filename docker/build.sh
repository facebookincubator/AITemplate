#!/bin/bash

TARGET=$1
COMMIT=$(git show --format="%H" --no-patch)
COMMIT_AUTHOR=$(git show --format="%an" --no-patch)
COMMIT_TIME=$(git show --format="%cI" --no-patch)
echo "$COMMIT" > COMMIT_INFO
echo "$COMMIT_AUTHOR" >> COMMIT_INFO
echo "$COMMIT_TIME" >> COMMIT_INFO

if [ "$TARGET" = "cuda" ]; then
    if [ "$2" = "debug" ]; then
        echo "Build in DEBUG mode with git files"
        echo "RUN apt install -y vim git" >> ./docker/Dockerfile.cuda
        echo "ADD .git /AITemplate/.git" >> ./docker/Dockerfile.cuda
    fi
    echo "Building CUDA Docker Image with tag ait:latest"
    docker build -f ./docker/Dockerfile.cuda -t ait .
elif [ "$TARGET" = "rocm" ]; then
    echo "Building ROCM Docker Image with tag ait:latest"
    docker build -f ./docker/Dockerfile.rocm -t ait .
else
    echo "Unknown target"
fi
