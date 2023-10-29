#!/bin/bash

docker run --gpus all -it --volume="$HOME/AITemplate:/AITemplate:rw" ait:latest