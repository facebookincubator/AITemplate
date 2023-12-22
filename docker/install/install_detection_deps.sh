#!/bin/bash

apt install -y ffmpeg libsm6 libxext6 wget
pip3 install yacs
pip3 install opencv-python
pip3 install tqdm
pip3 install timm
pip3 install transformers==4.25.0
pip3 install diffusers==0.24.0