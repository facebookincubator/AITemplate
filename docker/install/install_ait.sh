#!/bin/bash

cd /AITemplate/python
python3 setup.py bdist_wheel
pip3 install --no-input /AITemplate/python/dist/*.whl
