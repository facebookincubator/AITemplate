#!/bin/bash

#profile
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark_ait.py

#1GCD
HIP_VISIBLE_DEVICES=0 python3 benchmark_ait.py --batch-size "$1"

#2GCD
HIP_VISIBLE_DEVICES=0 python3 benchmark_ait.py --batch-size "$1" &
HIP_VISIBLE_DEVICES=1 python3 benchmark_ait.py --batch-size "$1" && fg
