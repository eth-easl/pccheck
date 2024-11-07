#!/bin/bash

python3.9 $HOME/pccheck/checkpoint_eval/models/vision/train_pccheck.py --dataset imagenet  --batchsize 32 --arch vgg16 --cfreq 50 --bench_total_steps 500 --max-async 4 \
 --num-threads 2 --c_lib_path $HOME/pccheck/checkpoint_eval/pccheck/libtest_ssd.so
