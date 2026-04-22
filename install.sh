#!/bin/bash

cd checkpoint_eval/gpm/ && make clean && make && cd ../../
cd checkpoint_eval/pccheck/ && make clean && make libtest_ssd.so && cd ../../

#python3.12 -m pip install -r requirements.txt # not needed if installed from docker file
python3.12 -m pip install -e .  --config-settings editable_mode=compat
