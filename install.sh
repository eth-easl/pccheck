#!/bin/bash

cd checkpoint_eval/gpm/ && make clean && make && cd ../../
cd checkpoint_eval/pccheck/ && make clean && make libtest_ssd.so && cd ../../

python3.9 -m pip install -r requirements.txt
python3.9 -m pip install -e .  --config-settings editable_mode=compat
