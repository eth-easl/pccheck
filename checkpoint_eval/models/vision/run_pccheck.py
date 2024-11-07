import os

iters = 200
cfreqs = [10]
model = "vgg16"
batchsize = 32

max_async = 4
num_threads = 2
psize = 8

for cf in cfreqs:
    print(f"Run with cfreq {cf}")
    proc = f"python3.9 train_pccheck.py --dataset imagenet  --batchsize {batchsize} --arch {model} --cfreq {cf} --bench_total_steps {iters} --max-async {max_async} --num-threads {num_threads} --psize {psize} --c_lib_path /home/fot/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
    os.system(proc)
