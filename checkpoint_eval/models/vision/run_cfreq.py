import os

iters = 1000
cfreqs = [0]
model = "vgg16"
batchsize = 32

for cf in cfreqs:
    print(f"Run with cfreq {cf}")
    proc = f"python3.9 train_checkfreq.py --dataset imagenet  --batchsize {batchsize} --arch {model} --cfreq {cf} --bench_total_steps {iters}"
    os.system(proc)
