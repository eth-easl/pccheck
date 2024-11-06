import os

iters = 300
cfreqs = [0]
batch_size = 64

max_async = 4
num_threads = 2

for cf in cfreqs:
    proc = f"python3.9 train_pccheck.py --config_file wt103_base.yaml --batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} --max_async {max_async} --num_threads {num_threads} --c_lib_path /home/fot/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
    os.system(proc)
