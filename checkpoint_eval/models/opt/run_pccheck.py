import os

iters = 300
cfreqs = [0]
model = "facebook/opt-1.3b"

max_async = 2
num_threads = 2
psize = 8
dram_ratio = 2.0

for cf in cfreqs:
        os.system("rm -rf output/*")
        proc = f"python3.9 run_clm_pccheck.py --model_name_or_path {model} --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --max_async {max_async} --num_threads {num_threads} --psize {psize} --dram_ratio {dram_ratio} --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} --c_lib_path /home/fot/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
        print(proc)
        os.system(proc)
