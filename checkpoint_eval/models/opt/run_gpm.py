import os

iters = 300
cfreqs = [75,100,150]
model = "facebook/opt-1.3b"

for cf in cfreqs:
    os.system("rm -rf output/*")
    proc = f"python run_clm_gpm.py --model_name_or_path {model} --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters}"
    os.system(proc)
