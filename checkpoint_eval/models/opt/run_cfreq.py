import os

iters = 300
cfreqs = [50]  # [50, 75, 100, 150, 250, 500, 1000]
model = "facebook/opt-1.3b"

par_dir = f"{os.path.expanduser('~')}/transformers/examples/pytorch/language-modeling"

for cf in cfreqs:
    os.system(f"rm -rf {par_dir}/output/*")
    proc = f"python3.9 {par_dir}/run_clm_cfreq.py --model_name_or_path {model} --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters}"
    os.system(proc)
