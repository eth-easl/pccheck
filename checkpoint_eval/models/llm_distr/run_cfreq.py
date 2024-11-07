import os

par_dir = f"{os.path.expanduser('~')}/transformers/examples/pytorch/language-modeling"
config_file = f"{os.path.expanduser('~')}/pccheck/checkpoint_eval/models/llm_distr/ds_config.json"
cmd = f"cd {par_dir} && deepspeed --num_gpus=1 --num_nodes 1 --hostfile hostfile --master_addr 127.0.0.1 --master_port 1234  run_clm_pp_cfreq.py --deepspeed {config_file} --ds_config {config_file} --model_name_or_path facebook/opt-350m --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --do_train --per_device_train_batch_size 1 --cfreq 10 --bench_total_steps 100"
os.system(cmd)
