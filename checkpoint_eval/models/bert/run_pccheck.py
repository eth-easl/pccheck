import os

iters = 300
cfreqs = [0]
batchsize = 3

max_async = 4
num_threads = 2

for cf in cfreqs:
    proc = f"python3.9 run_squad_pccheck.py --bert_model=bert-large-uncased --train_batch_size {batchsize} --output_dir output --vocab_file download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt  --config_file bert_configs/large.json --do_train --train_file download/squad/v1.1/train-v1.1.json --cfreq {cf} --max_async {max_async} --num_threads {num_threads} --bench_total_steps {iters} --c_lib_path /home/fot/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
    os.system(proc)
