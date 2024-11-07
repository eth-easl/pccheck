import os

iters = 300
cfreqs = [100, 150, 250, 500]
batchsize = 3

for cf in cfreqs:
        proc = f"python run_squad_gpm.py --bert_model=bert-large-uncased --train_batch_size 3 --output_dir output --vocab_file download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt  --config_file bert_configs/large.json --cfreq {cf} --bench_total_steps {iters} --max_steps {iters} --do_train --train_file download/squad/v1.1/train-v1.1.json"
        os.system(proc)
