import os

iters = 300
cfreqs = [100]
batchsize = 3

par_dir = f"{os.path.expanduser('~')}/DeepLearningExamples/PyTorch/LanguageModeling/BERT"

for cf in cfreqs:
        proc = f"python {par_dir}/run_squad_chfreq.py --bert_model=bert-large-uncased --train_batch_size 3 --output_dir output --vocab_file {par_dir}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt  --config_file {par_dir}/bert_configs/large.json --cfreq {cf} --bench_total_steps {iters} --max_steps {iters} --do_train --train_file {par_dir}/download/squad/v1.1/train-v1.1.json"
        os.system(proc)
