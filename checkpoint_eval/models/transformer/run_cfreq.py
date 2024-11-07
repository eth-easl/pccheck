import os

iters = 300
cfreqs = [100]
batchsize = 64

par_dir = f"{os.path.expanduser('~')}/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch"

for cf in cfreqs:
    proc = f"python {par_dir}/train_chfreq.py --config_file {par_dir}/wt103_base.yaml --batch_size {batchsize} --cfreq {cf} --bench_total_steps {iters}"
    os.system(proc)
