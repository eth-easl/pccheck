import os

iters = 300
cfreqs = [0]
batchsize = 64

for cf in cfreqs:
    proc = f"python ~/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/train_gpm.py --config_file wt103_base.yaml --batch_size {batchsize} --cfreq {cf} --bench_total_steps {iters}"
    os.system(proc)
