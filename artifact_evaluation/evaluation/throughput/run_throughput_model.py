import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

home_dir = os.path.expanduser("~")
lib_path = f"{home_dir}/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"

cfreqs = [0,1,10,25,50,75,100]

model_scripts_dir = {
    "transformer": f"{home_dir}/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch",
    "bert": f"{home_dir}/DeepLearningExamples/PyTorch/LanguageModeling/BERT",
    "opt13": f"{home_dir}/transformers/examples/pytorch/language-modeling"
}

batch_size_dir = {"opt13": 1, "transformer": 64, "bert": 3}
iters_dir = {"opt13": 200, "transformer": 300, "bert": 300}

label_dict = {
    "cfreq": "CheckFreq",
    "gpm": "GPM",
    "pccheck": "PCcheck"
}

WARMUP = 3 # iterations

def run_opt():
    os.makedirs("opt13", exist_ok=True)
    script_dir = model_scripts_dir[model]
    batch_size = batch_size_dir[model]
    iters = iters_dir[model]

    # run cfreq
    print("Run for CheckFreq")
    for cf in cfreqs:
        os.system("rm -rf output")
        print(f"Checkpoint Frequency {cf}")
        proc = f"python3.9 {script_dir}/run_clm_cfreq.py --model_name_or_path facebook/opt-1.3b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --per_device_train_batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} > opt13/log_opt13_cfreq_{cf}.txt"
        os.system(proc)

    # run GPM
    print("Run for GPM")
    for cf in cfreqs:
        os.system("rm -rf output")
        print(f"Checkpoint Frequency {cf}")
        proc = f"python3.9 {script_dir}/run_clm_gpm.py --model_name_or_path facebook/opt-1.3b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --per_device_train_batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} > opt13/log_opt13_gpm_{cf}.txt"
        os.system(proc)

    # # run PCcheck
    print("Run for PCcheck")
    for cf in cfreqs:
        os.system("rm -rf output")
        print(f"Checkpoint Frequency {cf}")
        proc = f"python3.9 {script_dir}/run_clm_pccheck.py --model_name_or_path facebook/opt-1.3b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --max_async 2 --num_threads 2 --psize 4 --per_device_train_batch_size 1 --cfreq {cf} --bench_total_steps {iters} --c_lib_path {lib_path} > opt13/log_opt13_pccheck_{cf}.txt"
        os.system(proc)



def run_transformer():
    os.makedirs("transformer", exist_ok=True)
    script_dir = model_scripts_dir[model]
    batch_size = batch_size_dir[model]
    iters = iters_dir[model]
    this_dir = f"{home_dir}/pccheck/artifact_evaluation/evaluation/throughput"

    # run cfreq
    print("Run for CheckFreq")
    for cf in cfreqs:
        print(f"Checkpoint Frequency {cf}")
        proc = f"cd {script_dir} && python3.9 train_checkfreq.py --config_file wt103_base.yaml --batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} > {this_dir}/transformer/log_transformer_cfreq_{cf}.txt"
        os.system(proc)

    print("Run for GPM")
    # run gpm
    for cf in cfreqs:
        print(f"Checkpoint Frequency {cf}")
        proc = f"cd {script_dir} && python3.9 train_gpm.py --config_file wt103_base.yaml --batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} > {this_dir}/transformer/log_transformer_gpm_{cf}.txt"
        os.system(proc)

    print("Run for PCCheck")
    # run pccheck
    for cf in cfreqs:
        print(f"Checkpoint Frequency {cf}")
        proc = f"cd {script_dir} && python3.9 train_pccheck.py --config_file wt103_base.yaml --batch_size {batch_size} --cfreq {cf} --bench_total_steps {iters} --max_async 4 --num_threads 2 --c_lib_path {lib_path} > {this_dir}/transformer/log_transformer_pccheck_{cf}.txt"
        os.system(proc)


def run_bert():
    os.makedirs("bert", exist_ok=True)
    script_dir = model_scripts_dir[model]
    batch_size = batch_size_dir[model]
    iters = iters_dir[model]

    print("Run for CheckFreq")
    for cf in cfreqs:
        print(f"Checkpoint Frequency {cf}")
        proc = f"python3.9 {script_dir}/run_squad_chfreq.py --bert_model=bert-large-uncased --train_batch_size {batch_size} --output_dir output --vocab_file {script_dir}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt  --config_file {script_dir}/bert_configs/large.json --cfreq {cf} --bench_total_steps {iters} --max_steps {iters} --do_train --train_file {script_dir}/download/squad/v1.1/train-v1.1.json > bert/log_bert_cfreq_{cf}.txt"
        os.system(proc)

    print("Run for GPM")
    # run gpm
    for cf in cfreqs:
        print(f"Checkpoint Frequency {cf}")
        proc = f"python3.9 {script_dir}/run_squad_gpm.py --bert_model=bert-large-uncased --train_batch_size {batch_size} --output_dir output --vocab_file {script_dir}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt  --config_file {script_dir}/bert_configs/large.json --cfreq {cf} --bench_total_steps {iters} --max_steps {iters} --do_train --train_file {script_dir}/download/squad/v1.1/train-v1.1.json > bert/log_bert_gpm_{cf}.txt"
        os.system(proc)

    print("Run for PCCheck")
    # run pccheck
    for cf in cfreqs:
        print(f"Checkpoint Frequency {cf}")
        proc = f"python3.9 {script_dir}/run_squad_pccheck.py --bert_model=bert-large-uncased --train_batch_size {batch_size} --output_dir output --vocab_file {script_dir}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt  --config_file {script_dir}/bert_configs/large.json --do_train --train_file {script_dir}/download/squad/v1.1/train-v1.1.json --cfreq {cf} --max_async 4 --num_threads 2 --bench_total_steps {iters} --c_lib_path {lib_path} > bert/log_bert_pccheck_{cf}.txt"
        os.system(proc)


def run(model):
    if model == "transformer":
        run_transformer()
    elif model == "bert":
        run_bert()
    elif model == "opt13":
        run_opt()
    else:
        raise NotImplementedError


def collect_model(model):
    def get_exec_throughput(input_file, baseline):
        exec_time  = 0.0
        extra_time = 0.0
        with open(input_file, 'r') as f:
            for line in f.readlines():
                if 'EXECUTION TIME' in line:
                    tokens = line.split(" ")
                    exec_time = float(tokens[-2])
                elif 'MMAP/UMAP' in line:
                    tokens = line.split(" ")
                    extra_time = float(tokens[-2])/1000 # convert in sec
        exec_time -= extra_time
        thr = (iters_dir[model]-WARMUP)/exec_time
        return thr

    throughput_dict = {}
    throughput_list = []

    for baseline in ["cfreq", "gpm", "pccheck"]:
        baseline_thr = []
        for cf in cfreqs:
            input_file = f"{model}/log_{model}_{baseline}_{cf}.txt"
            thr = get_exec_throughput(input_file, baseline)
            baseline_thr.append(thr)
        throughput_list.append(baseline_thr)
        throughput_dict[label_dict[baseline]] = baseline_thr

    print(throughput_list)
    column_header = [str(x) for x in cfreqs]
    index_header = ["CheckFreq", "GPM", "PCcheck"]
    df = pd.DataFrame(throughput_list, columns = column_header, index = index_header)
    df.to_csv(f'fig8_{model}.csv')
    return throughput_dict


def plot_model(model, data):

    colors = ['#4392B8', '#E27733','#A7B972']
    label_font_size = 36
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(cfreqs[1:]))
    bars = []

    print(data)
    for method_id, (method_key, method_data) in enumerate(data.items()):

        # slowdown = [baseline_throughput[0]/x for x in data[method]]
        # print(method, slowdown)
        bar = ax.bar(
            x + width * method_id, method_data[1:], width,
            label=method_key,
            align='edge',
            color=colors[method_id]
        )
        bars.append(bar)

    plt.yticks(fontsize=label_font_size)

    ax.plot(x+2*width, [data["PCcheck"][0]]*len(x), color='black', marker="s",linewidth=3,markersize=8)

    x_tick_positions = x + width * len(label_dict) / 2
    ax.set_xticks(
        ticks=x_tick_positions,
        labels=cfreqs[1:], fontsize=label_font_size,
    )
    plt.yticks(fontsize=label_font_size)

    ax.set_ylabel('Throughput (iterations/sec)', fontsize=label_font_size)
    ax.set_xlabel('Checkpoint interval(iterations)', fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()

    plt.legend(handles, labels, loc='upper left', ncol=4, fontsize=label_font_size-2, bbox_to_anchor=(-0.025, 1.2))
    plt.savefig(f"fig8_{model}.png", bbox_inches="tight", dpi=500, pad_inches=0.1)



if __name__ == "__main__":
    model = sys.argv[1]
    run(model)
    data = collect_model(model)
    plot_model(model, data)
