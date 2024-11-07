import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

home_dir = os.path.expanduser("~")
lib_path = f"{home_dir}/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"

cfreqs = [0, 1, 10, 25, 50, 100]

model_scripts_dir = {
    "transformer": f"{home_dir}/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch",
    "bert": f"{home_dir}/DeepLearningExamples/PyTorch/LanguageModeling/BERT",
    "opt13": f"{home_dir}/transformers/examples/pytorch/language-modeling",
}

batch_size_dir = {"opt13": 1, "transformer": 64, "bert": 3}
iters_dir = {"opt13": 300, "transformer": 300, "bert": 300}
label_dict = {"cfreq": "CheckFreq", "gpm": "GPM", "pccheck": "PCcheck"}

N_pccheck = {
    "transformer": 4,
    "opt13": 2,
    "bert": 4,
    "opt_27": 2,
    "bloom_7": 2,
}

colors = {
    "CheckFreq": '#4392B8',
    "GPM" : '#E27733',
    "PCcheck": '#A7B972'
}

markers = {
    "CheckFreq": '*',
    "GPM" : 's',
    "PCcheck": 'o'
}

def get_load_time(model):

    checkpoint_file = ""


    def run_transformer():
        script_dir = model_scripts_dir[model]
        checkpoint_file = f"{script_dir}/checkpoint-0-0.chk"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        proc = f"cd {script_dir} && python3.9 train_checkfreq.py --config_file wt103_base.yaml --batch_size 64 --cfreq 200 --bench_total_steps 200"
        os.system(proc)
        return checkpoint_file


    def run_bert():
        script_dir = model_scripts_dir[model]
        checkpoint_file = f"checkpoint-0-0.chk"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        proc = f"python3.9 {script_dir}/run_squad_chfreq.py --bert_model=bert-large-uncased --train_batch_size 3 --output_dir output --vocab_file {script_dir}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt  --config_file {script_dir}/bert_configs/large.json --cfreq 200 --bench_total_steps 200 --max_steps 200 --do_train --train_file {script_dir}/download/squad/v1.1/train-v1.1.json"
        os.system(proc)
        return checkpoint_file


    def run_opt():
        script_dir = model_scripts_dir[model]
        checkpoint_file = f"checkpoint-0-0.chk"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        os.system("rm -rf output")
        proc = f"python3.9 {script_dir}/run_clm_cfreq.py --model_name_or_path facebook/opt-1.3b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --per_device_train_batch_size 1 --cfreq 200 --bench_total_steps 200"
        os.system(proc)
        return checkpoint_file


    if model == "transformer":
        checkpoint_file = run_transformer()
    elif model == "bert":
        checkpoint_file = run_bert()
    elif model == "opt13":
        checkpoint_file = run_opt()
    else:
        raise NotImplementedError

    # 2. Load and get time

    if model=="transformer":
        script_dir = model_scripts_dir[model]
        this_dir = f"{home_dir}/pccheck/artifact_evaluation/evaluation/throughput"
        os.system(f"cp loading.py {script_dir}/")
        os.system(f"cd {script_dir} && python3.9 loading.py checkpoint-0-0.chk > {this_dir}/loading_log_{model}.txt")
    else:
        os.system(f"python3.9 loading.py {checkpoint_file} > loading_log_{model}.txt")

    # 3. Read file and return
    with open(f"loading_log_{model}.txt", 'r') as f:
        for line in f.readlines():
            if "Time is" in line:
                tokens = line.split(" ")
                load_time = float(tokens[-2])

    return load_time


def get_fails_iters(trace_file):
    data = list(trace_file["Time"].iloc[[0, -1]])
    time_sec = data[1] - data[0]

    num_fails = 0
    prev_cores = 0
    for it, row in trace_file.iterrows():
        if it >= 0:
            if row["GPUs"] != prev_cores:
                num_fails += abs(row["GPUs"] - prev_cores) // 4
        prev_cores = row["GPUs"]
    return num_fails, time_sec


def get_time_redo(baseline, cfreq, time_no_checkp, loading_time, model, Tw_pccheck):
    if baseline in ["CheckFreq", "Gemini"]:
        time_redo = cfreq * time_no_checkp + loading_time
    elif baseline == "GPM":
        time_redo = cfreq * time_no_checkp / 2 + loading_time
    elif baseline == "PCcheck":
        time_redo = loading_time + cfreq * time_no_checkp / 2
        time_redo += (
            time_no_checkp
            * min(Tw_pccheck / time_no_checkp, cfreq * N_pccheck[model])
            / 2
        )
    elif baseline == "Ideal":
        time_redo = cfreq * time_no_checkp / 2 + loading_time

    return time_redo


def get_goodput_model_baseline(
    baseline,
    num_fails,
    cfreq,
    total_time,
    avg_iter_time_checkp,
    loading_time,
    time_no_checkp,
    model,
    Tw_pccheck
):

    time_redo = get_time_redo(baseline, cfreq, time_no_checkp, loading_time, model, Tw_pccheck)

    time_redo_all = time_redo * num_fails
    # time_redo_all = 0
    time_rem = total_time - time_redo_all
    seen_batches = time_rem / avg_iter_time_checkp
    throughput = seen_batches / total_time
    # return seen_batches
    return max(0, throughput)


def get_goodput_model(model):

    load_time = get_load_time(model)
    print(load_time)
    num_fails, time_sec = get_fails_iters(pd.read_csv("gpus_trace.csv"))
    print(num_fails, time_sec)

    iter_times = {}
    iter_times_df = pd.read_csv(f"fig8_{model}.csv", header=0, index_col=0)
    baseline_list = ["CheckFreq", "GPM", "PCcheck"]

    for i, baseline in enumerate(baseline_list):
        throughput = list(iter_times_df.iloc[i])
        iter_times[baseline] = [1 / x for x in throughput]

    Tw_pccheck_model = [0]
    for cf in cfreqs[1:]:
        input_file = f"{model}/log_{model}_pccheck_{cf}.txt"
        with open(input_file, 'r') as f:
            for line in f.readlines():
                if "average is" in line:
                    tokens = line.split(" ")
                    average_time = float(tokens[-1]) # use the last one
        Tw_pccheck_model.append(average_time)

    goodputs_dict = {}
    goodputs_list = []
    baseline_list_with_ideal = baseline_list+["Ideal"]

    for baseline in baseline_list_with_ideal:
        goodput_baseline = []
        for i, cf in enumerate(cfreqs):
            goodput_cf = get_goodput_model_baseline(
                baseline,
                num_fails,
                cf,
                time_sec,
                iter_times["PCcheck"][0] if baseline=="Ideal" else iter_times[baseline][i],
                load_time,
                iter_times["PCcheck"][0] if baseline=="Ideal" else iter_times[baseline][0],
                model,
                Tw_pccheck_model[i]
            )
            goodput_baseline.append(goodput_cf)
        goodputs_list.append(goodput_baseline)
        goodputs_dict[baseline] = goodput_baseline

    column_header = [str(x) for x in cfreqs]
    index_header = ["CheckFreq", "GPM", "PCcheck", "Ideal"]
    df = pd.DataFrame(goodputs_list, columns = column_header, index = index_header)
    df.to_csv(f'fig9_{model}.csv')
    return goodputs_dict


def plot_model(model, data):
    x = range(len(cfreqs[1:]))
    label_font_size = 36
    fig, ax = plt.subplots(figsize=(14, 7))

    for method_id, (method_key, method_data) in enumerate(data.items()):
        print(method_key, method_data)
        if method_key=="Ideal":
            continue
        plt.plot(x, method_data[1:], label=method_key, linewidth=3,
             marker=markers[method_key], markersize=10, color=colors[method_key])

    plt.plot(x, data["Ideal"][1:], label='Ideal',
             linewidth=3, linestyle='--', color='grey')

    ax.set_ylabel('Goodput (batches/sec)', fontsize=label_font_size)
    ax.set_xlabel('Checkpoint interval(iterations)', fontsize=label_font_size)

    plt.yticks(fontsize=label_font_size)
    plt.xticks(x, cfreqs[1:], fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right', ncol=1, fontsize=33)
    plt.savefig(f"fig9_{model}.png", bbox_inches="tight", dpi=500, pad_inches=0.1)



if __name__ == "__main__":
    model = sys.argv[1]
    data = get_goodput_model(model)
    print(data)
    plot_model(model, data)
