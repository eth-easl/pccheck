import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

home_dir = os.path.expanduser("~")
lib_path = f"{home_dir}/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
script_dir = f"{home_dir}/pccheck/checkpoint_eval/models/opt/"

max_async = 2
num_threads = 2
cfreq = 15
iters = 200

dram_ratios = [1.0, 1.25, 1.5, 1.75, 2.0]
psizes = [1, 8, 16]

def run():
    for ratio in dram_ratios:
        for psize in psizes:
            os.system("rm -rf output")
            cmd = f"python3.9 {script_dir}/run_clm_pccheck.py --model_name_or_path facebook/opt-1.3b --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --max_async {max_async} --num_threads {num_threads} "\
                f"--psize {psize} --dram_ratio {ratio} --per_device_train_batch_size 1 --cfreq {cfreq} --bench_total_steps {iters} --c_lib_path {lib_path} > log_{ratio}_{psize}.txt"
            os.system(cmd)


def collect():
    def get_time(ratio, psize):
        exec_time  = 0.0
        input_file = f"log_{ratio}_{psize}.txt"
        with open(input_file, 'r') as f:
            for line in f.readlines():
                if 'EXECUTION TIME' in line:
                    tokens = line.split(" ")
                    exec_time = float(tokens[-2])
                    break
        return exec_time

    throughput_list = []
    for psize in psizes:
        thr_list_c = []
        for ratio in dram_ratios:
            exec_time = get_time(ratio, psize)
            thr_list_c.append(iters/exec_time)
        throughput_list.append(thr_list_c)

    column_header = [str(ratio) for ratio in dram_ratios]
    index_header = [str(psize) for psize in psizes]
    df = pd.DataFrame(throughput_list, columns = column_header, index = index_header)
    df.to_csv('fig14.csv')
    return throughput_list


def plot(data):
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(data[1]))
    xlabels = ['m','1.25*m','1.5*m', '1.75*m', '2*m']
    label_font_size = 23

    plt.plot(x, data[0], label='p1', linewidth=2, marker='o', color='dodgerblue')
    plt.plot(x, data[1], label='p8', linewidth=2, marker='^', color='forestgreen')
    plt.plot(x, data[2], label='p16', linewidth=2, marker="v", color='darkviolet')

    ax.set_ylabel('Throughput \n (iterations/sec)', fontsize=label_font_size)
    ax.set_xlabel('Amount of DRAM PCcheck uses', fontsize=label_font_size)

    plt.yticks(fontsize=label_font_size)
    plt.xticks(x,xlabels,fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper left', ncol=3, fontsize=label_font_size)
    plt.savefig(f"fig14.png", bbox_inches="tight", dpi=500, pad_inches=0.1)


if __name__ == "__main__":
    #run()
    df = collect()
    plot(df)