import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

home_dir = os.path.expanduser("~")
lib_path = f"{home_dir}/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
script_dir = f"{home_dir}/transformers/examples/pytorch/language-modeling/run_clm_pccheck.py"

iters = 100
model = "facebook/opt-350m"
batchsize = 2

cfreq = 10
max_async = [1,2,3]
num_threads = list(range(1,3))

# 1. Run
def run():

    # run 0 first
    os.system("rm -rf output")
    proc = f"python3.9 {script_dir} --model_name_or_path {model} --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train "\
        f"--max_async 1 --num_threads 1  --per_device_train_batch_size {batchsize} --cfreq 0 --bench_total_steps {iters} --c_lib_path {lib_path}"
    os.system(proc)

    for num_conc in max_async:
        for thread_count in num_threads:
            os.system("rm -rf output")
            print(f"Run with num_co {num_conc}, num_threads {thread_count}")
            proc = f"python3.9 {script_dir} --model_name_or_path {model} --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train "\
                f"--max_async {num_conc} --num_threads {thread_count} --psize 5 --per_device_train_batch_size {batchsize} --cfreq {cfreq} --bench_total_steps {iters} --c_lib_path {lib_path}"
            os.system(proc)


# 2. collect measurements
def collect():

    def get_time_0():
        with open('log_0.txt', 'r') as f:
            for line in f.readlines():
                if 'EXECUTION TIME' in line:
                    tokens = line.split(" ")
                    thr = float(tokens[-2])
                    break
        return thr

    def get_time(num_conc, thread_count):
        thr  = 0.0
        input_file = f"log_{num_conc}_{thread_count}.txt"
        with open(input_file, 'r') as f:
            for line in f.readlines():
                if 'EXECUTION TIME' in line:
                    tokens = line.split(" ")
                    thr = float(tokens[-2])
                    break
        return thr

    time_0 = get_time_0()
    slowdown_list = []
    for num_conc in max_async:
        thr_list_c = []
        for thread_count in num_threads:
            thr_list_c.append(get_time(num_conc, thread_count))
        slowdown_list_c = [x/time_0 for x in thr_list_c]
        slowdown_list.append(slowdown_list_c)

    print(slowdown_list)

    column_header = [str(x) for x in num_threads]
    index_header = [str(x) for x in max_async]
    df = pd.DataFrame(slowdown_list, columns = column_header, index = index_header)
    df.to_csv('fig13.csv')
    return slowdown_list


# 3. plot
def plot(data):
    width = 0.2
    label_font_size = 27

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(num_threads))
    bars = []
    async_checkp = [f"{x} async checkp" for x in max_async]
    for i,(d,l) in enumerate(zip(data,async_checkp)):

        bar = ax.bar(
            x + width * i, d, width,
            label=l, #yerr=method2err[method],
            align='edge'
        )
        bars.append(bar)

    plt.yticks(fontsize=label_font_size)
    #ax.set_ylim(0,5)

    x_tick_positions = x + width * len(max_async) / 2
    ax.set_xticks(
        ticks=x_tick_positions,
        labels=num_threads, fontsize=label_font_size
    )
    plt.yticks(fontsize=label_font_size)

    #ax.set_yscale('log')
    ax.set_ylabel('Slowdown over \n no checkpointing', fontsize=label_font_size)
    ax.set_xlabel('Number of threads per checkpoint', fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right', ncol=3, fontsize=20)
    #plt.title(model, fontsize=label_font_size)
    plt.savefig(f"fig13.pdf", bbox_inches="tight")


if __name__ == "__main__":
    run()
    df = collect()
    plot(df)