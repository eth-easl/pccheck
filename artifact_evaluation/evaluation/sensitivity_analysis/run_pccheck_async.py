import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

home_dir = os.path.expanduser("~")
lib_path = f"{home_dir}/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
script_dir = f"{home_dir}/pccheck/checkpoint_eval/models/vision/"

iters = 200
model = "vgg16"
batchsize = 32
num_threads = 1

cfreqs = [0, 1, 5, 10, 15, 20, 25, 30]
max_async = [1, 2, 4, 6, 8]

# 1. Run
def run():
    for num_conc in max_async:
        for cf in cfreqs:
            print(f"Run with num_co {num_conc} cfreq {cf}")
            proc = f"python {script_dir}/train_pccheck.py --dataset imagenet  --batchsize {batchsize} --arch {model} --cfreq {cf} "\
                f"--bench_total_steps {iters} --max-async {num_conc} --num-threads {num_threads} --c_lib_path {lib_path} > log_{num_conc}_{cf}.txt"
            os.system(proc)


# 2. collect measurements
def collect():

    def get_time(num_conc, cf):
        thr  = 0.0
        input_file = f"log_{num_conc}_{cf}.txt"
        with open(input_file, 'r') as f:
            for line in f.readlines():
                if 'EXECUTION TIME' in line:
                    tokens = line.split(" ")
                    thr = float(tokens[-2])
                    break
        return thr

    slowdown_list = []
    for num_conc in max_async:
        thr_list_c = []
        for cf in cfreqs:
            thr_list_c.append(get_time(num_conc, cf))
        slowdown_list_c = [x/thr_list_c[0] for x in thr_list_c]
        slowdown_list.append(slowdown_list_c)

    column_header = [str(cf) for cf in cfreqs]
    index_header = [str(x) for x in max_async]
    df = pd.DataFrame(slowdown_list, columns = column_header, index = index_header)
    df.to_csv('fig12.csv')
    return slowdown_list


# 3. plot
def plot(data):
    width = 0.15
    label_font_size = 27
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(cfreqs[1:]))
    bars = []
    async_checkp = [f"{x} async checkp" for x in max_async]

    for i,(d,l) in enumerate(zip(data, async_checkp)):

        print(d)
        bar = ax.bar(
            x + width * i, d[1:], width,
            label=l, #yerr=method2err[method],
            align='edge'
        )
        bars.append(bar)

    x_tick_positions = x + width * (len(async_checkp)/2)
    ax.set_xticks(
        ticks=x_tick_positions,
        labels=cfreqs[1:], fontsize=label_font_size
    )
    ax.tick_params(axis='both', which='minor', labelsize=label_font_size)

    ax.set_yscale('log')
    ax.set_ylabel('Slowdown over \n no checkpointing', fontsize=label_font_size)
    ax.set_xlabel('Checkpoint frequency', fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right', ncol=1, fontsize=24)
    #plt.title(model, fontsize=label_font_size)
    plt.savefig("fig12.png", bbox_inches="tight")



if __name__ == "__main__":
    run()
    df = collect()
    plot(df)