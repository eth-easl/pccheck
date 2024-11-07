import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

home_dir = os.path.expanduser("~")
lib_path = f"{home_dir}/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
script_dir = f"{home_dir}/pccheck/checkpoint_eval/models/microbenchmarks"
iters = 20

sizes_mb = [10,100,1000,10000]
baselines = ["CheckFreq", "GPM", "PCcheck"]

def run():
    for size in sizes_mb:
        run_cfreq = f"python3.9 {script_dir}/test_cfreq.py --size {size} --iterations 20 > cfreq_{size}.txt"
        os.system(run_cfreq)

        run_gpm = f"python3.9 {script_dir}/test_gpm.py --size {size} --iterations 20 > gpm_{size}.txt"
        os.system(run_gpm)

        run_pccheck = f"python3.9 {script_dir}/test_pccheck.py --size {size} --iterations 20 --num-threads 4 --c_lib_path {lib_path} > pccheck_{size}.txt"
        os.system(run_pccheck)


def collect():
    def get_time(baseline, size):
        micro_time = 0.0
        extra_time = 0.0 # != 0 only for GPM
        input_file = f"{baseline}_{size}.txt"
        with open(input_file, 'r') as f:
            for line in f.readlines():
                if 'AVERAGE Checkpoint' in line:
                    tokens = line.split(" ")
                    micro_time = float(tokens[-2])
                elif 'MMAP/UMAP' in line:
                    tokens = line.split(" ")
                    extra_time = float(tokens[-2])

        extra_time /= iters
        print(baseline, size, extra_time)
        micro_time -= extra_time
        return micro_time

    micro_times = []
    for baseline in ["cfreq", "gpm", "pccheck"]:
        micro_times_baseline = []
        for size in sizes_mb:
            time_baseline_size = get_time(baseline, size)
            micro_times_baseline.append(time_baseline_size)
        micro_times.append(micro_times_baseline)

    column_header = [str(sz) for sz in sizes_mb]
    index_header = baselines
    df = pd.DataFrame(micro_times, columns = column_header, index = index_header)
    df.to_csv('fig11.csv')
    return micro_times


def plot(data):

    label_font_size = 27
    colors = ['#4392B8', '#E27733','#A7B972']
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(sizes_mb))
    bars = []
    for i,(d,l) in enumerate(zip(data,baselines)):

        bar = ax.bar(
            x + width * i, d, width,
            label=l, color=colors[i], #yerr=method2err[method],
            align='edge'
        )
        bars.append(bar)

    # print([x/y for x,y in zip(data[0], data[2])])
    # print([x/y for x,y in zip(data[1], data[2])])
    #ax.set_ylim(0,5)

    x_tick_positions = x + width * len(baselines) / 2
    ax.set_xticks(
        ticks=x_tick_positions,
        labels=sizes_mb, fontsize=label_font_size
    )
    plt.yticks(fontsize=label_font_size)

    ax.set_yscale('log', base=10)
    ax.set_ylabel('Time to checkpoint (ms)', fontsize=label_font_size)
    ax.set_xlabel('Checkpoint Size (MB)', fontsize=label_font_size)

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper left', ncol=2, fontsize=label_font_size)
    plt.savefig(f"fig11.png", bbox_inches="tight")

if __name__ == "__main__":
    run()
    df = collect()
    plot(df)
