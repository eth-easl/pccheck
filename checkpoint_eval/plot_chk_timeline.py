"""Plot the checkpoint persist timeline produced by train_pccheck_timing.py.

    python3 plot_chk_timeline.py /tmp/chk_timeline.csv -o /tmp/chk_timeline.png

Two panels sharing one time axis:

  * the trace itself -- one row per checkpoint, split into the time it spent
    queued waiting for a free slot and the time it spent persisting;
  * interval between durable checkpoints vs how long each persist took, both
    in seconds on one axis so they can be compared directly. When the two
    lines sit on top of each other the writer is saturated: a new checkpoint
    starts as soon as the previous one lands.
"""

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the reference palette, validated for all-pairs CVD
# separation on a light surface. Colour follows the quantity, not the panel.
C_QUEUE = "#2a78d6"    # slot 1, blue
C_PERSIST = "#eb6834"  # slot 2, orange
C_INTERVAL = "#1baf7a"  # slot 3, aqua -- below 3:1 on this surface, so it is
                        # always shipped with visible direct labels
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8a85"
SURFACE = "#fcfcfb"


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no rows")

    out = []
    for r in rows:
        if not r.get("t_persist"):
            out.append({"idx": int(r["idx"]), "step": r.get("step", ""),
                        "persisted": False})
            continue
        t_request = float(r["t_request"])
        queue_s = float(r["queue_ms"]) / 1000.0
        persist_s = float(r["persist_ms"]) / 1000.0
        out.append({
            "idx": int(r["idx"]),
            "step": r.get("step", ""),
            "persisted": True,
            "t_request": t_request,
            "t_start": t_request + queue_s,
            "t_persist": float(r["t_persist"]),
            "queue_s": queue_s,
            "persist_s": persist_s,
            "since_prev_s": float(r["since_prev_s"]) if r.get("since_prev_s") else None,
        })
    return out


def plot_interval(rows, out_path, title, tight_y=False):
    """Just the interval between durable checkpoints -- one series, one panel."""
    done = [r for r in rows if r["persisted"]]
    if not done:
        raise SystemExit("no persisted checkpoints to plot")

    t0 = min(r["t_request"] for r in done)
    pts = [(r["t_persist"] - t0, r["since_prev_s"]) for r in done
           if r["since_prev_s"] is not None]
    if not pts:
        raise SystemExit("need at least 2 persisted checkpoints for an interval")
    xs, ys = zip(*pts)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    med = sorted(ys)[len(ys) // 2]

    ax.plot(xs, ys, color=C_INTERVAL, linewidth=2.5, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)

    ax.set_xlabel("seconds since first checkpoint request", fontsize=14,
                  color=INK)
    ax.set_ylabel("seconds between\ndurable checkpoints", fontsize=14,
                  color=INK)
    ax.set_title(title, fontsize=17, color=INK, loc="left", pad=14)
    # Zero baseline by default: it shows the interval is pinned at a constant
    # rate, which is the finding. --tight-y zooms to the data range instead,
    # for when the run-to-run variation is what you want to show. (A line
    # encodes position, not length, so a non-zero baseline is legitimate here
    # in a way it would not be for bars.)
    lo, hi = min(ys), max(ys)
    if tight_y:
        pad = max((hi - lo) * 0.25, hi * 0.02)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        ax.set_ylim(0, hi * 1.10)
    # Grid stays light so it sits behind the data; axes and all text are black
    # so they survive a projector.
    ax.grid(axis="y", color="#dcdbd6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(colors=INK, labelsize=13, width=1.2, length=5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)

    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out_path}  ({len(pts)} intervals, median {med:.3f}s)")


# Reference categorical order, slots 1-8. Assigned in fixed order, never
# cycled -- a series keeps its colour regardless of how many others are drawn.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def plot_compare(datasets, out_path, title, tight_y=False):
    """Overlay the interval series from several runs.

    datasets: list of (label, rows). Each run is aligned to its own first
    checkpoint request, so the x-axis compares cadence, not wall-clock.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    all_y = []
    for i, (label, rows) in enumerate(datasets):
        done = [r for r in rows if r["persisted"]]
        t0 = min(r["t_request"] for r in done)
        pts = [(r["t_persist"] - t0, r["since_prev_s"]) for r in done
               if r["since_prev_s"] is not None]
        if not pts:
            print(f"  skipping {label}: fewer than 2 persisted checkpoints")
            continue
        xs, ys = zip(*pts)
        all_y.extend(ys)
        color = SERIES_COLORS[i % len(SERIES_COLORS)]

        ax.plot(xs, ys, color=color, linewidth=2.5, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                label=label)
        # direct label at the end of each line, so identity is not carried by
        # colour alone even before anyone reads the legend
        ax.annotate(f" {label}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, 0), va="center", ha="left", fontsize=13,
                    color=INK)

    if not all_y:
        raise SystemExit("nothing to plot")

    ax.set_xlabel("seconds since first checkpoint request", fontsize=14,
                  color=INK)
    ax.set_ylabel("seconds between\ndurable checkpoints", fontsize=14,
                  color=INK)
    ax.set_title(title, fontsize=17, color=INK, loc="left", pad=14)

    lo, hi = min(all_y), max(all_y)
    if tight_y:
        pad = max((hi - lo) * 0.2, hi * 0.02)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        ax.set_ylim(0, hi * 1.10)
    # room on the right for the direct labels
    ax.set_xlim(right=ax.get_xlim()[1] * 1.18)

    ax.grid(axis="y", color="#dcdbd6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(colors=INK, labelsize=13, width=1.2, length=5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=13,
                    labelcolor=INK, ncol=len(datasets))

    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out_path}")


def plot_events(rows, out_path, title):
    """A pure event trace: one tick per checkpoint, at the instant it became
    durable. No y-axis -- the only variable is when."""
    done = [r for r in rows if r["persisted"]]
    if not done:
        raise SystemExit("no persisted checkpoints to plot")

    t0 = min(r["t_request"] for r in done)
    xs = [r["t_persist"] - t0 for r in done]

    fig, ax = plt.subplots(figsize=(11, 2.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.eventplot([xs], colors=[C_INTERVAL], lineoffsets=[0], linelengths=[1.4],
                 linewidths=3, zorder=3)

    ax.set_xlabel("seconds since first checkpoint request", fontsize=14,
                  color=INK)
    ax.set_title(title, fontsize=17, color=INK, loc="left", pad=14)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.grid(axis="x", color="#dcdbd6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="x", colors=INK, labelsize=13, width=1.2, length=5)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK)

    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out_path}  ({len(xs)} checkpoint events)")


def plot(rows, out_path, title):
    done = [r for r in rows if r["persisted"]]
    if not done:
        raise SystemExit("no persisted checkpoints to plot")

    t0 = min(r["t_request"] for r in done)
    for r in done:
        r["x_request"] = r["t_request"] - t0
        r["x_start"] = r["t_start"] - t0
        r["x_persist"] = r["t_persist"] - t0

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.18},
    )
    fig.patch.set_facecolor(SURFACE)

    # ---- panel 1: the trace -------------------------------------------------
    ax.set_facecolor(SURFACE)
    for r in done:
        y = r["idx"]
        # 2px surface gap between the two segments keeps them readable when the
        # queue time is short
        ax.barh(y, r["queue_s"], left=r["x_request"], height=0.55,
                color=C_QUEUE, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        ax.barh(y, r["persist_s"], left=r["x_start"], height=0.55,
                color=C_PERSIST, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        ax.text(r["x_persist"] + 0.012, y, f"{r['persist_s'] * 1000:.0f} ms",
                va="center", ha="left", fontsize=8, color=INK_SECONDARY,
                zorder=4)

    missing = [r for r in rows if not r["persisted"]]
    for r in missing:
        ax.text(0.002, r["idx"], "requested, never persisted", va="center",
                ha="left", fontsize=8, color=INK_MUTED, style="italic")

    ax.set_yticks([r["idx"] for r in rows])
    ax.set_yticklabels([f"#{r['idx']}  step {r['step']}" for r in rows],
                       fontsize=8, color=INK_SECONDARY)
    ax.invert_yaxis()
    ax.set_ylabel("checkpoint", fontsize=9, color=INK_SECONDARY)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=28)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d3")
    ax.tick_params(colors=INK_MUTED, length=0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_QUEUE),
        plt.Rectangle((0, 0), 1, 1, color=C_PERSIST),
    ]
    # Above the axes, not inside: the last bar reaches the bottom-right corner,
    # so an in-plot legend lands on top of the data.
    ax.legend(handles, ["queued (waiting for a free slot)",
                        "persisting (write + msync + publish)"],
              loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2,
              frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    # ---- panel 2: interval vs persist duration ------------------------------
    ax2.set_facecolor(SURFACE)
    xs = [r["x_persist"] for r in done]
    persist = [r["persist_s"] for r in done]
    gap_x = [r["x_persist"] for r in done if r["since_prev_s"] is not None]
    gap_y = [r["since_prev_s"] for r in done if r["since_prev_s"] is not None]

    ax2.plot(xs, persist, color=C_PERSIST, linewidth=2, marker="o",
             markersize=5, label="persist duration", zorder=3)
    ax2.plot(gap_x, gap_y, color=C_INTERVAL, linewidth=2, marker="o",
             markersize=5, label="interval since previous persist", zorder=3)

    # Direct labels on the interval series: required relief for a colour below
    # 3:1 against the surface, and it is the number being asked about.
    for x, y in zip(gap_x, gap_y):
        ax2.annotate(f"{y:.2f}s", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=INK_SECONDARY)

    ax2.set_xlabel("seconds since first checkpoint request", fontsize=9,
                   color=INK_SECONDARY)
    ax2.set_ylabel("seconds", fontsize=9, color=INK_SECONDARY)
    ax2.set_ylim(bottom=0)
    ax2.grid(axis="y", color="#e6e5e1", linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax2.spines[s].set_color("#d9d8d3")
    ax2.tick_params(colors=INK_MUTED, labelsize=8)
    ax2.legend(loc="lower right", frameon=False, fontsize=8,
               labelcolor=INK_SECONDARY)

    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out_path}")


def _auto_labels(paths):
    """Derive series labels from the chk_timeline{max_async}_{threads} naming.

    Whichever knob is held constant is dropped from the labels and belongs in
    the title instead, so the legend names only what actually varies.
    """
    parsed = []
    for path in paths:
        m = re.search(r"chk_timeline(\d+)_(\d+)", os.path.basename(path))
        parsed.append((int(m.group(1)), int(m.group(2))) if m else None)

    if any(x is None for x in parsed):
        return [os.path.splitext(os.path.basename(p))[0] for p in paths], None

    same_async = len({n for n, _ in parsed}) == 1
    same_thr = len({t for _, t in parsed}) == 1

    if same_async and not same_thr:
        return ([f"{t} thread{'s' if t > 1 else ''}" for _, t in parsed],
                f"max_async={parsed[0][0]}")
    if same_thr and not same_async:
        return ([f"max_async={n}" for n, _ in parsed],
                f"{parsed[0][1]} thread{'s' if parsed[0][1] > 1 else ''}")
    return ([f"max_async={n}, {t} thr" for n, t in parsed], None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+",
                    help="merged timeline(s) from train_pccheck_timing.py. "
                         "Give several to overlay their interval series.")
    ap.add_argument("-o", "--out", default=None, help="output PNG")
    ap.add_argument("--title", default=None)
    ap.add_argument("--labels", default=None,
                    help="comma-separated series labels, overriding the ones "
                         "derived from the filenames")
    ap.add_argument("--tight-y", dest="tight_y", action="store_true",
                    help="zoom the y-axis to the data range instead of "
                         "starting at zero")
    ap.add_argument("--events", action="store_true",
                    help="event strip: one tick per checkpoint, no y-axis")
    ap.add_argument("--full", action="store_true",
                    help="also draw the per-checkpoint trace panel "
                         "(default: interval line only)")
    a = ap.parse_args()

    if len(a.csv) > 1:
        if a.events or a.full:
            raise SystemExit("--events/--full take a single CSV")
        labels, held = (a.labels.split(","), None) if a.labels else _auto_labels(a.csv)
        datasets = [(lbl.strip(), load(path)) for lbl, path in zip(labels, a.csv)]
        out = a.out or "chk_timeline_compare.png"
        title = a.title or ("Interval between durable checkpoints"
                            + (f" — {held}" if held else ""))
        plot_compare(datasets, out, title, tight_y=a.tight_y)
        return

    path = a.csv[0]
    rows = load(path)
    out = a.out or os.path.splitext(path)[0] + ".png"
    done = [r for r in rows if r["persisted"]]
    span = max(r["t_persist"] for r in done) - min(r["t_request"] for r in done)

    if a.events:
        title = a.title or (
            f"When checkpoints became durable — {len(done)} checkpoints "
            f"over {span:.1f}s"
        )
        plot_events(rows, out, title)
    elif a.full:
        title = a.title or (
            f"pccheck checkpoint persist timeline — {len(done)} checkpoints "
            f"over {span:.1f}s"
        )
        plot(rows, out, title)
    else:
        title = a.title or (
            f"Interval between durable checkpoints — {len(done)} checkpoints "
            f"over {span:.1f}s"
        )
        plot_interval(rows, out, title, tight_y=a.tight_y)


if __name__ == "__main__":
    main()
