import matplotlib.pyplot as plt
from repaint_sampler import get_schedule
from pathlib import Path


import matplotlib.pyplot as plt
from repaint_sampler import get_schedule
from pathlib import Path


import matplotlib.pyplot as plt
from repaint_sampler import get_schedule
from pathlib import Path


def plot_schedule(t_T=250, jump_length=10, jump_n_sample=10):

    ts = get_schedule(t_T, jump_length, jump_n_sample)
    total_len = len(ts)

    first_n = 500
    last_n = 250

    ts_start = ts[:first_n]
    ts_end = ts[-last_n:]

    steps_start = list(range(len(ts_start)))
    steps_end = list(range(total_len - last_n, total_len))

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(10, 4),
        sharey=True,
        gridspec_kw={'wspace': 0.03, 'width_ratios': [2, 1]}
    )

    # Left
    ax1.plot(steps_start, ts_start)
    ax1.set_title("Start (500)")
    ax1.set_xlabel("Iter")
    ax1.set_ylabel("t")
    ax1.grid()

    # Right
    ax2.plot(steps_end, ts_end)
    ax2.set_title("End (250)")
    ax2.set_xlabel("Iter")
    ax2.grid()

    # Break marks
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)

    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)

    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)

    # Single title
    fig.suptitle(
        f"RePaint Schedule (T={t_T}, jl={jump_length}, jn={jump_n_sample}, N={total_len})",
        fontsize=12
    )

    plt.tight_layout()
    
    save_dir = Path("outputs/schedule_plots")
    save_dir.mkdir(exist_ok=True)

    plt.savefig(
        f"{save_dir}/{t_T}_{jump_length}_{jump_n_sample}_broken.png",
        bbox_inches='tight',
        dpi=150
    )

    plt.show()

    print(f"Total iterations: {total_len}")

def main():


    jump_lengths = [1, 5, 10]
    jump_ns = [5, 10]

    for jl in jump_lengths:
        for jn in jump_ns:
            plot_schedule(
                t_T=250,
                jump_length=jl,
                jump_n_sample=jn
            )
    """
    # Change these to experiment
    plot_schedule(
        t_T=250,
        jump_length=10,
        jump_n_sample=10
    )"""


if __name__ == "__main__":
    main()