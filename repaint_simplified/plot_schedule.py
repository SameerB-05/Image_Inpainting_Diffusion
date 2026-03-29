import matplotlib.pyplot as plt
from repaint_sampler import get_schedule
from pathlib import Path


def plot_schedule(t_T=250, jump_length=10, jump_n_sample=10):

    ts = get_schedule(t_T, jump_length, jump_n_sample)

    steps = list(range(len(ts)))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, ts)

    plt.xlabel("Iteration")
    plt.ylabel("Timestep (t)")
    plt.title(f"RePaint Schedule (T={t_T}, jl={jump_length}, jn={jump_n_sample})")

    plt.grid()

    save_dir = Path("outputs/schedule_plots")
    save_dir.mkdir(exist_ok=True)

    plt.savefig(f"{save_dir}/{t_T}_{jump_length}_{jump_n_sample}.png")
    plt.show()

    print(f"Total iterations: {len(ts)}")


def main():

    # Change these to experiment
    plot_schedule(
        t_T=250,
        jump_length=10,
        jump_n_sample=10
    )


if __name__ == "__main__":
    main()