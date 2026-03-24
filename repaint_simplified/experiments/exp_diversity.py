import subprocess
from pathlib import Path


def run_command(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():

    # CONFIG
    image = "repaint_simplified/data/gt/inet_0000.png"
    mask = "repaint_simplified/data/masks/000010.png"

    seeds = [0, 1, 2]

    base_output = Path("outputs/exp_diversity")
    base_output.mkdir(parents=True, exist_ok=True)

    # LOOP
    for seed in seeds:

        print(f"\n=== Running seed {seed} ===")

        save_dir = base_output / f"sample_{seed}"

        RUN_SCRIPT = Path("repaint_simplified/run_experiment.py")

        cmd = [
            "python", str(RUN_SCRIPT),
            "--image", image,
            "--mask", mask,
            "--steps", "250",
            "--jump_length", "10",
            "--jump_n_sample", "10",
            "--seed", str(seed),
            "--save_dir", str(save_dir)
        ]

        run_command(cmd)

    print("\nDiversity experiment completed!")


if __name__ == "__main__":
    main()