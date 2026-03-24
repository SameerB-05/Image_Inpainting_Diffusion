import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():

    # CONFIG
    image = "repaint_simplified/data/gt/inet_0000.png"
    mask = "repaint_simplified/data/masks/000010.png"

    base_output = Path("outputs/exp_resampling")
    base_output.mkdir(parents=True, exist_ok=True)

    experiments = {
        "baseline": {
            "jump_n_sample": 1   # no resampling
        },
        "repaint": {
            "jump_n_sample": 10  # with resampling
        }
    }

    # LOOP
    for name, params in experiments.items():

        print(f"\n=== Running {name.upper()} ===")

        save_dir = base_output / name

        RUN_SCRIPT = Path("repaint_simplified/run_experiment.py")

        cmd = [
            "python", str(RUN_SCRIPT),
            "--image", image,
            "--mask", mask,
            "--steps", "250",
            "--jump_length", "10",
            "--jump_n_sample", str(params["jump_n_sample"]),
            "--seed", "0",
            "--save_dir", str(save_dir)
        ]

        run_command(cmd)

    print("\nResampling experiment completed!")


if __name__ == "__main__":
    main()