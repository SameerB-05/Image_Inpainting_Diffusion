import subprocess
from pathlib import Path


def run_command(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():

    # CONFIG
    image = "repaint_simplified/data/gt/inet_0002.png"
    mask = "repaint_simplified/data/masks/000010.png"

    base_output = Path("outputs/exp_compute")
    base_output.mkdir(parents=True, exist_ok=True)

    experiments = {
        "repaint_250": {
            "steps": 250,
            "jump_length": 10,
            "jump_n_sample": 10
        },
        "diffusion_500": {
            "steps": 500,
            "jump_length": 1,   # no jumping
            "jump_n_sample": 1
        }
    }

    # LOOP
    for name, params in experiments.items():

        print(f"\n=== Running {name} ===")

        save_dir = base_output / name

        cmd = [
            "python", "run_experiment.py",
            "--image", image,
            "--mask", mask,
            "--steps", str(params["steps"]),
            "--jump_length", str(params["jump_length"]),
            "--jump_n_sample", str(params["jump_n_sample"]),
            "--seed", "0",
            "--save_dir", str(save_dir)
        ]

        run_command(cmd)

    print("\nCompute experiment completed!")


if __name__ == "__main__":
    main()