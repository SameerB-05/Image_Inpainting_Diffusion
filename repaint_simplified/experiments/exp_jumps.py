import subprocess
from pathlib import Path


def run_command(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():

    # CONFIG
    image = "repaint_simplified/data/gt/inet_0003.png"
    mask = "repaint_simplified/data/masks/000010.png"

    base_output = Path("outputs/exp_jumps")
    base_output.mkdir(parents=True, exist_ok=True)

    jump_lengths = [1, 5, 10]
    jump_ns = [5, 10]

    # LOOP
    for jl in jump_lengths:
        for jn in jump_ns:

            exp_name = f"jl_{jl}_jn_{jn}"
            print(f"\n=== Running {exp_name} ===")

            save_dir = base_output / exp_name

            cmd = [
                "python", "run_experiment.py",
                "--image", image,
                "--mask", mask,
                "--steps", "250",
                "--jump_length", str(jl),
                "--jump_n_sample", str(jn),
                "--seed", "0",
                "--save_dir", str(save_dir)
            ]

            run_command(cmd)

    print("\nJump ablation experiment completed!")


if __name__ == "__main__":
    main()