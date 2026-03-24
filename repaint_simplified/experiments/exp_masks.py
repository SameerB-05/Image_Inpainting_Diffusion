import subprocess
from pathlib import Path


def run_command(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():

    # CONFIG
    images = [
        "repaint_simplified/data/gt/inet_0001.png",
        "repaint_simplified/data/gt/inet_0002.png",
        "repaint_simplified/data/gt/inet_0003.png",
    ]

    masks = {
        "narrow": "repaint_simplified/data/masks/000001.png",
        "wide":   "repaint_simplified/data/masks/000010.png",
        "thin":   "repaint_simplified/data/masks/000020.png",
        "thick":  "repaint_simplified/data/masks/000030.png",
    }

    base_output = Path("outputs/exp_masks")
    base_output.mkdir(parents=True, exist_ok=True)

    # LOOP
    for img_path in images:

        img_name = Path(img_path).stem
        print(f"\n=== Processing Image: {img_name} ===")

        for mask_name, mask_path in masks.items():

            save_dir = base_output / img_name / mask_name

            cmd = [
                "python", "run_experiment.py",
                "--image", img_path,
                "--mask", mask_path,
                "--steps", "250",
                "--jump_length", "10",
                "--jump_n_sample", "10",
                "--seed", "0",
                "--save_dir", str(save_dir)
            ]

            run_command(cmd)

    print("\nMask experiment completed!")


if __name__ == "__main__":
    main()