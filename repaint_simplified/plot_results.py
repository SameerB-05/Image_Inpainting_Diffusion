import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import math


# Global styling (clean look)
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
})



# Utils
def load_img(path):
    return Image.open(path)


def show_grid(images, titles, n_cols=3, figsize=(12, 8), save_path=None):
    """
    Generic grid plotter for clean visualization
    """
    n_images = len(images)
    n_rows = math.ceil(n_images / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Always flatten safely
    if isinstance(axs, (list, tuple)):
        axs = list(axs)
    else:
        axs = axs.flatten()

    for i in range(len(axs)):
        if i < n_images:
            axs[i].imshow(images[i], cmap="gray" if titles[i] == "Mask" else None)
            axs[i].set_title(titles[i])
            axs[i].axis("off")
        else:
            axs[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



# 1. Single Experiment
def plot_single_result(exp_dir, save_path=None):
    """
    Layout (2x2):
    GT | Mask
    Masked | Output
    """
    exp_dir = Path(exp_dir)

    images = [
        load_img(exp_dir / "gt.png"),
        load_img(exp_dir / "mask.png"),
        load_img(exp_dir / "masked.png"),
        load_img(exp_dir / "output.png"),
    ]

    titles = ["GT", "Mask", "Masked Image", "Output"]

    show_grid(images, titles, n_cols=2, figsize=(8, 8), save_path=save_path)



# 2. Baseline vs RePaint
def plot_resampling_comparison(base_dir, save_path=None):
    """
    Layout:
    GT | Mask | Masked
    Baseline | RePaint
    """
    base_dir = Path(base_dir)

    baseline = base_dir / "baseline"
    repaint = base_dir / "repaint"

    images = [
        load_img(baseline / "gt.png"),
        load_img(baseline / "mask.png"),
        load_img(baseline / "masked.png"),
        load_img(baseline / "output.png"),
        load_img(repaint / "output.png"),
    ]

    titles = ["GT", "Mask", "Masked", "Baseline", "RePaint"]

    show_grid(images, titles, n_cols=3, figsize=(12, 6), save_path=save_path)



# 3. Diversity Visualization
def plot_diversity(base_dir, save_path=None):
    """
    Layout:
    GT | Mask | Masked | Sample1
    Sample2 | Sample3 | ...
    """
    base_dir = Path(base_dir)

    sample_dirs = sorted(base_dir.glob("sample_*"))

    if len(sample_dirs) == 0:
        print("No samples found!")
        return

    gt = load_img(sample_dirs[0] / "gt.png")
    mask = load_img(sample_dirs[0] / "mask.png")
    masked = load_img(sample_dirs[0] / "masked.png")

    outputs = [load_img(d / "output.png") for d in sample_dirs]

    images = [gt, mask, masked] + outputs
    titles = ["GT", "Mask", "Masked"] + [f"Sample {i+1}" for i in range(len(outputs))]

    show_grid(images, titles, n_cols=4, figsize=(16, 8), save_path=save_path)


def plot_jump_ablation(base_dir, save_path=None):
    """
    Visualize jump_length and jump_n_sample variations
    """

    base_dir = Path(base_dir)
    exp_dirs = sorted(base_dir.glob("jl_*"))

    if len(exp_dirs) == 0:
        print("No experiments found!")
        return

    # Load common inputs from first experiment
    gt = load_img(exp_dirs[0] / "gt.png")
    mask = load_img(exp_dirs[0] / "mask.png")
    masked = load_img(exp_dirs[0] / "masked.png")

    images = [gt, mask, masked]
    titles = ["GT", "Mask", "Masked"]

    # Load all experiment outputs
    for d in exp_dirs:
        output = load_img(d / "output.png")

        # Extract params from folder name
        name = d.name  # jl_5_jn_10
        parts = name.split("_")
        jl = parts[1]
        jn = parts[3]

        images.append(output)
        titles.append(f"jl={jl}, jn={jn}")

    show_grid(images, titles, n_cols=3, figsize=(12, 8), save_path=save_path)


def plot_mask_experiment(base_dir, save_path=None):
    """
    Visualize different mask types for each image
    """

    base_dir = Path(base_dir)
    image_dirs = sorted(base_dir.glob("*"))

    for img_dir in image_dirs:

        mask_dirs = sorted(img_dir.glob("*"))

        if len(mask_dirs) == 0:
            continue

        print(f"Plotting: {img_dir.name}")

        images = []
        titles = []

        for mdir in mask_dirs:
            gt = load_img(mdir / "gt.png")
            mask = load_img(mdir / "mask.png")
            masked = load_img(mdir / "masked.png")
            output = load_img(mdir / "output.png")

            # Add full row per mask
            images.extend([gt, mask, masked, output])
            titles.extend([
                f"{mdir.name} - GT",
                f"{mdir.name} - Mask",
                f"{mdir.name} - Masked",
                f"{mdir.name} - Output",
            ])

        # 4 columns per mask row
        show_grid(
            images,
            titles,
            n_cols=4,
            figsize=(16, 4 * len(mask_dirs)),
            save_path=f"{save_path}_{img_dir.name}.png" if save_path else None
        )

# Main
def main():

    save_path = Path("outputs/result_plots")
    Path(save_path).mkdir(exist_ok=True)

    # Uncomment what you want

    """plot_resampling_comparison(
        "outputs/exp_resampling",
        save_path = save_path / "resampling_comparison.png"
    )"""

    plot_diversity(
        "outputs/exp_diversity",
        save_path = save_path / "diversity.png"
    )

    """plot_jump_ablation(
        "outputs/exp_jumps",
        save_path = save_path / "jump_ablation.png"
    )"""
    
    """
    plot_mask_experiment(
        "outputs/exp_masks",
        save_path = save_path / "mask_exp.png"
    )"""

if __name__ == "__main__":
    main()