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
    Visualize mask variation per GT image
    Layout:
        Row 0: GT | wide mask | wide masked | wide output
        Row 1:    | thin mask | thin masked | thin output
        Row 2:    | thick mask| thick masked| thick output
    """

    base_dir = Path(base_dir)
    image_dirs = sorted(base_dir.glob("*"))

    mask_order = ["wide", "thin", "thick"]

    for img_dir in image_dirs:

        if not img_dir.is_dir():
            continue

        print(f"Plotting: {img_dir.name}")

        rows = []

        gt_img = None

        for i, mask_name in enumerate(mask_order):
            mdir = img_dir / mask_name

            if not mdir.exists():
                continue

            try:
                gt = load_img(mdir / "gt.png")
                mask = load_img(mdir / "mask.png")
                masked = load_img(mdir / "masked.png")
                output = load_img(mdir / "output.png")
            except Exception as e:
                print(f"Skipping {mdir}: {e}")
                continue

            if gt_img is None:
                gt_img = gt  # store once

            if i == 0:
                # First row includes GT
                rows.append([
                    gt, mask, masked, output
                ])
                titles = ["GT", f"{mask_name} Mask", "Masked", "Output"]
            else:
                # Remaining rows: empty first column
                rows.append([
                    None, mask, masked, output
                ])

        if len(rows) == 0:
            continue

        n_rows = len(rows)
        n_cols = 4

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))

        if n_rows == 1:
            axs = [axs]

        for r in range(n_rows):
            for c in range(n_cols):
                ax = axs[r][c]

                img = rows[r][c]

                if img is not None:
                    ax.imshow(img, cmap="gray" if c == 1 else None)

                # Remove ticks (but keep labels)
                ax.set_xticks([])
                ax.set_yticks([])

                # Remove borders
                for spine in ax.spines.values():
                    spine.set_visible(False)

                # Column titles
                if r == 0:
                    if c == 0:
                        ax.set_title("GT")
                    elif c == 1:
                        ax.set_title("Mask")
                    elif c == 2:
                        ax.set_title("Masked")
                    elif c == 3:
                        ax.set_title("Output")

                # Mask label below mask image
                if c == 1:
                    label = mask_order[r]
                    ax.set_xlabel(label.capitalize(), fontsize=10, labelpad=6)

        final_save_path = None
        if save_path:
            save_path = Path(save_path)
            final_save_path = save_path.parent / f"{save_path.stem}_{img_dir.name}.png"
            plt.savefig(final_save_path, dpi=300, bbox_inches="tight")

        plt.show()

# Main
def main():

    save_path = Path("outputs/result_plots")
    Path(save_path).mkdir(exist_ok=True)

    # Uncomment what you want

    """plot_resampling_comparison(
        "outputs/exp_resampling",
        save_path = save_path / "resampling_comparison.png"
    )"""

    """plot_diversity(
        "outputs/exp_diversity",
        save_path = save_path / "diversity.png"
    )"""

    """plot_jump_ablation(
        "outputs/exp_jumps",
        save_path = save_path / "jump_ablation.png"
    )"""
    
    plot_mask_experiment(
        "outputs/exp_masks",
        save_path = save_path / "mask_exp.png"
    )

if __name__ == "__main__":
    main()