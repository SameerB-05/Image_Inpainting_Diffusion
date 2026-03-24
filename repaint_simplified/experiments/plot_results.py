import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def load_img(path):
    return Image.open(path)


def plot_single_result(exp_dir, save_path=None):
    """
    Plot:
    GT | Mask | Input | Output
    """

    exp_dir = Path(exp_dir)

    gt = load_img(exp_dir / "gt.png")
    mask = load_img(exp_dir / "mask.png")
    masked = load_img(exp_dir / "masked.png")
    output = load_img(exp_dir / "output.png")

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(gt)
    axs[0].set_title("GT")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Mask")
    axs[1].axis("off")

    axs[2].imshow(masked)
    axs[2].set_title("Input")
    axs[2].axis("off")

    axs[3].imshow(output)
    axs[3].set_title("Output")
    axs[3].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_resampling_comparison(base_dir, save_path=None):
    """
    Compare:
    GT | Mask | Input | Baseline | RePaint
    """

    base_dir = Path(base_dir)

    baseline = base_dir / "baseline"
    repaint = base_dir / "repaint"

    gt = load_img(baseline / "gt.png")
    mask = load_img(baseline / "mask.png")
    masked = load_img(baseline / "masked.png")

    out_base = load_img(baseline / "output.png")
    out_repaint = load_img(repaint / "output.png")

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    titles = ["GT", "Mask", "Input", "Baseline", "RePaint"]
    images = [gt, mask, masked, out_base, out_repaint]

    for i in range(5):
        axs[i].imshow(images[i], cmap="gray" if i == 1 else None)
        axs[i].set_title(titles[i])
        axs[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_diversity(base_dir, save_path=None):
    """
    Plot multiple outputs for diversity:
    GT | Mask | Input | Sample1 | Sample2 | Sample3
    """

    base_dir = Path(base_dir)

    sample_dirs = sorted(base_dir.glob("sample_*"))

    gt = load_img(sample_dirs[0] / "gt.png")
    mask = load_img(sample_dirs[0] / "mask.png")
    masked = load_img(sample_dirs[0] / "masked.png")

    outputs = [load_img(d / "output.png") for d in sample_dirs]

    fig, axs = plt.subplots(1, 3 + len(outputs), figsize=(5 * (3 + len(outputs)), 4))

    titles = ["GT", "Mask", "Input"] + [f"Sample {i}" for i in range(len(outputs))]
    images = [gt, mask, masked] + outputs

    for i in range(len(images)):
        axs[i].imshow(images[i], cmap="gray" if i == 1 else None)
        axs[i].set_title(titles[i])
        axs[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def main():

    # Example usages

    # 1. Single result
    plot_single_result("outputs/exp1")

    # 2. Resampling comparison
    plot_resampling_comparison("outputs/exp_resampling")

    # 3. Diversity visualization
    plot_diversity("outputs/exp_diversity")


if __name__ == "__main__":
    main()