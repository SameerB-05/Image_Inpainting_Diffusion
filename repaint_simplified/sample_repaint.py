import torch
import random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from openai_guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults
)
from mask import load_mask
from repaint_sampler import repaint_sample


def load_image(path, image_size):

    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))

    img = np.array(img) / 255.0
    img = img * 2 - 1

    img = torch.from_numpy(img).permute(2, 0, 1)[None]

    return img.float()


def tensor_to_numpy(x):

    x = (x.clamp(-1, 1) + 1) / 2
    x = x[0].permute(1, 2, 0).cpu().numpy()
    return x


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = 256

    print("Using device:", device)

 
    # Load model
    defaults = model_and_diffusion_defaults()

    defaults.update(dict(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        channel_mult="1,1,2,2,4,4",
        resblock_updown=True,
        class_cond=True,
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="250"
    ))

    model, diffusion = create_model_and_diffusion(**defaults)


    weight_path = Path("repaint_simplified/pretrained_weights/256x256_diffusion.pt")

    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print("Loaded pretrained diffusion model")

 
    # Random GT image
    gt_folder = Path("repaint_simplified/data/gt")
    gt_files = list(gt_folder.glob("*.png"))

    gt_path = random.choice(gt_files)
    print("Selected GT:", gt_path.name)

    gt = load_image(gt_path, image_size).to(device)

 
    # Random mask
    mask_folder = Path("repaint_simplified/data/masks")
    mask_files = list(mask_folder.glob("*.png"))

    mask_path = random.choice(mask_files)
    print("Selected mask:", mask_path.name)

    mask = load_mask(mask_path, image_size).to(device)

 
    # Create masked image
    masked_gt = gt * (mask)

 
    # Run RePaint

    print("Running RePaint sampling...")

    with torch.no_grad():
        output = repaint_sample(
            model,
            diffusion,
            gt,
            mask,
            device
        )

 
    # Convert to numpy

    gt_np = tensor_to_numpy(gt)
    result_np = tensor_to_numpy(output)

    mask_np = mask[0, 0].cpu().numpy()

    masked_np = tensor_to_numpy(masked_gt)

 
    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    axs[0, 0].imshow(gt_np)
    axs[0, 0].set_title("Ground Truth")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(mask_np, cmap="gray")
    axs[0, 1].set_title("Mask")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(masked_np)
    axs[1, 0].set_title("Masked Image")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(result_np)
    axs[1, 1].set_title("RePaint Result")
    axs[1, 1].axis("off")

    plt.tight_layout()

 
 
    # Save figure
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    save_path = assets_dir / "repaint_result_plot6.png"

    plt.savefig(save_path)
    plt.close()

    print("Saved plot to:", save_path)


if __name__ == "__main__":
    main()