import torch
import random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio

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


def run_repaint(
    model,
    diffusion,
    gt,
    mask,
    device,
    num_steps=250,
    jump_length=10,
    jump_n_sample=10,
    seed=0,
    save_gif=False,
    gif_path=None
):

    # reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    masked_gt = gt * mask

    with torch.no_grad():
        if save_gif:
            output, ts, frames = repaint_sample(
                model,
                diffusion,
                gt,
                mask,
                device,
                num_steps=num_steps,
                jump_length=jump_length,
                jump_n_sample=jump_n_sample,
                return_frames=True
            )
        else:
            output, ts = repaint_sample(
                model,
                diffusion,
                gt,
                mask,
                device,
                num_steps=num_steps,
                jump_length=jump_length,
                jump_n_sample=jump_n_sample
            )

    # Save GIF if required
    if save_gif and gif_path is not None:
        gif_frames = []
        for f in frames:
            img = (f.clamp(-1, 1) + 1) / 2
            img = img[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            gif_frames.append(img)

        imageio.mimsave(gif_path, gif_frames, duration=0.05)

    return output, masked_gt, ts


def save_visualization(gt, mask, masked, output, save_path):

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    axs[0, 0].imshow(gt)
    axs[0, 0].set_title("Ground Truth")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(mask, cmap="gray")
    axs[0, 1].set_title("Mask")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(masked)
    axs[1, 0].set_title("Masked Image")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(output)
    axs[1, 1].set_title("RePaint Result")
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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
        class_cond=False,
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="250"
    ))

    model, diffusion = create_model_and_diffusion(**defaults)


    weight_path = Path("repaint_simplified/pretrained_weights/256x256_diffusion_uncond.pt")

    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print("Loaded pretrained diffusion model")

 
    # Random GT image
    gt_folder = Path("repaint_simplified/data/gt")
    gt_files = list(gt_folder.glob("*.png"))

    #gt_path = random.choice(gt_files)
    gt_path = Path(r"repaint_simplified\data\gt\inet_0015.png")
    print("Selected GT:", gt_path.name)

    gt = load_image(gt_path, image_size).to(device)

 
    # Random mask
    mask_folder = Path("repaint_simplified/data/masks")
    mask_files = list(mask_folder.glob("*.png"))

    #mask_path = random.choice(mask_files)
    mask_path = Path(r"repaint_simplified\data\masks\000010.png")
    print("Selected mask:", mask_path.name)

    mask = load_mask(mask_path, image_size).to(device)

 
    # Create masked image
    masked_gt = gt * (mask)

 
    # Run RePaint

    print("Running RePaint sampling...")

    output, masked_gt, ts = run_repaint(
        model,
        diffusion,
        gt,
        mask,
        device,
        save_gif=True,
        gif_path="assets/repaint_process.gif"
    )

 
    # Convert to numpy

    gt_np = tensor_to_numpy(gt)
    result_np = tensor_to_numpy(output)

    mask_np = mask[0, 0].cpu().numpy()

    masked_np = tensor_to_numpy(masked_gt)

    # Save figure
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    save_path = assets_dir / "repaint_result_plot8.png"

    save_visualization(
            gt_np,
            mask_np,
            masked_np,
            result_np,
            save_path
        )

    print("Saved plot to:", save_path)


if __name__ == "__main__":
    main()