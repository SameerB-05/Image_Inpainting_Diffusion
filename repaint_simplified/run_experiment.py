import argparse
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from openai_guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults
)

from mask import load_mask
from sample_repaint import load_image, tensor_to_numpy, run_repaint


def save_image(np_img, path):
    img = (np_img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def build_model(device):
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

    model.load_state_dict(
        torch.load(weight_path, map_location=device, weights_only=True)
    )

    model.to(device)
    model.eval()

    return model, diffusion


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)

    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--jump_length", type=int, default=10)
    parser.add_argument("--jump_n_sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_gif", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 256

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)

    # Load model
    model, diffusion = build_model(device)

    # Load inputs
    gt = load_image(args.image, image_size).to(device)
    mask = load_mask(args.mask, image_size).to(device)

    # Run repaint
    print("Running experiment...")

    gif_path = save_dir / "process.gif" if args.save_gif else None

    output, masked_gt, ts = run_repaint(
        model,
        diffusion,
        gt,
        mask,
        device,
        num_steps=args.steps,
        jump_length=args.jump_length,
        jump_n_sample=args.jump_n_sample,
        seed=args.seed,
        save_gif=args.save_gif,
        gif_path=str(gif_path) if gif_path else None
    )

    # Convert to numpy
    gt_np = tensor_to_numpy(gt)
    out_np = tensor_to_numpy(output)
    masked_np = tensor_to_numpy(masked_gt)
    mask_np = mask[0, 0].cpu().numpy()

    # Save images
    save_image(gt_np, save_dir / "gt.png")
    save_image(masked_np, save_dir / "masked.png")
    save_image(out_np, save_dir / "output.png")

    mask_img = (mask_np * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(save_dir / "mask.png")

    # Save config
    config = {
        "image": args.image,
        "mask": args.mask,
        "steps": args.steps,
        "jump_length": args.jump_length,
        "jump_n_sample": args.jump_n_sample,
        "seed": args.seed,
        "num_iterations": len(ts)
    }

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("Saved results to:", save_dir)


if __name__ == "__main__":
    main()