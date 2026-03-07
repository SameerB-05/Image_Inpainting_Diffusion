import torch
import matplotlib.pyplot as plt

from models.unet import UNet
from diffusion.gaussian_diffusion import DDPM

# EMA helper (same as in train.py)
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def load_shadow(self, state_dict):
        # load EMA weights from checkpoint (optional)
        for name, param in self.model.named_parameters():
            if name in state_dict:
                self.shadow[name] = state_dict[name].clone()

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # ------------------ BUILD MODEL ------------------
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_emb_dim=256
    )
    model = DDPM(unet, T=1000).to(device)

    # ------------------ LOAD CHECKPOINT ------------------
    #ckpt_path = "checkpoints/ddpm_celeba64/ddpm_celeba64_resume_20260221_114711.pth"
    ckpt_path = "checkpoints/ddpm_celeba64/ddpm_celeba64_best.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ------------------ APPLY EMA WEIGHTS ------------------
    ema = EMA(model)
    if "model_state_dict" in ckpt:
        ema.load_shadow(ckpt["model_state_dict"])  # use the same saved weights as EMA (optional)
    ema.apply_shadow()  # for sampling

    print("Checkpoint loaded. Sampling with EMA weights...")

    # ------------------ SAMPLE ------------------
    with torch.no_grad():
        samples = model.sample(
            shape=(16, 3, 64, 64),
            device=device
        ).cpu()

    # Convert from [-1, 1] → [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    # ------------------ VISUALIZE ------------------
    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axs.flatten()):
        img = samples[i].permute(1, 2, 0)  # CHW → HWC
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    ema.restore()  # optional, not really needed after sampling


if __name__ == "__main__":
    main()