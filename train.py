import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime
import argparse

from utils.dataset import CelebADataset
from models.unet import UNet
from diffusion.gaussian_diffusion import DDPM

# ------------------ PATHS ------------------
CHECKPOINT_DIR = "checkpoints/ddpm_celeba64"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------ Exponential Moving Average CLASS ------------------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

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

# ------------------ TRAIN FUNCTION ------------------
def train(resume=False, ckpt_path=None, accum_steps=2, epochs=22):
    """
    resume: bool, whether to continue from previous checkpoint
    ckpt_path: str, path to checkpoint file to resume from (required if resume=True)
    accum_steps: int, number of mini-batches to accumulate before optimizer step
    epochs: int, number of additional epochs to train
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------ DATA ------------------
    dataset = CelebADataset(
        root_dir="data/celeba/img_align_celeba",
        image_size=64,
        max_samples=280*128
    )

    loader = DataLoader(
        dataset,
        batch_size=128 // accum_steps,  # mini-batch size
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # ------------------ MODEL ------------------
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_emb_dim=256
    )
    model = DDPM(unet, T=1000).to(device)
    ema = EMA(model, decay=0.9999)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # ------------------ TIMESTAMP ------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------ RESUME OR SCRATCH ------------------
    if resume:
        if ckpt_path is None or not os.path.isfile(ckpt_path):
            raise ValueError("Checkpoint path invalid. Cannot resume.")

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint.get("loss", float("inf"))
        print(f"Resuming training from epoch {start_epoch} with best_loss={best_loss:.6f}")

        # create NEW checkpoint path with timestamp
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ddpm_celeba64_resume_{timestamp}.pth")
    else:
        start_epoch = 0
        best_loss = float("inf")
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ddpm_celeba64_{timestamp}.pth")
        print(f"Training from scratch. New checkpoint will be saved at {ckpt_path}")

    # ------------------ LOSSES PATHS ------------------
    losses_npy_path = os.path.join(CHECKPOINT_DIR, f"losses_{timestamp}.npy")
    loss_plot_path = os.path.join(CHECKPOINT_DIR, f"training_loss_{timestamp}.png")

    # ------------------ LOAD EXISTING LOSSES IF RESUMING ------------------
    if resume and os.path.isfile(os.path.join(CHECKPOINT_DIR, "losses.npy")):
        losses = list(np.load(os.path.join(CHECKPOINT_DIR, "losses.npy")))
        print(f"Loaded previous {len(losses)} epochs of loss")
    else:
        losses = []

    # ------------------ TRAIN LOOP ------------------
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, x in enumerate(loader):
            x = x.to(device)
            eps_pred, noise = model(x)
            loss = F.mse_loss(eps_pred, noise) / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # handle leftover batches
        if (batch_idx + 1) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{start_epoch + epochs} | Loss: {avg_loss:.6f}")

        # ------------------ SAVE BEST CHECKPOINT ------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            ema.apply_shadow()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "T": model.T,
                },
                ckpt_path
            )
            ema.restore()
            print(f"Saved best checkpoint: {ckpt_path}")

    # ------------------ SAVE LOSSES AND PLOT ------------------
    losses_np = np.array(losses)
    np.save(losses_npy_path, losses_np)

    plt.figure()
    plt.plot(losses_np)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss for DDPM (CelebA 64x64)")
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved {losses_npy_path} and {loss_plot_path}")

# ------------------ CLI ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to checkpoint file (required if --resume)"
    )
    parser.add_argument(
        "--accum_steps", type=int, default=2, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--epochs", type=int, default=22, help="Number of additional epochs to train"
    )
    args = parser.parse_args()

    train(
        resume=args.resume,
        ckpt_path=args.ckpt_path,
        accum_steps=args.accum_steps,
        epochs=args.epochs
    )