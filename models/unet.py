import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# Sinusoidal timestep embedding
# -----------------------------------------------------------

def timestep_embedding(timesteps, dim):
    """
    Create sinusoidal timestep embeddings.
    timesteps: (B,)
    returns: (B, dim)
    """
    device = timesteps.device
    half_dim = dim // 2

    if half_dim == 0:
        raise ValueError("Embedding dimension too small")

    emb_scale = math.log(10000) / max(half_dim - 1, 1)
    emb = torch.exp(
        torch.arange(half_dim, device=device) * -emb_scale
    )

    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


# -----------------------------------------------------------
# Residual Block
# -----------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))

        # inject time embedding
        time_emb = self.time_mlp(F.silu(t_emb))
        h = h + time_emb[:, :, None, None]

        h = self.conv2(F.silu(self.norm2(h)))

        return h + self.shortcut(x)


# -----------------------------------------------------------
# Self-Attention Block (single-head, simple)
# -----------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.reshape(B, C, H * W)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W)

        attn = torch.bmm(q.permute(0, 2, 1), k)
        attn = attn * (C ** -0.5)
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.reshape(B, C, H, W)

        return x + self.proj(out)


# -----------------------------------------------------------
# UNet
# -----------------------------------------------------------

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_emb_dim=256,
    ):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        # time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # input conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---------------- Down ----------------
        self.down1 = ResBlock(base_channels, base_channels, time_emb_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)

        self.downsample1 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1)

        self.down3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        self.downsample2 = nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1)

        # ---------------- Middle ----------------
        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid_attn = SelfAttention(base_channels * 4)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # ---------------- Up ----------------
        self.upsample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.up1 = ResBlock(base_channels * 8, base_channels * 2, time_emb_dim)

        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.up2 = ResBlock(base_channels * 4, base_channels, time_emb_dim)

        # output
        self.norm_out = nn.GroupNorm(8, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x, t):
        # time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # input
        x1 = self.conv_in(x)

        # down
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x3_ds = self.downsample1(x3)

        x4 = self.down3(x3_ds, t_emb)
        x4_ds = self.downsample2(x4)

        # middle
        mid = self.mid1(x4_ds, t_emb)
        mid = self.mid_attn(mid)
        mid = self.mid2(mid, t_emb)

        # up
        u1 = self.upsample1(mid)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.up1(u1, t_emb)

        u2 = self.upsample2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.up2(u2, t_emb)

        out = self.conv_out(F.silu(self.norm_out(u2)))

        return out