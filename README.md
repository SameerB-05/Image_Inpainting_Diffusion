# Diffusion Models and Image Inpainting — DDPM & RePaint

This repository contains an educational implementation and experimentation with **diffusion-based generative models** and **diffusion-driven image inpainting** using PyTorch.

The project focuses on understanding:

* **Denoising Diffusion Probabilistic Models (DDPM)** through a full training pipeline on the CelebA dataset
* **RePaint**, a sampling-based inpainting method that conditions diffusion models on known image regions

The repository includes both:

* a **from-scratch diffusion model implementation**
* a **simplified implementation of the RePaint inpainting algorithm**

The goal is to understand how diffusion models work internally and how their **sampling process can be modified for conditional generation tasks such as image inpainting**.

> **Note on training:** Due to compute constraints, the custom DDPM was trained with a limited budget, resulting in characteristic "cloud-like" outputs typical of undertrained diffusion models. The full training pipeline is preserved to demonstrate the implementation. The focus then shifted to implementing the RePaint sampling algorithm on top of OpenAI's pretrained weights, which yields strong inpainting results.

---

## Project Components

| Component | Description |
|---|---|
| **Custom DDPM** | DDPM trained from scratch on CelebA 64×64 using PyTorch |
| **RePaint (Simplified)** | RePaint inpainting using OpenAI's pretrained 256×256 Guided Diffusion model — no retraining required |

---

## Repository Structure

```
.
├── train.py                    # DDPM training script
├── sample.py                   # DDPM sampling script
├── celeba_download.py          # CelebA dataset downloader
├── requirements.txt
│
├── models/
│   └── unet.py                 # Custom U-Net denoising network
│
├── diffusion/
│   ├── gaussian_diffusion.py   # Forward/reverse diffusion logic
│   └── scheduler.py            # Cosine noise schedule
│
├── utils/
│   └── dataset.py              # CelebA dataset loader
│
├── data/
│   └── celeba/
│       └── img_align_celeba/   # CelebA 64×64 images (~202k images)
│
├── checkpoints/
│   └── ddpm_celeba64/
│       ├── ddpm_celeba64_best.pth
│       ├── losses.npy
│       └── training_loss.png
│
├── repaint_simplified/
│   ├── mask.py                 # Mask loading and preprocessing
│   ├── repaint_sampler.py      # RePaint sampling schedule and algorithm
│   ├── sample_repaint.py       # Inference script
│   ├── data/
│   │   ├── gt/                 # Ground truth images (256×256)
│   │   └── masks/              # Binary inpainting masks
│   ├── openai_guided_diffusion/ # OpenAI Guided Diffusion implementation
│   ├── pretrained_weights/
│   │   └── 256x256_diffusion.pt
│   └── utils/
│       └── get_layers_list.py
│
└── assets/
    ├── repaint_result_plot*.png
    └── self_tried_training_result.png
```

---

## Part 1 — Custom DDPM

### Model Architecture

The denoising network is implemented using a **U-Net backbone** that predicts the noise component added during diffusion.

**Input/Output:**
```
Input:  x_t (noisy image), t (diffusion timestep)
Output: εθ(x_t, t) (predicted noise)
```

**U-Net Structure:**
```
Encoder:    Conv → ResBlocks → Strided Conv (downsample) → ...
Bottleneck: ResBlock → Self-Attention → ResBlock
Decoder:    TransposedConv (upsample) + Skip Connections → ResBlocks → ...
```

---

### Residual Blocks

Each residual block contains:

* **GroupNorm (8 groups)**
* **SiLU activation**
* **3×3 convolution**
* **Residual skip connection**

Timestep embeddings are injected through a linear projection and added to intermediate feature maps.

---

### Timestep Embedding

The model uses **sinusoidal positional embeddings** similar to Transformers:

$$
\text{emb}(t) = [\sin(\omega_1 t), \cos(\omega_1 t), \ldots]
$$

The embedding is passed through an **MLP with SiLU activation** before being injected into residual blocks.

---

### Self-Attention Layer

A **single-head self-attention block** is used in the bottleneck to capture global spatial dependencies:

1. Group normalization
2. QKV projection (1×1 conv)
3. Scaled dot-product attention
4. Output projection

---

### U-Net Configuration

| Parameter                | Value                |
| ------------------------ | -------------------- |
| Input channels           | 3                    |
| Output channels          | 3                    |
| Base channels            | 64                   |
| Time embedding dimension | 256                  |
| Normalization            | GroupNorm (8 groups) |
| Activation               | SiLU                 |

---

### Diffusion Process

#### Forward Diffusion

Images are progressively corrupted by Gaussian noise:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$$

where $\epsilon \sim \mathcal{N}(0, I)$.

#### Noise Schedule

A **cosine noise schedule** is used:

$$
\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)
$$

with $s = 0.008$, $T = 1000$.

Buffers stored in the model: `beta`, `alpha`, `alpha_cumprod`, `alpha_cumprod_prev`, `sqrt_acp`, `sqrt_omacp`.

#### Training Objective

$$\mathcal{L} = \mathbb{E}_{x_0,t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

Implemented as `MSELoss(predicted_noise, true_noise)`.

---

### Training Configuration

#### Dataset

| Setting          | Value         |
| ---------------- | ------------- |
| Dataset          | CelebA        |
| Image size       | 64 × 64       |
| Max samples used | 35,840 images |

#### DataLoader

| Parameter             | Value |
| --------------------- | ----- |
| Batch size            | 128   |
| Gradient accumulation | 2     |
| Workers               | 4     |
| Pin memory            | True  |

#### Optimizer

| Setting        | Value  |
| -------------- | ------ |
| Optimizer      | Adam   |
| Learning rate  | 2e-4   |
| Grad clip norm | 1.0    |
| EMA decay      | 0.9999 |

**Gradient Accumulation** (`accum_steps = 2`) simulates larger effective batch sizes. **Exponential Moving Average (EMA)** stabilizes training and improves sample quality — EMA weights are used for checkpoint saving and sampling.

#### Training Loop (per batch)

1. Sample timestep $t$
2. Sample Gaussian noise $\epsilon$
3. Generate noisy image $x_t$
4. Predict noise $\hat{\epsilon}_\theta$ via U-Net
5. Compute MSE loss
6. Backpropagate and update parameters

Best checkpoints are saved based on lowest training loss as `.pth` files, alongside loss arrays (`.npy`) and plots (`.png`).

---

### Sampling (DDPM)

Generation starts from pure Gaussian noise: $x_T \sim \mathcal{N}(0, I)$.

The model iteratively denoises for $T = 1000$ steps. At each timestep:

1. Predict noise $\hat{\epsilon}_\theta(x_t, t)$
2. Estimate clean image $\hat{x}_0$
3. Compute posterior mean $\mu_t$
4. For $t > 0$: sample $x_{t-1} = \mu_t + \sigma_t z$; for $t = 0$: $x_0 = \mu_t$

Final images are clamped to $[-1, 1]$ then mapped to $[0, 1]$ for visualization. Samples are displayed as a **4×4 grid**.

---

## Part 2 — RePaint Inpainting

### Method Overview

RePaint performs inpainting by **modifying the sampling procedure of a pretrained unconditional DDPM**, without any retraining. Known pixels from the ground truth are injected during sampling, while unknown regions are generated. Repeated resampling improves boundary consistency between the two regions.

---

### Model Architecture (OpenAI Guided Diffusion)

The inpainting backbone uses the **UNetModel** from OpenAI Guided Diffusion.

| Parameter                 | Value              |
| ------------------------- | ------------------ |
| Image size                | 256 × 256          |
| Input channels            | 3 (RGB)            |
| Base channels             | 256                |
| Residual blocks per stage | 2                  |
| Channel multipliers       | (1, 1, 2, 2, 4, 4) |
| Attention heads           | 4                  |
| Attention head channels   | 64                 |
| Attention resolutions     | 32, 16, 8          |
| Class conditional         | Enabled            |
| Learn sigma               | Enabled            |
| Up/Down sampling          | ResBlock up/down   |

**Channel hierarchy** for a 256×256 image:
```
256 → 256 → 512 → 512 → 1024 → 1024
```

Self-attention is inserted at spatial resolutions **32×32**, **16×16**, and **8×8**, capturing the long-range dependencies critical for coherent inpainting across large masked regions.

#### Noise Schedule (OpenAI Model)

A **linear beta schedule** is used:

$$\beta_t \in [0.0001,\ 0.02]$$

Generated via `get_named_beta_schedule("linear", 1000)` and automatically rescaled for the chosen number of sampling steps.

| Setting        | Value                 |
| -------------- | --------------------- |
| Training steps | T = 1000              |
| Sampling steps | 250 (with resampling) |
| Noise schedule | Linear                |

---

### Pretrained Weights

| File                   | Source                           |
| ---------------------- | -------------------------------- |
| `256x256_diffusion.pt` | OpenAI Guided Diffusion          |
| `celeba256_250000.pt`  | OpenAI Guided Diffusion (CelebA) |

The model is instantiated using `create_model_and_diffusion()`.

---

### RePaint Algorithm

Let $x_0$ be the ground truth image and $m$ the binary mask:
```
m = 1  →  known pixels (preserve)
m = 0  →  pixels to inpaint (generate)
```

Sampling begins from pure Gaussian noise: $x_T \sim \mathcal{N}(0, I)$.

**Step 1 — Predict Unknown Region**

Standard reverse diffusion step for the unknown region:

$$x_{t-1}^{\text{unknown}} \sim p_\theta(x_{t-1} \mid x_t)$$

**Step 2 — Inject Known Region**

Known pixels are sampled from the forward diffusion distribution of the ground truth at timestep $t$:

$$
x_{t-1}^{\text{known}} \sim \mathcal{N}\left(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I\right)
$$

> **Key design choice:** Rather than copying clean ground-truth pixels directly, known pixels are sampled from the forward diffusion distribution at the current noise level. This ensures both regions share consistent noise statistics at every timestep.

**Step 3 — Merge Using Mask**

$$x_{t-1} = m \odot x_{t-1}^{\text{known}} + (1 - m) \odot x_{t-1}^{\text{unknown}}$$

In code:
```python
noisy_gt = diffusion.q_sample(gt, t)
x = mask * noisy_gt + (1 - mask) * x
```

**Step 4 — Resampling**

Direct conditioning can produce locally plausible but globally inconsistent results (e.g., texture misalignment at boundaries). RePaint addresses this by diffusing $x_{t-1}$ forward one step:

$$
x_t \sim \mathcal{N}\left(\sqrt{1-\beta_t} x_{t-1}, \beta_t I\right)
$$

...then denoising again. This loop runs $U$ times per timestep, giving the model multiple opportunities to reconcile generated content with the known context.

---

### Sampling Schedule

The non-monotonic schedule is generated by:

```python
get_schedule(t_T=250, jump_len=10, jump_n_sample=10)
```

| Parameter       | Value | Meaning                     |
| --------------- | ----- | --------------------------- |
| `t_T`           | 250   | Total sampling steps        |
| `jump_len`      | 10    | Length of each forward jump |
| `jump_n_sample` | 10    | Number of resampling cycles |

Instead of strictly progressing $T \to 0$, the schedule alternates forward and backward:

```
t → t-1 → t-2 → t-3
            ↑
       forward jump
t-2 → t-1 → t
```

This jump mechanism significantly improves **boundary consistency and semantic coherence** between known and generated regions.

---

### Image and Mask Representation

**Images** are normalized to `[-1, 1]`:

```python
img = np.array(img) / 255.0
img = img * 2 - 1
```

Tensor format: `[B, C, H, W]` — mapped back to `[0, 1]` for visualization.

**Masks** are processed in `mask.py`:

1. Load as grayscale image
2. Resize to image resolution
3. Normalize to `[0, 1]`
4. Binarize at threshold 0.5: `mask = (mask > 0.5).astype(np.float32)`
5. Convert to tensor shape `[1, 1, H, W]`

---

### Sampling Pipeline

The full inference process in `sample_repaint.py`:

1. Load pretrained diffusion model (`256x256_diffusion.pt`)
2. Load ground truth image (`data/gt/*.png`)
3. Load binary mask (`data/masks/*.png`)
4. Create masked image: `masked = gt * mask`
5. Run RePaint sampling: `output = repaint_sample(...)`
6. Convert output from `[-1, 1]` to `[0, 1]`
7. Save visualization to `assets/repaint_result_plot.png`

The script produces a **2×2 comparison grid**:

| Ground Truth | Mask           |
| ------------ | -------------- |
| Masked Image | RePaint Result |

---

## Results

### Part 1 — Custom DDPM Samples

The model was trained from scratch on CelebA 64×64 with a limited compute budget. The outputs exhibit the characteristic "cloud-like" blurriness of an undertrained diffusion model — global structure begins to emerge, but fine facial details are not yet resolved. The full training pipeline is intact and would converge given additional training time and compute.

<p align="center">
  <img src="assets/self_tried_training_result.png" width="420">
  <br>
  <em>4×4 grid of generated samples from the custom DDPM (CelebA 64×64, undertrained)</em>
</p>

---

### Part 2 — RePaint Inpainting Results

The following results are produced by running the RePaint sampling algorithm on top of OpenAI's pretrained 256×256 Guided Diffusion model. Each figure shows a 2×2 grid: **Ground Truth** | **Mask** / **Masked Image** | **RePaint Output**.

The inpainted regions show strong semantic coherence with the surrounding context — textures, lighting, and structure align naturally at the mask boundaries, which is the direct benefit of the resampling jump schedule.

<p align="center">
  <img src="assets/repaint_result_plot3.png" width="380">
  <img src="assets/repaint_result_plot4.png" width="380">
</p>

<p align="center">
  <img src="assets/repaint_result_plot5.png" width="380">
  <img src="assets/repaint_result_plot6.png" width="380">
</p>

<p align="center">
  <em>RePaint inpainting results across four test images (256×256). Each panel: Ground Truth, Mask, Masked Input, Inpainted Output.</em>
</p>

---

## Component Comparison

| Feature          | Custom DDPM              | RePaint (Simplified)      |
| ---------------- | ------------------------ | ------------------------- |
| Training         | From scratch             | Uses pretrained weights   |
| Dataset          | CelebA 64×64             | ImageNet / CelebA 256×256 |
| Task             | Unconditional generation | Image inpainting          |
| Noise schedule   | Cosine                   | Linear                    |
| Sampling steps   | 1000                     | 250 (with resampling)     |
| Attention        | Bottleneck only          | 32×32, 16×16, 8×8         |

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `PIL`, `matplotlib`, `tqdm`

---

## Usage

**Train the diffusion model:**
```bash
python train.py
```

**Generate samples:**
```bash
python sample.py
```

**Run RePaint inpainting:**
```bash
python repaint_simplified/sample_repaint.py
```

---

## References

* **RePaint** — https://github.com/andreas128/RePaint
* **OpenAI Guided Diffusion** — https://github.com/openai/guided-diffusion
* **Original DDPM implementation** — https://github.com/hojonathanho/diffusion

---

## Author

Independent exploration of diffusion-based generative modeling and diffusion-based image inpainting using PyTorch.
