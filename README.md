# Diffusion-Based Image Inpainting — A Study of RePaint

This repository is an implementation and experimental study of **RePaint** — a sampling-based method for image inpainting using pretrained denoising diffusion probabilistic models. The central question it addresses: *can an unconditional diffusion model be conditioned on known pixels purely through its sampling procedure, with no finetuning or retraining?*

The implementation builds on OpenAI's pretrained 256×256 Guided Diffusion model. All conditioning happens at inference time through a modified sampling procedure. A from-scratch DDPM trained on CelebA 64×64 is also included — it was the starting point for building intuition about diffusion model internals and motivates the shift to the pretrained-model approach.

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
├── diffusion/
│   ├── gaussian_diffusion.py   # Forward/reverse diffusion logic
│   └── scheduler.py            # Cosine noise schedule
├── utils/
│   └── dataset.py              # CelebA dataset loader
│
├── checkpoints/
│   └── ddpm_celeba64/          # Training checkpoints (contents gitignored)
│
└── repaint_simplified/
    ├── mask.py                  # Mask loading and preprocessing
    ├── repaint_sampler.py       # Sampling schedule and core RePaint loop
    ├── sample_repaint.py        # Inference entry point
    ├── run_experiment.py        # CLI experiment runner
    ├── plot_results.py          # Visualization toolkit
    ├── plot_schedule.py         # Timestep schedule visualization
    │
    ├── data/
    │   ├── gt/                  # Ground truth images (256×256, 9 images)
    │   └── masks/               # Binary inpainting masks (7 masks)
    │
    ├── experiments/
    │   ├── exp_resampling.py    # Effect of the resampling mechanism
    │   ├── exp_jumps.py         # Ablation over jump_length and jump_n_sample
    │   ├── exp_diversity.py     # Stochastic output diversity across seeds
    │   ├── exp_masks.py         # Robustness across mask geometries
    │   ├── exp_compute.py       # Compute cost comparison
    │   └── metrics.py           # L1, L2, and LPIPS evaluation (not yet used)
    │
    ├── openai_guided_diffusion/ # OpenAI Guided Diffusion UNet and diffusion logic
    ├── pretrained_weights/      # Pretrained model weights (.pt files)
    └── utils/
        └── get_layers_list.py
```

---

## The RePaint Algorithm

RePaint performs inpainting by **modifying the sampling procedure of a pretrained unconditional DDPM** — no finetuning required. At each reverse diffusion step, known pixels are replaced with noisy versions of the ground truth at the corresponding noise level, while the unknown region continues to evolve under the model's learned reverse process. A resampling mechanism then improves consistency at the boundary between the two regions.

### Notation

| Symbol | Meaning |
|--------|---------|
| $x_0$ | Ground truth image |
| $m$ | Binary mask ($m=1$: known, $m=0$: inpaint) |
| $\bar{\alpha}_t$ | Cumulative noise product at timestep $t$ |
| $\beta_t$ | Noise variance at timestep $t$ |
| $p_\theta$ | Learned reverse diffusion distribution |

Sampling begins from pure Gaussian noise: $x_T \sim \mathcal{N}(0, I)$.

---

### Per-Timestep Procedure

**Step 1 — Reverse diffusion step (unknown region)**

$$x_{t-1}^{\text{unknown}} \sim p_\theta(x_{t-1} \mid x_t)$$

The UNet predicts the noise component; the reverse posterior is sampled via the standard DDPM formula.

**Step 2 — Forward injection (known region)**

Rather than inserting clean ground-truth pixels — which would create a noise-level mismatch with the generated region — known pixels are noised to the current timestep $t$ using the forward process:

$$x_{t-1}^{\text{known}} = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

```python
noisy_gt = diffusion.q_sample(gt, t_tensor)
x = mask * noisy_gt + (1 - mask) * x
```

This ensures both regions carry consistent noise statistics, which is essential for the model to reason coherently across the boundary.

> **Note on deviation from the paper:** The algorithm (line 5) specifies $x_{t-1}^{\text{known}} = \sqrt{\bar{\alpha}_t}\, x_0 + (1 - \bar{\alpha}_t)\, \epsilon$ — with subscript $t-1$ on the left, implying the known region should be noised to level $t-1$, i.e. `q_sample(gt, t-1)`. This implementation calls `q_sample(gt, t)` instead, injecting a slightly more noisy version of the known region than the algorithm strictly specifies. This is a minor off-by-one that many implementations make in practice and has negligible effect on output quality.

**Step 3 — Mask merge**

$$x_{t-1} = m \odot x_{t-1}^{\text{known}} + (1 - m) \odot x_{t-1}^{\text{unknown}}$$

**Step 4 — Resampling (jump schedule)**

Mask merging alone can produce locally plausible but globally inconsistent results — the model generates the unknown region without sufficient awareness of its surroundings. RePaint addresses this by diffusing $x_{t-1}$ forward one step and denoising again:

$$x_{t_{\text{next}}} \sim \mathcal{N}\left(\sqrt{1 - \beta_{t_{\text{next}}}}\, x_{t-1}, \beta_{t_{\text{next}}}\, I\right) \quad \longrightarrow \quad \text{denoise again}$$

where $t_{\text{next}} > t$ is the target timestep in the forward jump, and `diffusion.betas[t_next]` is used in the code. In the paper's notation this is written as $\beta_{t-1}$ because the paper indexes the jump relative to the current $t$ — both refer to the same $\beta$ value at the destination timestep.

This loop runs $U$ times per window, giving the model repeated exposure to the boundary at each noise level.

---

### Sampling Schedule

The non-monotonic timestep schedule is generated by `repaint_sampler.get_schedule`:

```python
get_schedule(t_T=250, jump_length=10, jump_n_sample=10)
```

| Parameter       | Default | Meaning                                   |
|-----------------|---------|-------------------------------------------|
| `t_T`           | 250     | Total diffusion steps                     |
| `jump_length`   | 10      | Number of steps in each resampling window |
| `jump_n_sample` | 10      | Resampling repetitions per window         |

Instead of monotonically decreasing $T \to 0$, the schedule interleaves forward jumps. For example, with `jump_length=3` and `jump_n_sample=2`, a segment of the trajectory looks like:

```
Reverse (denoise):    10 → 9 → 8 → 7
                                    ↓
Forward jump (noise):           7 → 8 → 9 → 10   (jump back up by jump_length)
                                                  ↓
Reverse (denoise):        10 → 9 → 8 → 7         (repeat jump_n_sample times)
                                    ↓
Forward jump (noise):           7 → 8 → 9 → 10
                                                  ↓
Reverse (denoise):        10 → 9 → 8 → 7 → 6 → 5 → ...  (continue decreasing)
```

Each window of `jump_length` steps is resampled `jump_n_sample` times before the schedule moves on. The total number of UNet forward passes is therefore substantially higher than the nominal `t_T` count.

Setting `jump_length=1, jump_n_sample=1` disables the resampling jumps, reducing to a single-pass conditioned sampling loop. The known-pixel injection and mask merge still run at every step — this is not vanilla unconditional DDPM sampling, but it serves as the no-resampling baseline in the ablation experiments below.

---

## Model

The inpainting backbone is the **UNetModel** from OpenAI's Guided Diffusion, loaded with pretrained weights. No modifications to model weights are made.

| Parameter                 | Value              |
|---------------------------|--------------------|
| Image size                | 256 × 256          |
| Base channels             | 256                |
| Channel multipliers       | (1, 1, 2, 2, 4, 4) |
| Residual blocks per stage | 2                  |
| Attention heads           | 4                  |
| Attention head channels   | 64                 |
| Attention resolutions     | 32, 16, 8          |
| Noise schedule            | Linear β ∈ [0.0001, 0.02] |
| Class conditioning        | None (unconditional)       |
| Learn sigma               | Enabled (model outputs predicted variance alongside noise) |

Self-attention at spatial resolutions 32×32, 16×16, and 8×8 captures the long-range dependencies critical for coherent inpainting across large masked regions.

**Pretrained weights:**

| File | Description | Used |
|------|-------------|------|
| `256x256_diffusion_uncond.pt` | OpenAI unconditional diffusion model, ImageNet 256×256 | ✓ Primary model |
| `256x256_diffusion.pt` | OpenAI class-conditional diffusion model, ImageNet 256×256 | Downloaded, unused |
| `celeba256_250000.pt` | Guided diffusion model trained by RePaint authors on CelebA-HQ 256×256 (250k iterations) | Downloaded, unused |
| `256x256_classifier.pt` | OpenAI ImageNet classifier | Downloaded, unused |

> **On conditioning in the original RePaint repo:** The official implementation has two independent conditioning mechanisms — class conditioning (passing a label `y` to the U-Net) and classifier guidance (using `grad(log p(y|x_noisy))` to nudge the score function toward a target class at each step). These can be enabled independently via config. For example, the `test_inet256_genhalf` config uses both: `class_cond: true` with `256x256_diffusion.pt` plus `256x256_classifier.pt` for full guided diffusion. The `face_example` config uses neither effectively: `class_cond: false` and no `classifier_path`, so guidance is disabled and `celeba256_250000.pt` runs unconditionally.
>
> **This simplified implementation removes both mechanisms entirely.** There is no `cond_fn`, no classifier load, and `model_kwargs=None` is passed at every sampling step. The unconditional model `256x256_diffusion_uncond.pt` is used directly.

---

## Codebase

### `repaint_sampler.py`

The core of the repository. `get_schedule` builds the non-monotonic timestep sequence using a dictionary-tracked jump counter, emitting the sequence by alternating decrements (reverse steps) and increments (forward jumps). `repaint_sample` executes the full loop — for each transition it applies the reverse diffusion step, injects known pixels via `q_sample`, merges with the mask, and applies the forward jump when `t_next > t`. Intermediate frames can be collected at fixed intervals for GIF visualization.

Note that the paper's algorithm has an explicit inner loop over $u = 1, \ldots, U$. Here, the resampling is encoded implicitly: `get_schedule` pre-expands the jump repetitions into a flat timestep list, so there is no explicit `U` variable — `jump_n_sample` controls the number of resamplings indirectly through the schedule construction. This is functionally equivalent to the paper's formulation.

### `sample_repaint.py`

Primary inference entry point. Handles device setup, model instantiation via `create_model_and_diffusion`, image/mask loading, seed control, and output visualization. Produces a 2×2 comparison grid: ground truth, binary mask, masked input, and RePaint output.

### `run_experiment.py`

CLI wrapper accepting the following arguments: `image`, `mask`, `steps`, `jump_length`, `jump_n_sample`, `seed`, `save_dir`, and `save_gif`. Saves output images and a JSON metadata file per run — used by all experiment scripts for systematic configuration tracking.

### `mask.py`

Mask preprocessing pipeline: load as grayscale → resize to target resolution → normalize to $[0,1]$ → binarize at threshold 0.5 → convert to tensor shape $[1, 1, H, W]$.

### `plot_results.py`

Visualization toolkit with modes for: single-run display, resampling comparison (baseline vs RePaint), diversity grids across seeds, jump parameter ablation grids, and mask geometry comparisons organized by image row. Results are saved to `outputs/result_plots/`.

### `plot_schedule.py`

Plots the RePaint timestep sequence against iteration index for a given `(t_T, jump_length, jump_n_sample)` configuration. Downward slopes are reverse diffusion steps; upward segments are resampling jumps. Plots are saved to `outputs/schedule_plots/{t_T}_{jump_length}_{jump_n_sample}.png`.

---

## Inference

```
1. Load pretrained model        (256x256_diffusion_uncond.pt)
2. Load ground truth image      (data/gt/*.png)
3. Load binary mask             (data/masks/*.png)
4. Construct masked image:      x_masked = x_0 ⊙ m
5. Run RePaint sampling:        output = repaint_sample(...)
6. Rescale output:              [-1, 1] → [0, 1]
7. Save 2×2 visualization grid
```

```bash
# Quick inference
python repaint_simplified/sample_repaint.py

# With explicit arguments
python repaint_simplified/run_experiment.py \
    --image repaint_simplified/data/gt/inet_0000.png \
    --mask repaint_simplified/data/masks/000010.png \
    --steps 250 --jump_length 10 --jump_n_sample 10 \
    --seed 42 --save_dir results/
```

---

## Results

The inpainting results demonstrate strong semantic coherence between generated and known regions — textures, lighting, and structure align naturally at mask boundaries without any model finetuning. This is the direct effect of the resampling jump schedule.

<p align="center">
  <img src="assets/repaint_result_plot3.png" width="380">
  <img src="assets/repaint_result_plot4.png" width="380">
</p>

<p align="center">
  <img src="assets/repaint_result_plot5.png" width="380">
  <img src="assets/repaint_result_plot6.png" width="380">
</p>

<p align="center">
  <em>Inpainting results across four test images (256×256). Each panel: Ground Truth | Mask | Masked Input | Inpainted Output.</em>
</p>

---

## Experiments

All experiment scripts live in `repaint_simplified/experiments/`. Each script is self-contained, drives `run_experiment.py` with specific parameter configurations, and saves results to structured output directories. Visualizations are generated via `plot_results.py` and saved to `outputs/result_plots/`.

### Evaluation Metrics (`metrics.py`)

*(Not yet used in experiments — implemented for future quantitative evaluation.)*

Three metrics are defined, computed exclusively over the **unknown region** $(1 - m)$:

$$\text{L1} = \mathbb{E}\left[|(x - x_0) \odot (1-m)|\right]$$

$$\text{L2} = \mathbb{E}\left[(x - x_0)^2 \odot (1-m)\right]$$

$$\text{LPIPS} = \text{perceptual distance via pretrained AlexNet, masked region only}$$

`compute_all_metrics` returns all three. LPIPS captures structural and textural fidelity that pixel-level metrics miss and would be the most informative of the three for evaluating inpainting realism.

---

### Experiment 1 — Resampling Ablation (`exp_resampling.py`)

**Question:** How much does the jump resampling mechanism contribute to output quality?

**Setup:** A fixed image-mask pair is run under two configurations — standard DDPM (no resampling: `jump_length=1, jump_n_sample=1`) and full RePaint — with all other parameters held constant. This isolates the resampling contribution from the known-pixel injection, which is present in both.

**Key finding:** Resampling is critical for boundary coherence. Without it, generated content is often locally plausible but misaligned with surrounding context at mask edges. The jump schedule gives the model repeated exposure to the boundary at each noise level, enabling it to reconcile generated and known regions globally.

**Results** — `outputs/result_plots/resampling_comparison.png`:

<p align="center">
  <img src="repaint_simplified/outputs/result_plots/resampling_comparison.png" width="720">
  <br>
  <em>Left: standard DDPM sampling (no resampling). Right: full RePaint with jump schedule.</em>
</p>

---

### Experiment 2 — Jump Parameter Ablation (`exp_jumps.py`)

**Question:** How do `jump_length` and `jump_n_sample` affect output quality?

**Setup:** A grid of `(jump_length, jump_n_sample)` combinations is run on a fixed image-mask pair:

```
jump_lengths  = [1, 5, 10]
jump_n_sample = [5, 10]
```

Inpainting outputs are saved per configuration and visualized as a comparison grid. Sampling schedule plots are also generated for each combination to show the structure of the timestep trajectory.

**Inpainting results** — `outputs/result_plots/jump_ablation.png`:

<p align="center">
  <img src="repaint_simplified/outputs/result_plots/jump_ablation.png" width="720">
  <br>
  <em>Inpainting outputs across the (jump_length, jump_n_sample) grid.</em>
</p>

**Sampling schedules** — `outputs/schedule_plots/`:

<p align="center">
  <img src="repaint_simplified/outputs/schedule_plots/250_1_5.png" width="460">
  <img src="repaint_simplified/outputs/schedule_plots/250_1_10.png" width="460">
</p>
<p align="center">
  <em>jump_length=1, jump_n_sample=5 &nbsp;&nbsp;&nbsp;&nbsp; jump_length=1, jump_n_sample=10</em>
</p>

<p align="center">
  <img src="repaint_simplified/outputs/schedule_plots/250_5_5.png" width="460">
  <img src="repaint_simplified/outputs/schedule_plots/250_5_10.png" width="460">
</p>
<p align="center">
  <em>jump_length=5, jump_n_sample=5 &nbsp;&nbsp;&nbsp;&nbsp; jump_length=5, jump_n_sample=10</em>
</p>

<p align="center">
  <img src="repaint_simplified/outputs/schedule_plots/250_10_5.png" width="460">
  <img src="repaint_simplified/outputs/schedule_plots/250_10_10.png" width="460">
</p>
<p align="center">
  <em>jump_length=10, jump_n_sample=5 &nbsp;&nbsp;&nbsp;&nbsp; jump_length=10, jump_n_sample=10</em>
</p>

**Interpretation:** `jump_n_sample` controls how many times each resampling window repeats — higher values give the model more reconciliation passes but scale total compute nearly linearly. `jump_length` controls the window size, determining how much noise is reintroduced per jump and how far back the model revisits before denoising again. The schedule plots make the structure of each configuration directly visible: `jump_length=1, jump_n_sample=1` is a straight monotone descent; larger values produce increasingly jagged non-monotone trajectories.

---

### Experiment 3 — Stochastic Diversity (`exp_diversity.py`)

**Question:** How much variation does the model produce across runs on the same input?

**Setup:** A fixed image-mask pair is run across multiple random seeds. All outputs are saved for side-by-side comparison.

**Key finding:** Variation across seeds reflects the model's learned distribution over plausible completions — not instability. Different seeds produce outputs that are semantically consistent with the surrounding known region while differing in fine-grained details, indicating the model has captured meaningful constraints from context rather than memorising a fixed completion.

**Results** — `outputs/result_plots/diversity.png`:

<p align="center">
  <img src="repaint_simplified/outputs/result_plots/diversity.png" width="720">
  <br>
  <em>Multiple inpainting completions for the same input across different random seeds.</em>
</p>

---

### Experiment 4 — Mask Geometry Robustness (`exp_masks.py`)

**Question:** How does RePaint perform across different mask shapes, sizes, and positions?

**Setup:** All combinations of 9 ground truth images and 7 masks are run, covering a range of mask geometries. Outputs are organized hierarchically by image and mask; `plot_results.py` renders them as a grid with rows per mask type.

**Motivation:** RePaint's conditioning mechanism is geometry-agnostic at the algorithm level, but inpainting difficulty varies significantly with mask size and proximity to semantically important regions. This experiment surfaces that variation empirically.

**Results** — `outputs/result_plots/mask_exp.png`:

<p align="center">
  <img src="repaint_simplified/outputs/result_plots/mask_exp.png" width="720">
  <br>
  <em>Inpainting results across all image–mask combinations. Rows correspond to mask types.</em>
</p>

---

### Running All Experiments

```bash
python repaint_simplified/experiments/exp_resampling.py
python repaint_simplified/experiments/exp_jumps.py
python repaint_simplified/experiments/exp_diversity.py
python repaint_simplified/experiments/exp_masks.py

# Generate sampling schedule plots
python repaint_simplified/plot_schedule.py

# Visualize all experiment results
python repaint_simplified/plot_results.py
```

---

## Background: Custom DDPM

Before implementing RePaint, a DDPM was trained from scratch on CelebA 64×64 to build concrete intuition for the forward process, noise schedules, and reverse diffusion mechanics. The full pipeline is intact and included for completeness.

### Architecture

The denoising network is a U-Net that takes a noisy image $x_t$ and timestep $t$ as input and predicts the noise component $\epsilon_\theta(x_t, t)$.

```
Encoder:    Conv → ResBlocks → Strided Conv (downsample) → ...
Bottleneck: ResBlock → Self-Attention → ResBlock
Decoder:    TransposedConv (upsample) + Skip Connections → ResBlocks → ...
```

Each residual block contains GroupNorm (8 groups), SiLU activation, a 3×3 convolution, and a residual skip connection. Timestep embeddings are injected via a linear projection added to intermediate feature maps.

**Timestep embedding** — sinusoidal positional embeddings passed through an MLP with SiLU activation:

$$\text{emb}(t) = [\sin(\omega_1 t),\, \cos(\omega_1 t),\, \ldots]$$

**Self-attention** — a single-head block in the bottleneck: GroupNorm → QKV projection (1×1 conv) → scaled dot-product attention → output projection.

| Parameter                | Value                |
|--------------------------|----------------------|
| Input / output channels  | 3                    |
| Base channels            | 64                   |
| Time embedding dimension | 256                  |
| Normalization            | GroupNorm (8 groups) |
| Activation               | SiLU                 |

### Diffusion Process

**Forward diffusion** corrupts images progressively:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Cosine noise schedule** with $s = 0.008$, $T = 1000$:

$$\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

**Training objective:**

$$\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

implemented as `MSELoss(predicted_noise, true_noise)`.

### Training Configuration

| Setting               | Value         |
|-----------------------|---------------|
| Dataset               | CelebA 64×64  |
| Max samples           | 35,840        |
| Batch size            | 128           |
| Gradient accumulation | 2 steps       |
| Optimizer             | Adam, lr 2e-4 |
| Grad clip norm        | 1.0           |
| EMA decay             | 0.9999        |

Gradient accumulation (`accum_steps=2`) simulates a larger effective batch size. EMA weights are used for checkpoint saving and sampling. Best checkpoints are saved based on lowest training loss as `.pth` files, alongside loss arrays (`.npy`) and plots (`.png`).

### Sampling

Generation starts from $x_T \sim \mathcal{N}(0, I)$ and denoises for $T = 1000$ steps. At each timestep: predict noise $\hat{\epsilon}_\theta(x_t, t)$ → estimate clean image $\hat{x}_0$ → compute posterior mean $\mu_t$ → sample $x_{t-1} = \mu_t + \sigma_t z$ (for $t > 0$) or return $\mu_t$ (for $t = 0$). Final images are clamped to $[-1, 1]$ and mapped to $[0, 1]$ for visualization as a 4×4 grid.

### Results

The outputs exhibit the characteristic "cloud-like" blurriness of an undertrained diffusion model — global structure begins to emerge but fine detail does not resolve. The full pipeline would converge given additional compute.

<p align="center">
  <img src="assets/self_tried_training_result.png" width="380">
  <br>
  <em>4×4 sample grid — custom DDPM, CelebA 64×64, undertrained.</em>
</p>

```bash
python train.py   # train
python sample.py  # sample
```

---

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `Pillow`, `matplotlib`, `tqdm`, `imageio`, `lpips`

Place pretrained weights in `repaint_simplified/pretrained_weights/` before running inference.

---

## References

- Lugmayr et al., *RePaint: Inpainting using Denoising Diffusion Probabilistic Models* — https://github.com/andreas128/RePaint
- Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis* — https://github.com/openai/guided-diffusion
- Ho et al., *Denoising Diffusion Probabilistic Models* — https://github.com/hojonathanho/diffusion

---

## Author

Independent study of diffusion-based image inpainting using PyTorch.
