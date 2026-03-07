# Diffusion Models From Scratch — DDPM Implementation

This repository contains a **PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM)** trained on the **CelebA 64×64 dataset**.
The goal of this project is to understand diffusion-based generative modeling by implementing the architecture and training pipeline from scratch.

This project serves as the **first phase of experimentation**, focusing on building and training a diffusion model without relying on pretrained architectures.

---

## Project Motivation

Diffusion models have recently become one of the most powerful approaches for generative modeling in computer vision. Instead of generating data directly, they learn to **reverse a gradual noise corruption process**.

This repository explores:

* implementing a **DDPM training pipeline from scratch**
* understanding the **forward and reverse diffusion processes**
* experimenting with **UNet-based noise prediction**
* training a diffusion model on a real image dataset

Further work will explore **diffusion-based inpainting using the RePaint algorithm**.

---

## Implemented Components

The following components are implemented from scratch.

### 1. Diffusion Process

The forward diffusion process gradually corrupts an image with Gaussian noise:

[
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
]

where

* (x_0) = original image
* (x_t) = noisy image at timestep (t)
* (\epsilon) = Gaussian noise

A **cosine noise schedule** is used to control the noise variance.

---

### 2. UNet Noise Predictor

The model learns to predict the noise added to an image.

Architecture features:

* UNet encoder–decoder backbone
* sinusoidal timestep embeddings
* residual blocks with GroupNorm
* self-attention block in the bottleneck
* skip connections between encoder and decoder

The network predicts:

[
\epsilon_\theta(x_t, t)
]

which is trained using **MSE loss** against the true noise.

---

### 3. Training Pipeline

Training procedure:

1. Sample an image (x_0) from CelebA
2. Randomly sample timestep (t)
3. Add noise using the forward diffusion equation
4. Predict noise using the UNet
5. Minimize

[
L = ||\epsilon - \epsilon_\theta(x_t, t)||^2
]

The model learns to **denoise images step-by-step**, enabling generation from pure noise.

---

## Repository Structure

```
.
├── diffusion/
│   ├── gaussian_diffusion.py
│   └── scheduler.py
│
├── models/
│   └── unet.py
│
├── utils/
│   └── dataset.py
│
├── train.py
├── sample.py
├── celeba_download.py
│
├── data/            # placeholder for dataset
├── checkpoints/     # placeholder for model checkpoints
│
└── assets/
    ├── training_loss.png
    └── ddpm_samples.png
```

---

## Dataset

Training was performed using the **CelebA dataset**, resized to **64×64 resolution**.

To download the dataset:

```
python celeba_download.py
```

Images are normalized to the range **[-1, 1]** before training.

---

## Training

Install dependencies:

```
pip install -r requirements.txt
```

Run training:

```
python train.py
```

The model uses:

* cosine noise schedule
* Adam optimizer
* gradient accumulation
* exponential moving average (EMA) weights

---

## Sampling

After training, images can be generated using:

```
python sample.py
```

The sampling process performs **reverse diffusion**, starting from random Gaussian noise and iteratively denoising it to produce an image.

---

## Results

Due to limited compute resources, the model was trained for a relatively small number of epochs.
While the generated samples are not fully realistic, they demonstrate the **complete diffusion training and sampling pipeline**.

Example outputs and training curves are provided in the `assets/` directory.

---

## Next Phase

The next stage of this project explores **diffusion-based image inpainting using the RePaint algorithm**, which modifies the diffusion sampling process to condition on known image regions.

This will involve:

* integrating pretrained diffusion models
* implementing RePaint resampling
* generating inpainted images from masked inputs

---

## Related Work

This project builds on the concepts introduced in diffusion-based generative modeling literature, including:

* *Denoising Diffusion Probabilistic Models*
* *Improved DDPM*
* *RePaint: Inpainting using Denoising Diffusion Models*

---

## Author

Independent exploration of diffusion-based generative modeling and image synthesis using PyTorch.
