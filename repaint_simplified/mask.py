import torch
import numpy as np
from PIL import Image


def load_mask(path, image_size):
    mask = Image.open(path).convert("L")
    mask = mask.resize((image_size, image_size))

    mask = np.array(mask) / 255.0
    mask = (mask > 0.5).astype(np.float32)

    mask = torch.from_numpy(mask)[None, None, :, :]

    return mask