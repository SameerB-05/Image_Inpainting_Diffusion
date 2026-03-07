import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, root_dir, image_size=64, max_samples=None):
        """
        root_dir (str): Path to img_align_celeba folder
        image_size (int): Final image resolution (64 or 128 eg)
        max_samples (int, optional): Limit number of images (eg 20000)
        """
        self.root_dir = root_dir

        self.image_paths = sorted([
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith(".jpg")
        ])

        # -------- LIMIT DATASET SIZE --------
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])  # scale to [-1,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image