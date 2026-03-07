from torchvision import datasets

datasets.CelebA(
    root="./data",
    split="train",
    download=True
)