import torch

ckpt = torch.load(r"C:\Users\samee\Documents\GitHub_Repos\Image_Inpainting_Diffusion\repaint_simplified\pretrained_weights\256x256_classifier.pt", map_location="cpu")

# if it is a state_dict
if isinstance(ckpt, dict):
    for k in ckpt.keys():
        print(k)