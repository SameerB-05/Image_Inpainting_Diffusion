import torch
import torch.nn.functional as F
import lpips


# Initialize once (global)
lpips_model = lpips.LPIPS(net='alex')
lpips_model.eval()


def prepare_tensor(x):
    """
    Ensure tensor is:
    - shape: [1, 3, H, W]
    - range: [-1, 1]
    """
    if isinstance(x, torch.Tensor):
        return x
    else:
        raise ValueError("Input must be torch tensor")


def compute_l1(output, gt, mask):
    """
    L1 error ONLY on unknown region
    """
    unknown = (1 - mask)
    return torch.mean(torch.abs((output - gt) * unknown)).item()


def compute_l2(output, gt, mask):
    """
    L2 error ONLY on unknown region
    """
    unknown = (1 - mask)
    return torch.mean(((output - gt) ** 2) * unknown).item()


def compute_lpips(output, gt, mask, device="cpu"):
    """
    LPIPS on unknown region only
    """

    output = output.to(device)
    gt = gt.to(device)
    mask = mask.to(device)

    unknown = (1 - mask)

    # Apply mask
    output_masked = output * unknown
    gt_masked = gt * unknown

    with torch.no_grad():
        dist = lpips_model(output_masked, gt_masked)

    return dist.item()


def compute_all_metrics(output, gt, mask, device="cpu"):
    """
    Convenience function
    """
    return {
        "l1": compute_l1(output, gt, mask),
        "l2": compute_l2(output, gt, mask),
        "lpips": compute_lpips(output, gt, mask, device)
    }