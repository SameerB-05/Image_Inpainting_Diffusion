import torch


def make_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    beta = torch.clamp(beta, 0.0001, 0.999)

    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    alpha_cumprod_prev = torch.cat(
        [torch.ones(1), alpha_cumprod[:-1]], dim=0
    )

    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_cumprod": alpha_cumprod,
        "alpha_cumprod_prev": alpha_cumprod_prev,
        "sqrt_acp": torch.sqrt(alpha_cumprod),
        "sqrt_omacp": torch.sqrt(1.0 - alpha_cumprod),
    }