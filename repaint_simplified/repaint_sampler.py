import torch
from tqdm import tqdm


def get_schedule(t_T=250, jump_len=10, jump_n_sample=10):
    """
    The RePaint resampling timestep schedule.
    Based on the pseudocode from the paper.
    """

    jumps = {}
    for j in range(0, t_T - jump_len, jump_len):
        jumps[j] = jump_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1

            for _ in range(jump_len):
                t = t + 1
                ts.append(t)

    ts.append(-1)
    return ts


def repaint_sample(
    model,
    diffusion,
    gt,
    mask,
    device,
    num_steps=250,
    jump_length=10,
    jump_n_sample=10,
    return_frames=False
):

    B, C, H, W = gt.shape

    #y = torch.zeros(B, dtype=torch.long, device=device)
    #y = None
    #y = torch.randint(0, 1000, (B,), device=device)

    x = torch.randn(B, C, H, W, device=device)

 
    # Generate RePaint timestep schedule
    T = diffusion.num_timesteps
    ts = get_schedule(T, jump_length, jump_n_sample)

    is_ddpm = (jump_length == 1 and jump_n_sample == 1)

    frames = [] if return_frames else None
 
    # Diffusion sampling
    for i in tqdm(range(len(ts) - 1), desc="RePaint Sampling"):

        t = ts[i]
        t_next = ts[i + 1]

        # stop condition
        if t_next < 0:
            break

        t_tensor = torch.tensor([t] * B, device=device)


        # PURE DDPM MODE
        if is_ddpm:
            # Reverse diffusion step only
            out = diffusion.p_sample(model, x, t_tensor, model_kwargs=None)
            x = out["sample"]

            # enforce known region
            noisy_gt = diffusion.q_sample(gt, t_tensor)
            x = mask * noisy_gt + (1 - mask) * x

        # REPAINT MODE
        else:
            # Reverse diffusion
            if t_next < t:
                out = diffusion.p_sample(model, x, t_tensor, model_kwargs=None)
                x = out["sample"]

                noisy_gt = diffusion.q_sample(gt, t_tensor)
                x = mask * noisy_gt + (1 - mask) * x

            # Forward diffusion (resampling)
            else:
                beta = torch.tensor(diffusion.betas[t_next], device=x.device)
                noise = torch.randn_like(x)
                x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise

        if return_frames and i % 100 == 0:
            frames.append(x.clone())
    
    if return_frames:
        return x, ts, frames
    else:
        return x, ts