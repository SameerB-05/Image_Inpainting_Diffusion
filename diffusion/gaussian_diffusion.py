import torch
import torch.nn as nn
from diffusion.scheduler import make_beta_schedule


class DDPM(nn.Module):
    def __init__(self, model, T=1000):
        super().__init__()
        self.model = model
        self.T = T

        # create schedule
        schedule = make_beta_schedule(T)

        # register buffers so they move with .to(device)
        for k, v in schedule.items():
            self.register_buffer(k, v)

    # --------------------------------------------------
    # Forward diffusion (q sample)
    # --------------------------------------------------
    def q_sample(self, x0, t, noise):
        """
        x0: clean image
        t: (B,)
        noise: standard Gaussian noise
        """
        return (
            self.sqrt_acp[t][:, None, None, None] * x0 +
            self.sqrt_omacp[t][:, None, None, None] * noise
        )

    # --------------------------------------------------
    # Training forward pass
    # --------------------------------------------------
    def forward(self, x0):
        b = x0.size(0)

        # sample random timestep for each image
        t = torch.randint(0, self.T, (b,), device=x0.device)

        noise = torch.randn_like(x0)

        # diffuse image
        x_t = self.q_sample(x0, t, noise)

        # predict noise
        eps_pred = self.model(x_t, t)

        return eps_pred, noise

    # --------------------------------------------------
    # Reverse sampling
    # --------------------------------------------------
    @torch.no_grad()
    def sample(self, shape, device):
        """
        shape: (B, C, H, W)
        """
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.T)):
            t_batch = torch.full(
                (shape[0],),
                t,
                device=device,
                dtype=torch.long
            )

            eps = self.model(x, t_batch)

            beta = self.beta[t]
            alpha = self.alpha[t]
            alpha_cumprod = self.alpha_cumprod[t]
            alpha_cumprod_prev = self.alpha_cumprod_prev[t]

            beta = beta.view(1, 1, 1, 1)
            alpha = alpha.view(1, 1, 1, 1)
            alpha_cumprod = alpha_cumprod.view(1, 1, 1, 1)
            alpha_cumprod_prev = alpha_cumprod_prev.view(1, 1, 1, 1)

            # predict x0
            x0 = (x - torch.sqrt(1 - alpha_cumprod) * eps) / torch.sqrt(alpha_cumprod)
            x0 = x0.clamp(-1, 1)

            # compute mean
            mean = (
                torch.sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod) * x0
                + torch.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * x
            )

            if t > 0:
                noise = torch.randn_like(x)
                var = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean

        return x.clamp(-1, 1)