import numpy as np
import torch
import torch.nn as nn

class CosineSchedule(nn.Module):

    def __init__(self, num_steps=100, s=0.01):
        """Initialize the cosine schedule.

        Args:
            num_steps (int, optional): Number of steps in the schedule. Defaults to 100.
            s (float, optional): Offset for the cosine schedule. Defaults to 0.01.
        """
        super().__init__()
        self.num_steps = num_steps
        self.s = s

        # Calculate alpha bars
        T = num_steps
        t = torch.arange(0, num_steps + 1, dtype=torch.float)
        f_t = torch.cos((np.pi / 2) * ((t/T) + s) / (1 + s)) ** 2
        alpha_bars = f_t / f_t[0]

        # Calculate betas
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        # Calculate sigmas
        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas', sigmas)

class LinearSchedule(nn.Module):

    def __init__(self, num_steps=100, beta_min=0.0001, beta_max=0.2):
        """Initialize the linear schedule.

        Args:
            num_steps (int, optional): Number of steps in the schedule. Defaults to 100.
            beta_min (float, optional): Minimum noise variance. Defaults to 0.0001.
            beta_max (float, optional): Maximum noise variance. Defaults to 0.2.
        """
        super().__init__()
        self.num_steps = num_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

        # Calculate betas
        betas = torch.linspace(beta_min, beta_max, num_steps + 1)

        # Calculate alphas
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Calculate sigmas
        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas', sigmas)
