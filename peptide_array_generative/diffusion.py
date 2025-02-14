# Import modules
from peptide_array_generative.models import ConditionedFFNN
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Initialize logger
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class MultinomialDiffusion(nn.Module):
    def __init__(self, data_loader, num_steps, num_classes=20, device='cuda', schedule="linear"):
        super().__init__()
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.device = device
        self.beta = self.build_noise_schedule(schedule).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Initialize U-Net model
        self.model = ConditionedFFNN(
            input_dim=2 * 28 * 28,
            hidden_dim=512,
            output_dim=2 * 28 * 28,
            condition_dim=10,
        ).to(device)
    
    @staticmethod
    def sample(distribution):
        """
        Sample from a multinomial distribution.

        Args:
            distribution: The multinomial distribution, (N, ..., K).

        Returns:
            torch.Tensor: The sampled values, (N, ..., K).
        """
        size = distribution.size()
        distribution = distribution.view(-1, size[-1]) + 1e-8
        x = torch.multinomial(distribution, 1)
        x = F.one_hot(x, num_classes=size[-1]).view(size).float()
        return x
       
    def build_noise_schedule(self, schedule):
        """Build a noise schedule based on the chosen method.

        Args:
            schedule: The noise schedule, (N,).

        Raises:
            ValueError: Unknown schedule.

        Returns:
            torch.Tensor: The noise schedule, (N,).
        """
        if schedule == "linear":
            return torch.linspace(0.0001, 0.02, self.num_steps)
        elif schedule == "cosine":
            s = 0.008
            t = torch.linspace(0, self.num_steps, self.num_steps) / self.num_steps
            return 1 - (torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2)
        elif schedule == "exponential":
            return torch.exp(torch.linspace(torch.log(torch.tensor(0.0001)), torch.log(torch.tensor(0.02)), self.num_steps))
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def calculate_posterior(self, x_t, x_0, t):
        """Calculate the posterior probability at step t-1.

        Args:
            x_t:    Noised sample x_t, (N, ..., K).
            x_0:    Actual or predicted x_0, (N, ..., K).
            t:      Diffusion step index, (N,).
            
        Returns:
            theta:  Posterior probability at step t-1, (N, ..., K).
        """
        alpha = self.alpha[t].view(-1, *([1] * (len(x_0.shape) - 1)))
        alpha_bar = self.alpha_bar[t].view(-1, *([1] * (len(x_0.shape) - 1)))
        theta_x_t = (alpha * x_t) + ((1 - alpha) / self.num_classes)
        theta_x_0 = (alpha_bar * x_0) + ((1 - alpha_bar) / self.num_classes)
        theta = theta_x_t * theta_x_0
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta
    
    def generate(self, x_t, c):
        """Generate samples from the model.

        Args:
            x_t (torch.Tensor): Noised sample x_t, (N, ..., K).
            c (torch.Tensor): Conditioning variable, (N, ..., C).

        Returns:
            samples (torch.Tensor): Generated samples, (N, ..., K).
        """
        # Switch model into evaluation mode
        self.model.eval()
        with torch.no_grad():

            # Iteratively denoise from t=T to t=0
            for t in tqdm(reversed(range(1, self.num_steps)), position=0):
                
                # Get timestep tensor
                timesteps = torch.ones(c.shape[0], device=self.device).long() * t
                
                # Get model prediction of x_0
                x_0_pred = self.model(x_t, c, timesteps)

                # Calculate posterior distribution
                q_posterior = self.calculate_posterior(x_t, x_0_pred, timesteps)
                
                # Sample from posterior to get x_{t-1}
                x_t = self.sample(q_posterior)

        # Switch model back into training mode
        self.model.train()

        return x_0_pred
    
    def noise(self, x_0, t):
        """Add multinomial noise to x_0.

        Args:
            x_0: The input tensor (N, ..., K).
            t: The diffusion step index.

        Returns:
            q_t:    Noised probability distribution q(x_t | x_0), (N, ..., K).
            x_t:    Noised sample x_t, (N, ..., K).
        """
        alpha_bar = self.alpha_bar[t].view(-1, *([1] * (len(x_0.shape) - 1)))
        q_t = (alpha_bar * x_0) + ((1 - alpha_bar) / self.num_classes)
        x_t = self.sample(q_t)
        return q_t, x_t
    
    def denoise(self, x_t, x_0_pred, t):
        """Remove multinomial noise from x_t.

        Args:
            x_t:        Noised sample x_t, (N, ..., K).
            x_0_pred:   Predicted x_0, (N, ..., K).
            t:          Diffusion step index, (N,).

        Returns:
            q_t_1:      Denoised probability distribution q(x_{t-1} | x_t, x_0), (N, ..., K).
            x_t_1:      Denoised sample x_{t-1}, (N, ..., K).
        """
        q_t_1 = self.calculate_posterior(x_t, x_0_pred, t=t)
        x_t_1 = self.sample(q_t_1)
        return q_t_1, x_t_1

    def train(self, epochs=500, learning_rate=3e-4):
        # Set optimizer and CrossEntropyLoss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.KLDivLoss(reduction='batchmean')

        for epoch in range(epochs + 1):
            logging.info(f'Epoch {epoch}')
            progress_bar = tqdm(self.data_loader)

            for data, labels in progress_bar:

                # Predict noise in data
                x_0 = data.to(self.device)
                c = labels.to(self.device).float()
                t = torch.randint(low=1, high=self.num_steps, size=(data.shape[0],)).to(self.device)
                _, x_t = self.noise(x_0, t)
                x_0_pred = self.model(x_t, c, t)

                # Calculate true posterior q(x_{t-1} | x_t, x_0)
                q_posterior = self.calculate_posterior(x_t, x_0, t)
                
                # Calculate predicted posterior p(x_{t-1} | x_t)
                p_posterior = self.calculate_posterior(x_t, x_0_pred, t)
                
                # Compute KL divergence loss
                loss = criterion(p_posterior.log(), q_posterior)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Report loss
                progress_bar.set_postfix(loss=loss.item())

            # Save samples and model checkpoints at regular intervals
            epoch_id = str(epoch).zfill(len(str(epochs)))
            if epoch_id[-1] == '0':
                # Sample from model
                with torch.no_grad():
                    x_t = (torch.ones_like(x_0) / self.num_classes)[:10].to(self.device)
                    c = torch.eye(10).to(self.device)
                    samples = self.generate(x_t, c)
                
                # Plot samples
                rows = math.ceil(samples.shape[0] / 8)
                fig, axes = plt.subplots(rows, 8, figsize=(8, rows))
                
                # Get number of classes from last dimension
                num_classes = samples.shape[-1]

                # Get colormap with enough colors for all classes
                cmap = plt.cm.get_cmap('tab20')

                for i in range(samples.shape[0]):
                    # Create visualization using all feature channels
                    img = samples[i].cpu().numpy()  # Shape: (32, 32, num_classes)
                    
                    # Create weighted color map using all channels
                    colored_map = np.zeros((img.shape[0], img.shape[1], 4))
                    for j in range(num_classes):
                        color = np.array(cmap(j/num_classes))
                        colored_map += img[:,:,j][...,None] * color
                    
                    # Plot with automatic colormap
                    axes[i//8, i%8].imshow(colored_map)
                    axes[i//8, i%8].axis("off")

                plt.tight_layout()
                plt.savefig(f'results/samples_{epoch_id}.png')
                plt.close(fig)
    
if __name__ == '__main__':
    pass
