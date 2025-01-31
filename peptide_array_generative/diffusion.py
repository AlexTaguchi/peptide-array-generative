# Import modules
from peptide_array_generative.unet import UNet
from glob import glob
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm

# Initialize logger
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class Diffusion:
    def __init__(self, data_loader, beta_start=1e-4, beta_end=0.02, device='cuda', noise_steps=1000):

        # Inspect data loader
        self.data_loader = data_loader
        inputs, labels = next(iter(self.data_loader))
        self.inputs_shape = inputs.shape
        # self.inputs_shape = torch.Size([16, 1, 32, 32]) # !!!
        self.labels_shape = labels.shape
        print(*self.inputs_shape, *self.labels_shape)
        print(f'Inputs: {self.inputs_shape}, {inputs.dtype}')
        print(f'Labels: {self.labels_shape}, {labels.dtype}')
        
        # Initialize U-Net model
        self.model = UNet(
            channels=self.inputs_shape[1],
            time_dim=256,
            conditional_dim=self.labels_shape[0],
        ).to(device)

        # Set diffusion parameters
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # Prepare noise schedule
        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def generate(self, batches, conditional=None):

        # Switch model into evaluation mode
        self.model.eval()
        with torch.no_grad():

            # Start with random noise as input
            x = torch.randn((batches, *self.inputs_shape[1:])).to(self.device)

            # Incrementally denoise input
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

                # Set denoising parameters for current time step
                t = (torch.ones(batches) * i).long().to(self.device)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # Partially remove noise from input
                conditional = conditional.to(self.device)
                predicted_noise = self.model(x, t, conditional)
                predicted_noise_scaled = ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                noise_scaled = torch.sqrt(beta) * noise
                x = 1 / torch.sqrt(alpha) * (x - predicted_noise_scaled) + noise_scaled

        # Rescale input
        x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)

        # Switch model back into training mode
        self.model.train()

        return x

    def noise_data(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(self.device)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)

    def train(self, epochs=500, learning_rate=3e-4):
        # Set optimizer and CrossEntropyLoss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(epochs + 1):
            logging.info(f'Epoch {epoch}')
            progress_bar = tqdm(self.data_loader)

            for data, labels in progress_bar:

                # Predict noise in data
                data = data.to(self.device)[:, :1, :, :]
                labels = labels.to(self.device).float()
                t = self.sample_timesteps(data.shape[0])
                x_t, noise = self.noise_data(data, t)
                predicted_noise = self.model(x_t, t, labels) # <!!!> labels[..., None]/10

                # Optimize loss function
                loss = criterion(noise, predicted_noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Report loss
                progress_bar.set_postfix(loss=loss.item())

            # Save samples and model checkpoints at regular intervals
            epoch_id = str(epoch).zfill(len(str(epochs)))
            if epoch_id[-1] == '0':
                samples_generated = self.generate(16, labels[:16])  # <!!!> labels[:16, None]/10
                # predicted_classes = samples_generated[:, 1, :, :]
                # print(predicted_classes)
                # predicted_classes = torch.argmax(samples_generated, dim=1)
                predicted_classes = samples_generated[:, 0, :, :]
                print(predicted_classes)
                cmap = plt.cm.get_cmap('tab20')
                rows = math.ceil(samples_generated.shape[0] / 8)
                fig, axes = plt.subplots(rows, 8, figsize=(8, rows))
                for i in range(16):
                    class_map = predicted_classes[i].cpu().numpy()
                    # colored_map = cmap(class_map / samples_generated.shape[1])
                    axes[i//8, i%8].imshow(class_map[:, :], vmin=0, vmax=1)
                    axes[i//8, i%8].set_title(labels[i].item())
                    axes[i//8, i%8].axis("off")
                plt.tight_layout()
                plt.savefig(f'results/epoch-{epoch_id}.jpg')
                plt.close(fig)
            # samples_grid = torchvision.utils.make_grid(samples_generated)
            # samples_array = samples_grid.permute(1, 2, 0).to('cpu').numpy()
            # image = Image.fromarray(samples_array).convert('RGB')
            # if not os.path.exists('results'):
            #     os.mkdir('results')
            # image.save(f'results/epoch-{epoch_id}.jpg')

    def save_samples_and_model(self, samples_generated, epoch_id):
        samples_array = samples_generated.to('cpu').numpy()  # Convert generated samples to numpy
        # Save generated samples as a numpy array for easier interpretation
        if not os.path.exists('results'):
            os.mkdir('results')
        samples_generated = self.generate(self.classes if self.classes else 16)
                

        # Save model checkpoint
        if not os.path.exists('models'):
            os.mkdir('models')
        model_dimensions = '-'.join([str(x) for x in [self.sequence_dim, self.position_dim, self.amino_acid_dim]])
        torch.save(self.model.state_dict(), f'models/diffusion_unet-{model_dimensions}_epoch-{epoch_id}.pt')
        if len(glob('models/*.pt')) > 1:
            for checkpoint in sorted(glob('models/*.pt'))[:-1]:
                os.remove(checkpoint)

class MultinomialDiffusion:
    def __init__(self, num_classes=2, num_timesteps=100, schedule="cosine"):
        """
        Multinomial diffusion process for categorical data.

        Args:
            num_classes (int): Number of categories per pixel or amino acid (e.g., 2 for MNIST, 20 for peptides).
            num_timesteps (int): Number of diffusion steps.
            schedule (str): Type of noise schedule ("linear", "cosine", "exponential").
        """
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.beta_schedule = self.get_noise_schedule(schedule)
        self.alpha_bar = torch.cumprod(1 - self.beta_schedule, dim=0)  # Accumulated product

    def get_noise_schedule(self, schedule):
        """Generate a beta schedule based on the chosen method."""
        if schedule == "linear":
            return torch.linspace(0.0001, 0.02, self.num_timesteps)
        elif schedule == "cosine":
            s = 0.008
            t = torch.linspace(0, self.num_timesteps, self.num_timesteps) / self.num_timesteps
            return 1 - (torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2)
        elif schedule == "exponential":
            return torch.exp(torch.linspace(torch.log(torch.tensor(0.0001)), torch.log(torch.tensor(0.02)), self.num_timesteps))
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def q_sample(self, x_start, t):
        """
        Forward process: Adds categorical noise using accumulated product alpha_bar.

        Args:
            x_start (Tensor): One-hot encoded original input (batch, feature_dim, num_classes).
            t (Tensor): Diffusion step index.

        Returns:
            Tensor: Noised version of x_start.
        """
        alpha_t = self.alpha_bar.to(t.device)[t].view(-1, 1)  # Ensure correct shape
        noise = (1 - alpha_t) / self.num_classes
        q_xt = alpha_t * x_start + noise
        return torch.softmax(q_xt, dim=-1)  # Return noised one-hot vector

    def p_sample(self, model, x_t, condition, t):
        """
        Reverse process: denoises step-by-step.

        Args:
            model (nn.Module): The trained neural network.
            x_t (Tensor): Noisy input.
            condition (Tensor): Conditioning variable (e.g., digit label or binding value).
            t (int): Current timestep.

        Returns:
            Tensor: Denoised version of x_t.
        """
        logits = model(x_t, condition, t)
        return torch.softmax(logits, dim=-1)

    def sample(self, model, condition, shape, device="cuda"):
        """
        Generates new samples using reverse diffusion.

        Args:
            model: The trained diffusion model.
            condition: One-hot encoded labels for conditional generation.
            shape: Tuple representing (batch_size, flattened_dim, num_classes).
            device: Computation device.

        Returns:
            Tensor of generated images in shape (batch_size, flattened_dim).
        """
        batch_size, flattened_dim, num_classes = shape

        # Start from pure noise
        x_t = torch.randn(batch_size, flattened_dim * num_classes, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, condition, t_tensor)

        return x_t.view(batch_size, flattened_dim, num_classes).argmax(dim=-1)  # Convert back to discrete pixel values

if __name__ == '__main__':
    pass
