# Import modules
from peptide_array_generative.unet import UNet
from glob import glob
import logging
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm

# Initialize logger
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class Diffusion:
    def __init__(self, sequence_dim, position_dim, amino_acid_dim, conditional_dim=None,
                 beta_start=1e-4, beta_end=0.02, device='cuda', noise_steps=1000):

        # Build U-Net model
        self.sequence_dim = sequence_dim
        self.position_dim = position_dim
        self.amino_acid_dim = amino_acid_dim
        self.conditional_dim = conditional_dim
        self.model = UNet(
            channels=amino_acid_dim,
            time_dim=256,
            conditional_dim=conditional_dim,
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
            x = torch.randn((batches, self.amino_acid_dim, self.sequence_dim, self.position_dim)).to(self.device)

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
        x = (x * 255).type(torch.uint8)

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

    def train(self, dataloader, epochs=500, learning_rate=3e-4):
        # Set optimizer and CrossEntropyLoss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs + 1):
            logging.info(f'Epoch {epoch}')
            progress_bar = tqdm(dataloader)

            for data, labels in progress_bar:

                # Predict noise in data
                data = data.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(data.shape[0])
                x_t, noise = self.noise_data(data, t)
                predicted_noise = self.model(x_t, t, labels[..., None]/10) # <!!!>

                # Optimize loss function
                loss = criterion(noise, predicted_noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Report CEL loss
                progress_bar.set_postfix(CEL=loss.item())

            # Save samples and model checkpoints at regular intervals
            epoch_id = str(epoch).zfill(len(str(epochs)))
            if epoch_id[-1] == '0':
                samples_generated = self.generate(16, labels[:16, None]/10)  # <!!!>
                self.save_samples_and_model(samples_generated, epoch_id, epochs)

    def save_samples_and_model(self, samples_generated, epoch_id, epochs):
        samples_array = samples_generated.to('cpu').numpy()  # Convert generated samples to numpy
        # Save generated samples as a numpy array for easier interpretation
        if not os.path.exists('results'):
            os.mkdir('results')
        np.save(f'results/generated_samples_epoch-{epoch_id}.npy', samples_array)

        # Save model checkpoint
        if not os.path.exists('models'):
            os.mkdir('models')
        model_dimensions = '-'.join([str(x) for x in [self.sequence_dim, self.position_dim, self.amino_acid_dim]])
        torch.save(self.model.state_dict(), f'models/diffusion_unet-{model_dimensions}_epoch-{epoch_id}.pt')
        if len(glob('models/*.pt')) > 1:
            for checkpoint in sorted(glob('models/*.pt'))[:-1]:
                os.remove(checkpoint)


if __name__ == '__main__':
    pass
