# Import modules
import matplotlib.pyplot as plt
import logging
from peptide_array_generative.utils.plotting import plot_segmentation_maps
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Initialize logger
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class MultinomialDiffusion(nn.Module):
    def __init__(self, data_loader, neural_network, noise_schedule, device=None):
        """Initialize the MultinomialDiffusion class.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for the dataset.
            neural_network (torch.nn.Module): Model that predicts x_0 from x_t, t, and c.
            noise_schedule (torch.nn.Module): Noise schedule for forward diffusion process.
            device (str, optional): Device to run the model on. Defaults to None for automatic detection.
        """
        super().__init__()

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Import data loader and determine number of categories
        self.data_loader = data_loader
        self.K = next(iter(data_loader))[0].shape[-1]

        # Move model to device
        self.model = neural_network.to(self.device)

        # Build noise schedule
        self.num_steps = noise_schedule.num_steps
        self.betas = noise_schedule.betas.to(self.device)
        self.alphas = noise_schedule.alphas.to(self.device)
        self.alpha_bars = noise_schedule.alpha_bars.to(self.device)
    
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
        alphas = self.alphas[t].view(-1, *([1] * (len(x_0.shape) - 1)))
        alpha_bars = self.alpha_bars[t].view(-1, *([1] * (len(x_0.shape) - 1)))
        theta_x_t = (alphas * x_t) + ((1 - alphas) / self.K)
        theta_x_0 = (alpha_bars * x_0) + ((1 - alpha_bars) / self.K)
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
                x_0_pred = self.model(x_t, timesteps, c)

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
        alpha_bars = self.alpha_bars[t].view(-1, *([1] * (len(x_0.shape) - 1)))
        q_t = (alpha_bars * x_0) + ((1 - alpha_bars) / self.K)
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

    def train(self, epochs=10, learning_rate=1e-3, validation_model=None):
        """Train the model.

        Args:
            epochs (int, optional): Number of epochs to train the model. Defaults to 100.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            validation_model (torch.nn.Module, optional): Model to validate on. Defaults to None.
        """
        # Set optimizer and CrossEntropyLoss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion_train = nn.KLDivLoss(reduction='batchmean')
        criterion_test = nn.MSELoss()

        for epoch in range(epochs + 1):
            logging.info(f'Epoch {epoch}')
            progress_bar = tqdm(self.data_loader)

            for data, labels in progress_bar:

                # Predict noise in data
                x_0 = data.to(self.device)
                c = labels.to(self.device).float()
                t = torch.randint(low=1, high=self.num_steps, size=(data.shape[0],)).to(self.device)
                _, x_t = self.noise(x_0, t)
                x_0_pred = self.model(x_t, t, c)

                # Calculate true posterior q(x_{t-1} | x_t, x_0)
                q_posterior = self.calculate_posterior(x_t, x_0, t)
                
                # Calculate predicted posterior p(x_{t-1} | x_t)
                p_posterior = self.calculate_posterior(x_t, x_0_pred, t)
                
                # Compute KL divergence loss
                loss = criterion_train(p_posterior.log(), q_posterior)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Report loss
                progress_bar.set_postfix(loss=loss.item())

            # # Save samples at regular intervals
            # epoch_id = str(epoch).zfill(len(str(epochs)))
            # if epoch_id[-1] == '0':
            #     with torch.no_grad():
            #         x_t = (torch.ones_like(x_0) / self.K)[:10].to(self.device)
            #         c = torch.eye(10).to(self.device)
            #         samples = self.generate(x_t, c)
            #     plot_segmentation_maps(samples, c, f'results/epoch_{epoch_id}.png')

            # Validate generated samples with model
            if validation_model is not None:
                with torch.no_grad():
                    x_t = (torch.ones(1000, *x_0.shape[1:]) / self.K).to(self.device)
                    c = (2.5 * torch.rand(1000, c.shape[1]) + 2.5).to(self.device)
                    samples = self.generate(x_t, c)
                    max_indices = torch.argmax(samples, dim=-1)
                    samples_one_hot = torch.nn.functional.one_hot(max_indices, num_classes=samples.shape[-1]).float()
                    c_pred = validation_model(samples_one_hot)
                    loss = criterion_test(c_pred, c)
                    logging.info(f'Validation loss: {loss.item():.5f}')

                    x_plot = c.flatten().cpu().numpy()
                    y_plot = c_pred.flatten().cpu().numpy()
                    plt.scatter(x_plot, y_plot)
                    plt.xlabel('Desired Binding')
                    plt.ylabel('Generated Binding')
                    plt.savefig('peptide_array_diffusion.png', dpi=300)
                    plt.close()
    
if __name__ == '__main__':
    pass
