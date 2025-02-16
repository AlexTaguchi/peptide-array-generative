import sys
sys.path.append('..')

# Import modules
from peptide_array_generative.diffusion import MultinomialDiffusion
from peptide_array_generative.schedules import CosineSchedule
from peptide_array_generative.models import FiLMNet
from peptide_array_generative.dataset import MNISTBinary
from peptide_array_generative.utils import plot_segmentation_maps
from torch.utils.data import DataLoader

# Set data loader
data_loader = DataLoader(
    MNISTBinary(train=False),
    batch_size=16,
    shuffle=True
)
inputs, labels = next(iter(data_loader))
plot_segmentation_maps(inputs, labels, 'results/mnist_binary.png')

# Set neural network
neural_network = FiLMNet(
    input_dim=2 * 28 * 28,
    hidden_dim=512,
    output_dim=2 * 28 * 28,
    condition_dim=10,
)

# Set noise schedule
noise_schedule = CosineSchedule(num_steps=100)

# Train model
MultinomialDiffusion(
    data_loader=data_loader,
    neural_network=neural_network,
    noise_schedule=noise_schedule
).train()
