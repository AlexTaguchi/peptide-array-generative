import sys
sys.path.append('..')

# Import modules
from peptide_array_generative.datasets.mnist import MNISTCategorical
from peptide_array_generative.models.unet import UNet
from peptide_array_generative.schedules import CosineSchedule
from peptide_array_generative.training.diffusion import MultinomialDiffusion
from peptide_array_generative.utils import plot_segmentation_maps
from torch.utils.data import DataLoader

# Set data loader
data_loader = DataLoader(
    MNISTCategorical(train=False),
    batch_size=16,
    shuffle=True
)
inputs, labels = next(iter(data_loader))
plot_segmentation_maps(inputs, labels, 'results/mnist_categorical.png')

# Set neural network
neural_network = UNet(
    channels=2,
    time_embedding_dim=256,
    conditional_dim=10
)

# Set noise schedule
noise_schedule = CosineSchedule(num_steps=100)

# Train model
MultinomialDiffusion(
    data_loader=data_loader,
    neural_network=neural_network,
    noise_schedule=noise_schedule
).train()
