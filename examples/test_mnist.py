import sys
sys.path.append('..')

# Import modules
from peptide_array_generative.diffusion import Diffusion
import torch
from torchvision import datasets, transforms

# Set training parameters
batch_size = 16

# Initialize data loader
data_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
    ])),
    batch_size=batch_size,
    shuffle=True
)

# Iterate through the DataLoader
for batch_idx, (images, labels) in enumerate(data_loader):
    print(f"Batch {batch_idx}")
    print("Peptides (tensor shape):", images.shape, images.dtype)
    print("Labels:", labels, labels.dtype)
    break

# Train model
device = torch.device("mps")
diffusion = Diffusion(32, 32, 1, conditional_dim=1, device=device)
diffusion.train(dataloader=data_loader)
