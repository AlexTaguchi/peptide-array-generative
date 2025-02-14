import sys
sys.path.append('..')

# Import modules
from peptide_array_generative.diffusion import MultinomialDiffusion
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

class ToOneHot:
    def __call__(self, tensor):
        one_hot = torch.zeros((tensor.size(1), tensor.size(2), 2))
        one_hot[:, :, 0] = (tensor > 0.2).float()
        one_hot[:, :, 1] = (tensor <= 0.2).float()
        return one_hot

class LabelToOneHot:
    def __call__(self, label):
        one_hot = torch.zeros(10)
        one_hot[label] = 1.0
        return one_hot

# Set training parameters
batch_size = 16

# Initialize data loader
data_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            ToOneHot()
        ]),
        target_transform=LabelToOneHot()
    ),
    batch_size=batch_size,
    shuffle=True
)

# Plot training data
inputs, labels = next(iter(data_loader))

rows = math.ceil(inputs.shape[0] / 8)
fig, axes = plt.subplots(rows, 8, figsize=(8, rows))

# Get number of classes from last dimension
num_classes = inputs.shape[-1]

# Get colormap with enough colors for all classes
cmap = plt.cm.get_cmap('tab20')

for i in range(batch_size):
    # Create visualization using all feature channels
    img = inputs[i].numpy()  # Shape: (32, 32, num_classes)
    
    # Create weighted color map using all channels
    colored_map = np.zeros((img.shape[0], img.shape[1], 4))
    for j in range(num_classes):
        color = np.array(cmap(j/num_classes))
        colored_map += img[:,:,j][...,None] * color
    
    # Plot with automatic colormap
    axes[i//8, i%8].imshow(colored_map)
    axes[i//8, i%8].set_title(f'Label: {labels.argmax(dim=-1)[i].item()}')
    axes[i//8, i%8].axis("off")

plt.tight_layout()
plt.savefig('results/mnist_train.png')
plt.close(fig)

# Train model
device = torch.device("mps")
diffusion = MultinomialDiffusion(data_loader, device=device, num_steps=50, num_classes=2)
diffusion.train()
