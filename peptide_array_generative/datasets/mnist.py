import os
import torch
from torchvision import datasets, transforms


class MNISTCategorical(datasets.MNIST):
    """Custom MNIST dataset for image generation with multinomial diffusion."""

    def __init__(self, train=True):
        super().__init__(
            root=os.path.join(os.path.dirname(__file__), '../data'),
            train=train,
            transform=transforms.Compose([transforms.ToTensor()]),
            download=True
        )

    def __getitem__(self, index):
        data, label = super().__getitem__(index)
        data = data > 0.2
        data = torch.stack([data, ~data], dim=-1).squeeze().long()
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=10)
        return data, label
