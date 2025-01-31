import sys
sys.path.append('..')
from peptide_array_generative.train import train_model
from peptide_array_generative.dataset import get_mnist_dataloader

train_loader, _ = get_mnist_dataloader(batch_size=128)
train_model(train_loader, device_type=None, epochs=20)