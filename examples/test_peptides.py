import sys
sys.path.append('..')

# Import modules
from peptide_array_generative.dataset import PeptideDataset
from peptide_array_generative.diffusion import Diffusion
import torch
from torch.utils.data import DataLoader

# Paths to files
data_file = '../data/test/peptides.txt'
label_file = '../data/test/labels.txt'
max_seq_len = 16

# Create the Dataset and DataLoader
dataset = PeptideDataset(data_file, label_file, max_seq_len)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Train model
device = torch.device("mps")
diffusion = Diffusion(16, 16, 21, conditional_dim=1, device=device)
diffusion.train(dataloader=dataloader)
