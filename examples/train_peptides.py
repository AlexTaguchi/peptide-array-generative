import sys
sys.path.append('..')

# Import modules
from peptide_array_generative.datasets.peptides import PeptideDataset
from peptide_array_generative.models.film import FiLMNet
from peptide_array_generative.schedules import CosineSchedule
from peptide_array_generative.diffusion import MultinomialDiffusion
from torch.utils.data import DataLoader

# Set data loader
data_loader = DataLoader(PeptideDataset(
    dataset_path='../data/peptides/FNR.csv',
    max_length=12,
    amino_acids='ADEFGHKLNPQRSVWY'
), batch_size=4, shuffle=True)
sequences, labels = next(iter(data_loader))

# Set neural network
neural_network = FiLMNet(
    input_dim=sequences.shape[1] * sequences.shape[2],
    hidden_dim=256,
    output_dim=sequences.shape[1] * sequences.shape[2],
    condition_dim=labels.shape[-1]
)

# Set noise schedule
noise_schedule = CosineSchedule(num_steps=100)

# Train model
MultinomialDiffusion(
    data_loader=data_loader,
    neural_network=neural_network,
    noise_schedule=noise_schedule
).train()
