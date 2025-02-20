import sys
sys.path.append('..')

# Import modules
from peptide_array_generative.datasets.peptides import PeptideDataset
from peptide_array_generative.models.film import FiLMNet
from peptide_array_generative.models.mlp import MLP
from peptide_array_generative.schedules import CosineSchedule
from peptide_array_generative.training.regression import RegressionTrainer
from peptide_array_generative.training.diffusion import MultinomialDiffusion
import random
from torch.utils.data import DataLoader

# Set data loaders
random_seed = random.randint(0, 1e9)
data_loader_train = DataLoader(PeptideDataset(
    dataset_path='../data/peptides/FNR.csv',
    train=True,
    remove_gsg=True,
    random_seed=random_seed
), batch_size=32, shuffle=True)
data_loader_test = DataLoader(PeptideDataset(
    dataset_path='../data/peptides/FNR.csv',
    train=False,
    remove_gsg=True,
    random_seed=random_seed
), batch_size=1000, shuffle=True)

# Set neural network
sequences, labels = next(iter(data_loader_train))
neural_network = MLP(
    input_dim=sequences.shape[1] * sequences.shape[2],
    hidden_dim=256,
    output_dim=1
)

# Train model
validation_model = RegressionTrainer(
    data_loader_train=data_loader_train,
    data_loader_test=data_loader_test,
    neural_network=neural_network
)
validation_model.train()

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
    data_loader=data_loader_train,
    neural_network=neural_network,
    noise_schedule=noise_schedule
).train(validation_model=validation_model.model)
