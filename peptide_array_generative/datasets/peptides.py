import torch
from torch.utils.data import IterableDataset, Dataset
import numpy as np
import pandas as pd


class PeptideDataset(Dataset):
    def __init__(self, dataset_path, max_length, amino_acids='ACDEFGHIKLMNPQRSTVWY'):
        """
        Args:
            dataset_path (str): Path to the csv file containing peptide sequences and labels.
            max_length (int): Maximum peptide sequence length.
            amino_acids (str): Allowed amino acids on the peptide array.
        """
        self.dataset = pd.read_csv(dataset_path, header=None)
        self.max_length = max_length
        self.tokens = {aa: i for i, aa in enumerate(amino_acids + '-')}

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.dataset)

    def one_hot_encode(self, sequence):
        """
        Converts a peptide sequence into a one-hot encoded tensor.
        Args:
            sequence (str): Amino acid sequence.

        Returns:
            Tensor: One-hot encoded peptide sequence, (position_dim, amino_acid_dim).
        """
        padded_sequence = sequence[:self.max_length].ljust(self.max_length, '-')
        one_hot_sequence = np.zeros((self.max_length, len(self.tokens)))

        for i, token in enumerate(padded_sequence):
            one_hot_sequence[i, self.tokens[token]] = 1.0

        return torch.tensor(one_hot_sequence, dtype=torch.float32)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (peptides_tensor, label), where peptides_tensor is a 3D tensor
                of shape [amino_acid_dim, sequence_dim, position_dim], and label is a float.
        """
        sequence = self.one_hot_encode(self.dataset.iloc[index, 0])
        label = torch.tensor(self.dataset.iloc[index, 1:].astype(float).values, dtype=torch.float32)

        return sequence, label


class PeptideSyntheticDataset(IterableDataset):
    def __init__(self, model, input_dim, output_dim, batch_size=16, device="cpu"):
        """Infinite dataset that generates data in batches from a model."""
        super().__init__()
        self.model = model.to(device)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        """Continuously generates batches of data."""
        while True:
            # Generate batch of random inputs in one forward pass
            random_inputs = torch.randn(self.batch_size, self.input_dim, device=self.device)
            generated_outputs = self.model(random_inputs)  # Model processes the whole batch

            # Yield each sample in the batch individually
            for i in range(self.batch_size):
                yield random_inputs[i], generated_outputs[i]
