import torch
from torch.utils.data import IterableDataset, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class PeptideDataset(Dataset):
    def __init__(self, dataset_path, train=True, log10=True, remove_gsg=False,
                 train_test_ratio=0.9, amino_acids='ADEFGHKLNPQRSVWY', random_seed=42):
        """
        Args:
            dataset_path (str): Path to the csv file containing peptide sequences and labels.
            train (bool): Whether to use the train or test set.
            log10 (bool): Whether to apply log10 to the labels.
            remove_gsg (bool): Whether to remove the GSG linker from the peptide sequences.
            train_test_ratio (float): Ratio of the dataset to use for training.
            amino_acids (str): Allowed amino acids on the peptide array.
        """
        self.dataset = pd.read_csv(dataset_path, header=None)
        self.dataset = self.dataset.groupby(0, as_index=False).mean()
        if log10:
            self.dataset.loc[:, 1:] = np.log10(self.dataset.loc[:, 1:] + 100)
        if remove_gsg:
            self.dataset[0] = self.dataset[0].str[:-3]
        train_data, test_data = train_test_split(self.dataset, train_size=train_test_ratio, random_state=random_seed, shuffle=True)
        if train == True:
            self.dataset = train_data.reset_index(drop=True)
        else:
            self.dataset = test_data.reset_index(drop=True)

        self.max_length = self.dataset[0].str.len().max()
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
            tuple: Sequence and label of the sample ((sequence_dim, amino_acid_dim), (label_dim,)).
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
