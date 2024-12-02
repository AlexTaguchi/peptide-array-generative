import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PeptideDataset(Dataset):
    def __init__(self, data_file, label_file, max_seq_len, amino_acid_vocab='ACDEFGHIKLMNPQRSTVWY-', transform=None):
        """
        Args:
            data_file (str): Path to the file containing comma-separated sets of peptide sequences.
            label_file (str): Path to the file containing float labels.
            max_seq_len (int): Maximum number of amino acids per sequence (for padding/truncation).
            amino_acid_vocab (dict): Mapping from amino acid characters to indices for one-hot encoding.
            transform (callable, optional): Optional transform to apply to the peptide sequences.
        """
        # Load peptide data and labels
        with open(data_file, 'r') as f:
            self.data = [line.strip().split(',') for line in f.readlines()]  # Parse peptide sequences

        with open(label_file, 'r') as f:
            self.labels = [float(line.strip()) for line in f.readlines()]  # Convert labels to floats

        # Ensure data and labels have the same length
        assert len(self.data) == len(self.labels), "Data and labels must have the same number of lines."

        self.max_seq_len = max_seq_len
        self.amino_acid_vocab = {aa: idx for idx, aa in enumerate(amino_acid_vocab)}
        self.amino_acid_dim = len(amino_acid_vocab)
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)

    def one_hot_encode(self, sequence):
        """
        Converts a peptide sequence into a one-hot encoded tensor.
        Args:
            sequence (str): Peptide sequence (e.g., "PEPTIDE").

        Returns:
            Tensor: One-hot encoded tensor of shape [position_dim, amino_acid_dim].
        """
        # Pad or truncate the sequence to max_seq_len
        padded_sequence = sequence[:self.max_seq_len].ljust(self.max_seq_len, '-')  # Use '-' for padding
        one_hot = np.zeros((self.max_seq_len, self.amino_acid_dim), dtype=np.float32)

        for i, amino_acid in enumerate(padded_sequence):
            if amino_acid in self.amino_acid_vocab:  # Only encode valid amino acids
                one_hot[i, self.amino_acid_vocab[amino_acid]] = 1.0

        return torch.tensor(one_hot)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (peptides_tensor, label), where peptides_tensor is a 3D tensor
                of shape [amino_acid_dim, sequence_dim, position_dim], and label is a float.
        """
        peptides = self.data[idx]  # List of peptide sequences
        label = self.labels[idx]  # Corresponding float label

        # Convert all peptide sequences in the set to one-hot encoded tensors
        peptides_tensor = torch.stack([self.one_hot_encode(seq) for seq in peptides])  # Shape: [sequence_dim, position_dim, amino_acid_dim]

        # Transpose to [amino_acid_dim, sequence_dim, position_dim]
        peptides_tensor = peptides_tensor.permute(2, 0, 1)

        # Ensure the label is float32
        label = torch.tensor(label, dtype=torch.float32)

        # Apply transform to the peptides tensor if specified
        if self.transform:
            peptides_tensor = self.transform(peptides_tensor)

        return peptides_tensor, label


# Example usage
if __name__ == "__main__":
    
    # Paths to files
    data_file = 'peptides.txt'  # Replace with your data file
    label_file = 'labels.txt'  # Replace with your label file
    max_seq_len = 10  # Maximum amino acid positions in a sequence

    # Create the dataset
    dataset = PeptideDataset(data_file, label_file, max_seq_len)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Iterate through the DataLoader
    for batch_idx, (peptides, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print("Peptides (tensor shape):", peptides.shape)  # Should be [batch_size, sequence_dim, position_dim, amino_acid_dim]
        print("Labels:", labels)
        break  # Remove break to iterate through all batches