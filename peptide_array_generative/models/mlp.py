import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        """Simple feed-forward network with FiLM conditioning."""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x.view(x.shape[0], -1))
        x = torch.relu(x)
        x = self.fc2(x)
        return x
