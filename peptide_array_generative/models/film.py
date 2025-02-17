import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, condition_dim, feature_dim):
        """FiLM conditioning layer using a simple linear transformation."""
        super(FiLM, self).__init__()
        self.gamma_layer = nn.Linear(condition_dim, feature_dim)
        self.beta_layer = nn.Linear(condition_dim, feature_dim)

    def forward(self, condition):
        gamma = self.gamma_layer(condition)  # Scaling factors
        beta = self.beta_layer(condition)    # Shifting factors
        return gamma, beta

class FiLMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, condition_dim):
        """Simple feed-forward network with FiLM conditioning."""
        super(FiLMNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.film = FiLM(condition_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t, y):
        x_shape = x.shape
        h = self.fc1(x.view(x_shape[0], -1))
        gamma, beta = self.film(torch.cat((y, t[:, None]), dim=1))
        h = gamma * h + beta
        h = torch.relu(h)
        h = self.fc2(h)
        h = h.view(*x_shape)
        h = torch.softmax(h, dim=-1)
        return h