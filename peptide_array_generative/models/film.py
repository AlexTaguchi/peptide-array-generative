import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, condition_dim, feature_dim):
        """FiLM conditioning layer using a simple linear transformation."""
        super(FiLM, self).__init__()
        self.gamma_layer = nn.Linear(condition_dim, feature_dim)
        self.beta_layer = nn.Linear(condition_dim, feature_dim)

    def forward(self, condition):
        gamma = self.gamma_layer(condition)
        beta = self.beta_layer(condition)
        return gamma, beta

class FiLMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, condition_dim, hidden_layers=1):
        """Simple feed-forward network with FiLM conditioning."""
        super(FiLMNet, self).__init__()
        self.film = FiLM(condition_dim + 1, hidden_dim)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x, t, y):
        x_shape = x.shape
        h = self.layers[0](x.view(x_shape[0], -1))
        gamma, beta = self.film(torch.cat((y, t[:, None]), dim=1))
        h = gamma * h + beta
        h = torch.relu(h)
        for layer in self.layers[1:-1]:
            h = torch.relu(layer(h))
        h = self.layers[-1](h)
        h = h.view(*x_shape)
        h = torch.softmax(h, dim=-1)
        return h