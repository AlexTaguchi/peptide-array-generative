# Import modules
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Initialize logger
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class RegressionTrainer(nn.Module):
    def __init__(self, data_loader, neural_network, device=None):
        """Initialize the RegressionTrainer class.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for the dataset.
            neural_network (torch.nn.Module): Model to train for regression tasks.
            device (str, optional): Device to run the model on. Defaults to None for automatic detection.
        """
        super().__init__()

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Import data loader and determine number of categories
        self.data_loader = data_loader

        # Move model to device
        self.model = neural_network.to(self.device)

    def train(self, epochs=100, learning_rate=1e-3):
        """Train the model.

        Args:
            epochs (int, optional): Number of epochs to train the model. Defaults to 100.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        """
        # Set optimizer and CrossEntropyLoss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(epochs + 1):
            logging.info(f'Epoch {epoch}')
            progress_bar = tqdm(self.data_loader)

            for data, labels in progress_bar:

                # Predict labels
                x = data.to(self.device)
                y = labels.to(self.device).float()
                y_pred = self.model(x)
                
                # Compute MSE loss
                loss = criterion(y_pred, y)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Report loss
                progress_bar.set_postfix(loss=loss.item())

        # # Save model and validate on test set
        # with torch.no_grad():
        #     x_t = (torch.ones_like(x_0) / self.K)[:10].to(self.device)
        #     c = torch.eye(10).to(self.device)

    
if __name__ == '__main__':
    pass
