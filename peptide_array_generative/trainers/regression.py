# Import modules
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Initialize logger
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class RegressionTrainer(nn.Module):
    def __init__(self, data_loader_train, data_loader_test, neural_network, device=None):
        """Initialize the RegressionTrainer class.

        Args:
            data_loader_train (torch.utils.data.DataLoader): Data loader for the training dataset.
            data_loader_test (torch.utils.data.DataLoader): Data loader for the test dataset.
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
        self.data_loader_train = data_loader_train
        self.data_loader_test = data_loader_test

        # Move model to device
        self.model = neural_network.to(self.device)

    def train(self, epochs=10, learning_rate=1e-3):
        """Train the model.

        Args:
            epochs (int, optional): Number of epochs to train the model. Defaults to 100.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        """
        # Set optimizer and CrossEntropyLoss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch, (data_test, labels_test) in zip(range(epochs + 1), self.data_loader_test):
            logging.info(f'Epoch {epoch}')
            progress_bar = tqdm(self.data_loader_train)

            for data_train, labels_train in progress_bar:

                # Predict labels
                x = data_train.to(self.device)
                y = labels_train.to(self.device).float()
                y_train_pred = self.model(x)
                
                # Compute MSE loss
                loss = criterion(y_train_pred, y)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Report loss
                progress_bar.set_postfix(loss=loss.item())

            # Save model and validate on test set
            with torch.no_grad():
                self.model.eval()
                y_test_pred = self.model(data_test.to(self.device))
                loss_test = criterion(y_test_pred, labels_test.to(self.device).float()).item()
                logging.info(f'Test loss: {loss_test:.5f}')
                self.model.train()

    
if __name__ == '__main__':
    pass
