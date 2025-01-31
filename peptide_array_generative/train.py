import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from peptide_array_generative.diffusion import MultinomialDiffusion
from peptide_array_generative.models import ConditionedFFNN

def save_images(tensor, labels, filename, num_rows=4, num_cols=4):
    """
    Save generated images as a grid with their respective class labels.
    
    Args:
        tensor (torch.Tensor): Tensor containing generated images.
        labels (torch.Tensor): Corresponding class labels.
        filename (str): Path to save the image.
        num_rows (int): Number of rows in the image grid.
        num_cols (int): Number of columns in the image grid.
    """
    tensor = tensor.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()  # Convert labels to NumPy array
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, 6))

    for i, ax in enumerate(axes.flatten()):
        if i >= tensor.shape[0]:  
            break  
        ax.imshow(tensor[i, :, :], cmap="gray")  # Use first channel
        ax.set_title(f"Class: {int(labels[i])}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train_model(train_loader, device_type=None, epochs=20, lr=1e-3, num_timesteps=100, hidden_dim=512, save_dir="generated_images"):
    os.makedirs(save_dir, exist_ok=True)

    # Device selection
    if device_type is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_type)

    print(f"Using device: {device}")

    # Get dataset info dynamically
    first_batch, first_labels = next(iter(train_loader))
    input_shape = first_batch.shape[1:]  
    num_classes = first_labels.shape[-1]  # ✅ Fixed num_classes!

    flattened_input_dim = torch.prod(torch.tensor(input_shape)).item()
    print(f"Training on dataset with input shape: {input_shape}, num_classes: {num_classes}")

    # Define model & diffusion
    model = ConditionedFFNN(flattened_input_dim, hidden_dim, flattened_input_dim, num_classes).to(device)
    diffusion = MultinomialDiffusion(num_classes=num_classes, num_timesteps=num_timesteps, schedule="cosine")  

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch") as progress_bar:
            for data, labels in progress_bar:
                data, labels = data.to(device), labels.to(device)
                if len(data.shape) > 2:  
                    data = data.view(data.shape[0], -1)

                if len(labels.shape) == 1 and num_classes > 1:  
                    labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)

                t = torch.randint(0, num_timesteps, (data.shape[0],), device=device)
                noisy_data = diffusion.q_sample(data, t)
                logits = model(noisy_data, labels, t)

                # 🔥 Fixed loss computation
                loss = criterion(
                    logits.view(data.shape[0], data.shape[1] // first_batch.shape[-1], first_batch.shape[-1]).permute(0, 2, 1),  # Swap logits dimensions
                    data.view(data.shape[0], -1, first_batch.shape[-1]).argmax(dim=-1)  # Ensure target shape matches logits
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # 🔹 Update tqdm progress bar with loss
                progress_bar.set_postfix(loss=f"{total_loss / len(train_loader):.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

        # 🔥 Generate and save MNIST-like images after each epoch
        model.eval()
        with torch.no_grad():
            batch_size = 16
            shape = (batch_size, flattened_input_dim // first_batch.shape[-1], first_batch.shape[-1])

            # Generate random digit labels
            sample_conditions = torch.randint(0, num_classes, (batch_size,), device=device)
            sample_conditions = torch.nn.functional.one_hot(sample_conditions, num_classes=num_classes).float()

            # Generate samples
            generated_samples = diffusion.sample(model, sample_conditions, shape, device)
            print(f"Generated samples shape: {generated_samples.shape}")

            # Reshape generated samples for visualization
            generated_samples = generated_samples.view(batch_size, 28, 28)

            save_filename = os.path.join(save_dir, f"epoch_{epoch+1}.png")
            save_images(generated_samples, sample_conditions.argmax(dim=-1), save_filename)
        
    print("Training complete.")