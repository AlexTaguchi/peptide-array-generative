import matplotlib.pyplot as plt

def plot_segmentation_maps(images, labels, filename, num_cols=8):
    """Save segmentation maps as a grid.
    
    Args:
        images (torch.Tensor): Tensor containing segmentation maps of shape [N, H, W, K].
        labels (torch.Tensor): Tensor containing labels of shape [N, C].
        filename (str): Path to save the image.
        num_cols (int): Number of columns in the image grid. Defaults to 8.
    """
    # Convert to numpy and get segmentation by taking argmax of last dimension
    images = images.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    segmentation = images.argmax(axis=-1)
    
    # Calculate number of rows needed
    num_rows = (images.shape[0] + num_cols - 1) // num_cols
    
    # Create figure and axes
    _, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    
    # Make axes indexable if there's only one row
    if num_rows == 1:
        axes = axes[None, :]
    
    # Plot each segmentation map
    for i in range(images.shape[0]):
        row = i // num_cols
        col = i % num_cols
        
        # Plot segmentation map with a distinct color for each class
        im = axes[row, col].imshow(segmentation[i], cmap='tab20', interpolation='nearest')
        axes[row, col].axis("off")
        
        # Add label as title
        label = labels.argmax(axis=1)[i]
        axes[row, col].set_title(label)
    
    # Hide empty subplots
    for i in range(images.shape[0], num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()