
"""Training of the VAE: loss function and one epoch of training."""

import torch
import torch.nn.functional as F


def loss_function(x, x_hat, mu, logvar):
    """Compute VAE loss averaged over images in a batch.

    Returns:
        (loss, recon, kl): three scalar tensors.
    """

    batch_size = x.size(0)

    # 1. Reconstruction loss (Binary Cross Entropy)
    recon = F.binary_cross_entropy(
        x_hat,
        x,
        reduction="sum"
    ) / batch_size

    # 2. KL divergence
    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ) / batch_size

    # 3. Total VAE loss
    loss = recon + kl

    return loss, recon, kl


def train_one_epoch(model, optimizer, dataloader, device):
    """Train the VAE for one epoch.

    Returns:
        dict containing average loss, recon, and kl per image.
    """

    # Enable training mode
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_images = 0

    for x in dataloader:

        # Move images to CPU or GPU
        x = x.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        x_hat, mu, logvar = model(x)

        # Calculate loss
        loss, recon, kl = loss_function(
            x, x_hat, mu, logvar
        )

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate metrics weighted by batch size
        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_recon += recon.item() * batch_size
        total_kl += kl.item() * batch_size
        total_images += batch_size

    return {
        "loss": total_loss / total_images,
        "recon": total_recon / total_images,
        "kl": total_kl / total_images
    }
