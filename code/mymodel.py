"""Variational autoencoder (VAE).

TODO: complete the class `VAE`.

You are free to design the architecture (type and number of layers, their
sizes, the latent dimension), but keep the interface described in the
docstrings: the notebook and the other files rely on it.

Notation
    B      : number of images in a batch
    x      : images,                   shape (B, 1, 32, 32), values in [0, 1]
    z      : latent variables,         shape (B, latent_dim)
    mu     : mean of q(z|x),           shape (B, latent_dim)
    logvar : log-variance of q(z|x),   shape (B, latent_dim)
    x_hat  : decoded images,           shape (B, 1, 32, 32), values in [0, 1]
"""

"""Variational Autoencoder (VAE) using CNN."""

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Dimension of the latent variable z
        self.latent_dim = 16

        # Encoder: (B, 1, 32, 32) -> (B, 128, 4, 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten()
        )

        # Mean and log-variance of q(z|x)
        self.fc_mu = nn.Linear(128 * 4 * 4, self.latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, self.latent_dim)

        # Latent vector -> decoder feature map
        self.decoder_input = nn.Linear(
            self.latent_dim, 128 * 4 * 4
        )

        # Decoder: (B, 128, 4, 4) -> (B, 1, 32, 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid()
        )

    def encode(self, x):
        # Extract image features
        h = self.encoder(x)

        # Parameters of the latent distribution
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Standard deviation
        std = torch.exp(0.5 * logvar)

        # Random noise from N(0, I)
        eps = torch.randn_like(std)

        # Reparameterization trick
        z = mu + eps * std

        return z

    def decode(self, z):
        # Convert latent vector into feature map
        h = self.decoder_input(z)
        h = h.view(-1, 128, 4, 4)

        # Reconstruct the image
        x_hat = self.decoder(h)

        return x_hat

    def forward(self, x):
        # Encode input image
        mu, logvar = self.encode(x)

        # Sample latent vector
        z = self.reparameterize(mu, logvar)

        # Decode latent vector
        x_hat = self.decode(z)

        return x_hat, mu, logvar
