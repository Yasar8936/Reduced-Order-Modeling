import torch
import torch.nn as nn


class Simple3DAutoEncoder(nn.Module):


    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # Encoder: reduce spatial size, increase channels
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1),  # -> (16, 16, 16, 2)
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, 8, 8, 1)
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), # -> (64, 8, 8, 1)
            nn.ReLU(inplace=True),
        )

        # We need to know the flattened size after encoder.
        # For your default cutout (z=32,y=32,x=4):
        # After convs: (64, 8, 8, 1) -> flat = 64*8*8*1 = 4096
        self.flat_dim = 64 * 8 * 8 * 1

        self.to_latent = nn.Linear(self.flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.flat_dim)

        # Decoder: upsample back to original size
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 16,16,2)
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),  # -> (16, 32,32,4)
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1)             # -> (3, 32,32,4)
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)
        z = self.to_latent(h)
        return z

    def decode(self, z):
        h = self.from_latent(z)
        h = h.view(z.size(0), 64, 8, 8, 1)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
