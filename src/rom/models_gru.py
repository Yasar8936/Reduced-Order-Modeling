import torch.nn as nn


class LatentGRU(nn.Module):
    """
    Input:  latent[t]  (B, latent_dim)
    Output: latent[t+1] (B, latent_dim)
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (B, 1, latent_dim)  one-step input
        out, _ = self.gru(x)    # (B, 1, hidden_dim)
        y = self.fc(out[:, -1, :])  # (B, latent_dim)
        return y
