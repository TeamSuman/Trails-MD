
import torch
import torch.nn as nn


class TVAEBottleneckEncoder(nn.Module):
    """Bottleneck TVAE encoder with configurable hidden dimensions and dropout."""

    def __init__(self, input_dim: int, latent_dim: int = 2,
                 hidden_dims: list[int] | None = None, dropout_rate: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # Latent outputs (mean and log-variance)
        self.mean_layer = nn.Linear(curr_dim, latent_dim)
        self.logvar_layer = nn.Linear(curr_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mean_layer(h), self.logvar_layer(h)

class TVAEBottleneckDecoder(nn.Module):
    """Bottleneck TVAE decoder to reconstruct the original features from the latent space."""

    def __init__(self, latent_dim: int, output_dim: int,
                 hidden_dims: list[int] | None = None, dropout_rate: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [128, 256]
        layers = []
        curr_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(curr_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.backbone(z)
        return self.output_layer(h)
