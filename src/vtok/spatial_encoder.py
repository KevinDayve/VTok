import torch
import torch.nn as nn

class SpatialEncoder(nn.Module):
    """
    Converts a key frame into a spatial token.
    """
    def __init__(self, grid_size: int = 4, feat_dim: int = 1024, token_dim: int = 1024) -> None:
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.feat_dim = feat_dim
        self.token_dim = token_dim
        self.proj = None
        if self.feat_dim != self.token_dim:
            self.proj = nn.Linear(self.feat_dim, self.token_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.adaptive_pool(features)
        Batch, dim, g_h, g_w = pooled.shape
        spatial_pooled = pooled.reshape(Batch, g_h*g_w, dim)
        if self.proj is not None:
            spatial_pooled = self.proj(spatial_pooled)
        return spatial_pooled
