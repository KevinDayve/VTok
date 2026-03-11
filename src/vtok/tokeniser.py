import torch
import torch.nn as nn

from .config import VTokConfig
from .feature_extractor import VGGFeatureExtractor, ClipFeatureExtractor
from .spatial_encoder import SpatialEncoder
from .motion_encoder import MotionEncoder

class VTokeniser(nn.Module):
    """
    Complete video tokeniser.
    Input:
        (B, T, 3, H, W)
    Outputs:
        (B, S + T_motion, dim_v)
    """
    
    def __init__(self, config: VTokConfig) -> None:
        super().__init__()
        self.config = config

        if config.backbone == "clip":
            self.feature_extractor = ClipFeatureExtractor(freeze=config.freeze_backbone)
            feature_dim = 1024
        elif config.backbone == "vgg19":
            self.feature_extractor = VGGFeatureExtractor(layer_index=config.vgg_layer_index, freeze=config.freeze_backbone)
            feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")

        self.spatialEncoder = SpatialEncoder(
            grid_size=config.spatial_grid_size,
            feat_dim=feature_dim,
            token_dim=config.token_dim
        )
        self.motionEncoder = MotionEncoder(
            feature_dim=feature_dim,
            token_dim=config.token_dim,
        )
    
    def forward(self, video: torch.Tensor, key_frame_index: int | None = None) -> torch.Tensor:
        """
        Args:
            video:         (B, T, 3, H, W)
            key_frame_idx: Override for key frame selection.

        Returns:
            (B, S + T_motion, d_v)
        """
        B, T, C, H, W = video.shape
        key_frame_index = key_frame_index if key_frame_index is not None else self.config.key_frame_index

        all_features = self.feature_extractor(video.reshape(B * T, C, H, W))
        feat_c, feat_h, feat_w = all_features.shape[1:]
        all_features = all_features.reshape(B, T, feat_c, feat_h, feat_w)

        key_features = all_features[:, key_frame_index]
        spatial_tokens = self.spatialEncoder(key_features)

        motions_tokens = []
        temporal_indices = list(range(0, T, self.config.temporal_stride))
        if key_frame_index in temporal_indices:
            temporal_indices.remove(key_frame_index)

        for t in temporal_indices:
            frame_features = all_features[:, t]
            motion_token = self.motionEncoder(frame_features, key_features)
            motions_tokens.append(motion_token)

        if motions_tokens:
            motions_tokens = torch.stack(motions_tokens, dim = 1)
            tokens = torch.cat([spatial_tokens, motions_tokens], dim = 1)
        else:
            tokens = spatial_tokens
        
        return tokens

class VTokTokeniser(VTokeniser):
    pass