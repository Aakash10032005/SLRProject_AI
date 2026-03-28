import torch
import torch.nn as nn
import timm


class ViTStream(nn.Module):
    """Vision Transformer stream for global frame context extraction."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        backbone = timm.create_model(
            'vit_small_patch16_224', pretrained=True
        )
        # Remove classification head
        backbone.head = nn.Identity()
        self.backbone = backbone

        self.projection = nn.Sequential(
            nn.Linear(384, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )

        self.to(device)
        if device.type == 'cuda':
            self.half()  # FP16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 224, 224] -> [B, 512]"""
        features = self.backbone(x)
        return self.projection(features)
