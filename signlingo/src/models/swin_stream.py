import torch
import torch.nn as nn
import timm


class SwinStream(nn.Module):
    """Swin Transformer stream for local spatial hand feature extraction."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True
        )
        # Remove classification head
        backbone.head = nn.Identity()
        self.backbone = backbone

        self.projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU()
        )

        self.to(device)
        if device.type == 'cuda':
            self.half()  # FP16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 224, 224] -> [B, 768]"""
        features = self.backbone(x)
        return self.projection(features)
