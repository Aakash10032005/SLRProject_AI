import torch
import torch.nn as nn


class CrossAttentionBridge(nn.Module):
    """
    Fuses Swin (local) and ViT (global) features via cross-attention.
    Swin features act as query; ViT features act as key and value.
    """

    def __init__(self, swin_dim: int = 768, vit_dim: int = 512,
                 num_heads: int = 4, output_dim: int = 1024):
        super().__init__()
        self.output_dim = output_dim

        self.query_proj = nn.Linear(swin_dim, output_dim)
        self.key_proj = nn.Linear(vit_dim, output_dim)
        self.value_proj = nn.Linear(vit_dim, output_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, swin_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        """
        swin_feat: [B, 768]
        vit_feat:  [B, 512]
        returns:   [B, 1024]
        """
        q = self.query_proj(swin_feat).unsqueeze(1)   # [B, 1, 1024]
        k = self.key_proj(vit_feat).unsqueeze(1)       # [B, 1, 1024]
        v = self.value_proj(vit_feat).unsqueeze(1)     # [B, 1, 1024]

        attn_out, _ = self.attention(q, k, v)          # [B, 1, 1024]
        return attn_out.squeeze(1)                     # [B, 1024]
