import os
import logging
import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path

from .swin_stream import SwinStream
from .vit_stream import ViTStream
from .cross_attention import CrossAttentionBridge
from .temporal_lstm import TemporalLSTM

logger = logging.getLogger(__name__)


class HSTFe(nn.Module):
    """
    Hybrid Swin-ViT Temporal Fusion Encoder.
    Composes: SwinStream + ViTStream + CrossAttentionBridge + TemporalLSTM
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        device_str = config.get('device', 'cpu')
        self.device = torch.device(
            device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu'
        )

        self.swin = SwinStream(self.device)
        self.vit = ViTStream(self.device)
        self.cross_attn = CrossAttentionBridge(
            swin_dim=config.get('swin_dim', 768),
            vit_dim=config.get('vit_dim', 512),
            num_heads=config.get('cross_attention_heads', 4),
            output_dim=config.get('cross_attention_output_dim', 1024)
        )
        self.temporal = TemporalLSTM(
            input_dim=config.get('cross_attention_output_dim', 1024),
            hidden_dim=config.get('lstm_hidden_dim', 256),
            num_layers=config.get('lstm_num_layers', 2)
        )

        self.cross_attn.to(self.device)
        self.temporal.to(self.device)

        if self.device.type == 'cuda' and config.get('use_fp16', True):
            self.cross_attn.half()
            self.temporal.half()

    def forward(self, hand_crop: torch.Tensor, full_frame: torch.Tensor,
                temporal_buffer: list) -> Tuple[torch.Tensor, float]:
        """
        hand_crop: [B, 3, 224, 224]
        full_frame: [B, 3, 224, 224]
        temporal_buffer: list of previous feature tensors (unused here, handled by buffer_manager)
        returns: (feature_vector [B, 512], confidence: float)
        """
        try:
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                swin_feat = self.swin(hand_crop)
                vit_feat = self.vit(full_frame)
                fused = self.cross_attn(swin_feat, vit_feat)  # [B, 1024]
                fused_seq = fused.unsqueeze(1)                 # [B, 1, 1024]
                output = self.temporal(fused_seq)              # [B, 512]

            confidence = float(torch.sigmoid(output.abs().mean()).item())
            return output, confidence

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error("CUDA OOM in HSTFe.forward — falling back to CPU for this frame")
            self.to('cpu')
            hand_crop = hand_crop.cpu().float()
            full_frame = full_frame.cpu().float()
            swin_feat = self.swin(hand_crop)
            vit_feat = self.vit(full_frame)
            fused = self.cross_attn(swin_feat, vit_feat)
            output = self.temporal(fused.unsqueeze(1))
            confidence = float(torch.sigmoid(output.abs().mean()).item())
            return output, confidence

    @classmethod
    def load_weights(cls, path: str, config: dict) -> 'HSTFe':
        """Load pretrained weights if available, else return untrained model."""
        model = cls(config)
        weights_path = Path(path)
        if weights_path.exists():
            try:
                state = torch.load(str(weights_path), map_location=model.device)
                model.load_state_dict(state, strict=False)
                logger.info(f"Loaded HSTFe weights from {path}")
            except Exception as e:
                logger.warning(f"Failed to load weights from {path}: {e}")
        else:
            logger.warning(
                f"HSTFe weights not found at {path}, using untrained model. "
                "Run training/train_hstfe.py or download pretrained weights."
            )
        return model

    def get_device(self) -> torch.device:
        return self.device

    def reset_temporal_state(self):
        self.temporal.reset_state()
