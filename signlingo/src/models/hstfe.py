import logging
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
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
            num_layers=config.get('lstm_num_layers', 2),
            dropout=float(config.get('lstm_dropout', 0.0)),
        )

        self.cross_attn.to(self.device)
        self.temporal.to(self.device)

        if self.device.type == 'cuda' and config.get('use_fp16', True):
            self.cross_attn.half()
            self.temporal.half()

    def encode_spatial(self, hand_crop: torch.Tensor, full_frame: torch.Tensor) -> torch.Tensor:
        """Swin + ViT + cross-attention → fused [B, 1024]."""
        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            swin_feat = self.swin(hand_crop)
            vit_feat = self.vit(full_frame)
            return self.cross_attn(swin_feat, vit_feat)

    def forward_temporal(
        self,
        fused_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        fused_seq: [B, T, 1024] — stack of per-frame fused embeddings.
        Returns (out [B, T, 512], new_hidden).
        """
        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            return self.temporal(fused_seq, hidden)

    def forward(self, hand_crop: torch.Tensor, full_frame: torch.Tensor,
                temporal_buffer: list) -> Tuple[torch.Tensor, float]:
        """
        Training / single-frame path: T=1 temporal slice.

        hand_crop: [B, 3, 224, 224]
        full_frame: [B, 3, 224, 224]
        temporal_buffer: unused (legacy)
        returns: (feature_vector [B, 512], confidence: float)
        """
        try:
            fused = self.encode_spatial(hand_crop, full_frame)
            out, _ = self.forward_temporal(fused.unsqueeze(1), hidden=None)
            output = out[:, -1, :]
            confidence = float(torch.sigmoid(output.abs().mean()).item())
            return output, confidence

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error("CUDA OOM in HSTFe.forward — falling back to CPU for this frame")
            self.to('cpu')
            hand_crop = hand_crop.cpu().float()
            full_frame = full_frame.cpu().float()
            fused = self.encode_spatial(hand_crop, full_frame)
            out, _ = self.forward_temporal(fused.unsqueeze(1), hidden=None)
            output = out[:, -1, :]
            confidence = float(torch.sigmoid(output.abs().mean()).item())
            return output, confidence

    @classmethod
    def load_weights(cls, path: str, config: dict) -> Tuple['HSTFe', Optional[Dict[str, torch.Tensor]]]:
        """
        Load pretrained backbone weights if available, else return an untrained model.

        Supports:
        - Training checkpoints: {model_state, classifier_state?, ...}
        - Alternate wrapper: {state_dict, ...}
        - Raw ``nn.Module`` state_dict at top level
        Returns (model, classifier_state_or_none).
        """
        model = cls(config)
        weights_path = Path(path)
        classifier_state: Optional[Dict[str, torch.Tensor]] = None

        if not weights_path.exists():
            logger.warning(
                f"HSTFe weights not found at {path}, using untrained model. "
                "Run training/train_hstfe.py or download pretrained weights."
            )
            return model, None

        try:
            try:
                state_obj = torch.load(
                    str(weights_path), map_location=model.device, weights_only=True
                )
            except Exception:
                state_obj = torch.load(
                    str(weights_path), map_location=model.device, weights_only=False
                )

            backbone_state = state_obj
            if isinstance(state_obj, dict):
                if 'model_state' in state_obj:
                    backbone_state = state_obj['model_state']
                    raw_cls = state_obj.get('classifier_state')
                    if isinstance(raw_cls, dict):
                        classifier_state = raw_cls
                elif 'state_dict' in state_obj:
                    backbone_state = state_obj['state_dict']

            incompatible = model.load_state_dict(backbone_state, strict=False)
            n_missing = len(incompatible.missing_keys)
            n_unexpected = len(incompatible.unexpected_keys)
            if n_missing:
                logger.warning(f"HSTFe load: {n_missing} missing keys (strict=False)")
            if n_unexpected:
                logger.warning(f"HSTFe load: {n_unexpected} unexpected keys (strict=False)")
            logger.info(f"Loaded HSTFe backbone from {path}")
        except Exception as e:
            logger.warning(f"Failed to load weights from {path}: {e}")
            classifier_state = None

        return model, classifier_state

    def get_device(self) -> torch.device:
        return self.device

    def reset_temporal_state(self):
        self.temporal.reset_state()
