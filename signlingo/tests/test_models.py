import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_swin_stream():
    from src.models.swin_stream import SwinStream
    device = torch.device('cpu')
    model = SwinStream(device)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 768), f"Expected (1,768), got {out.shape}"
    print(f"SwinStream: {x.shape} -> {out.shape} PASSED")


def test_vit_stream():
    from src.models.vit_stream import ViTStream
    device = torch.device('cpu')
    model = ViTStream(device)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 512), f"Expected (1,512), got {out.shape}"
    print(f"ViTStream: {x.shape} -> {out.shape} PASSED")


def test_cross_attention():
    from src.models.cross_attention import CrossAttentionBridge
    model = CrossAttentionBridge()
    swin_feat = torch.randn(1, 768)
    vit_feat = torch.randn(1, 512)
    out = model(swin_feat, vit_feat)
    assert out.shape == (1, 1024), f"Expected (1,1024), got {out.shape}"
    print(f"CrossAttentionBridge: (1,768)+(1,512) -> {out.shape} PASSED")


def test_temporal_lstm():
    from src.models.temporal_lstm import TemporalLSTM
    model = TemporalLSTM()
    x = torch.randn(1, 8, 1024)
    out = model(x)
    assert out.shape == (1, 512), f"Expected (1,512), got {out.shape}"
    print(f"TemporalLSTM: {x.shape} -> {out.shape} PASSED")


def test_classifier_head():
    from src.models.classifier_head import ClassifierHead
    model = ClassifierHead()
    x = torch.randn(1, 512)
    logits, probs = model(x)
    assert logits.shape == (1, 536), f"Expected (1,536), got {logits.shape}"
    assert probs.shape == (1, 536)
    assert abs(probs.sum().item() - 1.0) < 1e-4
    label, conf = model.predict(probs)
    assert isinstance(label, str)
    assert 0.0 <= conf <= 1.0
    print(f"ClassifierHead: (1,512) -> logits{logits.shape} PASSED")


def test_fallback_recognizer():
    import numpy as np
    from src.models.fallback_recognizer import FallbackRecognizer
    rec = FallbackRecognizer()
    # All zeros — should return no match
    lm = np.zeros((21, 3))
    label, conf = rec.predict(lm)
    assert isinstance(label, str)
    assert isinstance(conf, float)
    print(f"FallbackRecognizer: zeros -> ('{label}', {conf}) PASSED")


if __name__ == '__main__':
    test_swin_stream()
    test_vit_stream()
    test_cross_attention()
    test_temporal_lstm()
    test_classifier_head()
    test_fallback_recognizer()
    print("\nAll model tests passed.")
