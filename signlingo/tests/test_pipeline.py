import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)


def _make_config():
    return {
        'app': {'log_dir': 'logs', 'transcript_dir': 'transcripts'},
        'camera': {'device_id': 0, 'fps': 30},
        'detection': {
            'max_num_hands': 2,
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.6,
            'model_complexity': 0
        },
        'models': {
            'device': 'cpu',
            'use_fp16': False,
            'weights_path': 'models/weights/hstfe_v1.pth',
            'swin_dim': 768, 'vit_dim': 512,
            'cross_attention_heads': 4,
            'cross_attention_output_dim': 1024,
            'lstm_hidden_dim': 256, 'lstm_num_layers': 2,
            'num_classes': 536, 'classifier_dropout': 0.3
        },
        'gating': {
            'buffer_min_frames': 16, 'buffer_max_frames': 64,
            'optical_flow_low_thresh': 0.2, 'optical_flow_high_thresh': 0.6,
            'confidence_commit_threshold': 0.75,
            'commit_debounce_frames': 8,
            'confidence_drop_threshold': 0.15,
            'confidence_drop_frames': 3,
            'sentence_pause_threshold': 1.5
        },
        'translation': {
            'backend': 'rule_based',
            'prompts_path': 'config/language_prompts.yaml',
            'default_language': 'hindi',
        },
        'tts': {'enabled': False}
    }


def test_pipeline_init_no_camera():
    """Pipeline should initialize without crashing (camera mocked)."""
    from src.pipeline import SignLingoPipeline

    config = _make_config()
    language_prompts = {'hindi': {'system_prompt': 'Test prompt'}}

    with patch('src.pipeline.CameraManager') as MockCam:
        MockCam.return_value = MagicMock()

        pipeline = SignLingoPipeline(config, language_prompts)
        assert pipeline is not None
        assert pipeline._active_language == 'hindi'
        print("Pipeline init PASSED")


def test_set_language():
    """set_language should update active language."""
    from src.pipeline import SignLingoPipeline

    config = _make_config()
    language_prompts = {}

    with patch('src.pipeline.CameraManager'):
        pipeline = SignLingoPipeline(config, language_prompts)
        pipeline.set_language('tamil')
        assert pipeline._active_language == 'tamil'
        print("set_language PASSED")


def test_callback_registration():
    """Callbacks should be registered correctly."""
    from src.pipeline import SignLingoPipeline

    config = _make_config()
    language_prompts = {}

    with patch('src.pipeline.CameraManager'):
        pipeline = SignLingoPipeline(config, language_prompts)

        ui_cb = MagicMock()
        trans_cb = MagicMock()
        pipeline.set_ui_callback(ui_cb)
        pipeline.set_translation_callback(trans_cb)

        assert pipeline._ui_callback is ui_cb
        assert pipeline._translation_callback is trans_cb
        print("Callback registration PASSED")


if __name__ == '__main__':
    test_pipeline_init_no_camera()
    test_set_language()
    test_callback_registration()
    print("\nAll pipeline tests passed.")
