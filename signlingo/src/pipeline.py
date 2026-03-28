import time
import logging
import threading
import numpy as np
import torch
from pathlib import Path
from typing import Callable

from .capture.camera_manager import CameraManager
from .capture.frame_preprocessor import FramePreprocessor
from .detection.mediapipe_detector import HandDetector
from .detection.landmark_normalizer import LandmarkNormalizer
from .models.hstfe import HSTFe
from .models.classifier_head import ClassifierHead
from .models.fallback_recognizer import FallbackRecognizer
from .gating.optical_flow import OpticalFlowAnalyzer
from .gating.buffer_manager import AdaptiveBuffer
from .gating.sign_boundary import SignBoundaryDetector
from .translation.ollama_client import OllamaClient
from .translation.cgme import CGME
from .translation.sentence_composer import SentenceComposer
from .output.tts_engine import TTSEngine
from .output.transcript_logger import TranscriptLogger

logger = logging.getLogger(__name__)


class SignLingoPipeline:
    """
    Orchestrates all SignLingo modules.
    Camera capture, detection, recognition, translation, TTS, and logging.
    """

    def __init__(self, config: dict, language_prompts: dict):
        self.config = config
        self._active_language = config.get('translation', {}).get('default_language', 'hindi')
        self._running = False
        self._process_thread: threading.Thread | None = None
        self._ui_callback: Callable | None = None
        self._translation_callback: Callable | None = None
        self._confidence_history: list[float] = []
        self._prev_landmarks = None
        self._last_sign_time: float = 0.0
        self._no_hand_counter: int = 0

        # Device
        device_str = config.get('models', {}).get('device', 'cpu')
        self.device = torch.device(
            device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu'
        )
        logger.info(f"Pipeline using device: {self.device}")

        # Components
        cam_cfg = config.get('camera', {})
        self.camera = CameraManager(
            device_id=cam_cfg.get('device_id', 0),
            fps=cam_cfg.get('fps', 30)
        )
        self.preprocessor = FramePreprocessor(self.device)
        self.detector = HandDetector(
            **config.get('detection', {})
        )
        self.normalizer = LandmarkNormalizer()
        self.flow_analyzer = OpticalFlowAnalyzer()

        gating_cfg = config.get('gating', {})
        self.buffer = AdaptiveBuffer(
            min_frames=gating_cfg.get('buffer_min_frames', 16),
            max_frames=gating_cfg.get('buffer_max_frames', 64),
            low_thresh=gating_cfg.get('optical_flow_low_thresh', 0.2),
            high_thresh=gating_cfg.get('optical_flow_high_thresh', 0.6)
        )
        self.boundary_detector = SignBoundaryDetector(
            drop_threshold=gating_cfg.get('confidence_drop_threshold', 0.15),
            drop_frames=gating_cfg.get('confidence_drop_frames', 3),
            debounce_frames=gating_cfg.get('commit_debounce_frames', 8)
        )

        # Model: HSTFe or fallback
        models_cfg = config.get('models', {})
        weights_path = models_cfg.get('weights_path', 'models/weights/hstfe_v1.pth')
        self._use_fallback = not Path(weights_path).exists()

        if self._use_fallback:
            logger.warning("HSTFe weights not found, using fallback recognizer")
            self.recognizer = FallbackRecognizer()
            self.hstfe = None
            self.classifier = None
        else:
            self.hstfe = HSTFe.load_weights(weights_path, models_cfg)
            self.classifier = ClassifierHead(
                input_dim=512,
                num_classes=models_cfg.get('num_classes', 536),
                dropout=models_cfg.get('classifier_dropout', 0.3)
            ).to(self.device)
            self.recognizer = None

        # Translation
        trans_cfg = config.get('translation', {})
        self.ollama = OllamaClient(
            base_url=trans_cfg.get('ollama_base_url', 'http://localhost:11434'),
            model=trans_cfg.get('ollama_model', 'llama3.2:3b-instruct-q4_K_M'),
            timeout=trans_cfg.get('ollama_timeout', 10)
        )
        self.cgme = CGME(self.ollama, language_prompts)
        self.composer = SentenceComposer()
        self.tts = TTSEngine()
        self.transcript = TranscriptLogger(
            config.get('app', {}).get('transcript_dir', 'transcripts')
        )

        self._ollama_available = self.ollama.check_connection()
        if not self._ollama_available:
            logger.warning("Ollama not detected — showing ASL gloss only")

    def set_ui_callback(self, callback: Callable):
        self._ui_callback = callback

    def set_translation_callback(self, callback: Callable):
        self._translation_callback = callback

    def set_language(self, language: str):
        self._active_language = language
        logger.info(f"Active language set to: {language}")

    def start(self):
        """Start camera and processing loop."""
        self.camera.start()
        self._running = True
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._process_thread.start()
        logger.info("SignLingo pipeline started")

    def stop(self):
        """Gracefully shutdown all threads."""
        self._running = False
        self.camera.stop()
        self.detector.close()
        self.tts.stop_speaking()
        self.transcript.close()
        if self._process_thread:
            self._process_thread.join(timeout=3.0)
        logger.info("SignLingo pipeline stopped")

    def _process_loop(self):
        """Main processing loop — runs in background thread."""
        commit_threshold = self.config.get('gating', {}).get(
            'confidence_commit_threshold', 0.75
        )

        while self._running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # 1. CLAHE preprocessing
            frame = self.preprocessor.apply_clahe(frame)

            # 2. Hand detection
            mp_frame = self.preprocessor.preprocess_for_mediapipe(frame)
            detection = self.detector.detect(mp_frame)

            # 3. No hands
            if detection.num_hands == 0:
                self._no_hand_counter += 1
                if self._ui_callback:
                    self._ui_callback(detection.annotated_frame, '', 0.0,
                                      self.composer.get_gloss())
                continue
            self._no_hand_counter = 0

            # 4. Normalize landmarks
            norm_landmarks = self.normalizer.flatten_to_vector(
                detection.landmarks_per_hand
            )

            # 5. Optical flow complexity
            complexity = self.flow_analyzer.compute_complexity(
                self._prev_landmarks, norm_landmarks
            )
            self._prev_landmarks = norm_landmarks.copy()

            # 6. Recognize sign
            label, confidence = self._recognize(frame, detection)

            # 7. Update confidence history and buffer
            self._confidence_history.append(confidence)
            if len(self._confidence_history) > 64:
                self._confidence_history.pop(0)

            # 8. Check sign boundary commitment
            if (confidence >= commit_threshold and
                    self.boundary_detector.is_committed(self._confidence_history)):
                self.composer.add_sign(label)
                self.transcript.log_sign(label, confidence, time.time())
                self._last_sign_time = time.time()
                self._confidence_history.clear()
                logger.info(f"Committed sign: {label} ({confidence:.2f})")

            # 9. Check sentence pause
            pause = self.boundary_detector.detect_sentence_pause(
                self._last_sign_time,
                self.config.get('gating', {}).get('sentence_pause_threshold', 1.5)
            )

            if self.composer.is_sentence_ready(pause):
                gloss = self.composer.get_gloss()
                self.composer.clear()
                self._last_sign_time = 0.0
                threading.Thread(
                    target=self._translate_and_emit,
                    args=(gloss,),
                    daemon=True
                ).start()

            # 10. UI update
            if self._ui_callback:
                self._ui_callback(
                    detection.annotated_frame, label, confidence,
                    self.composer.get_gloss()
                )

    def _recognize(self, frame, detection):
        """Run recognition — HSTFe or fallback."""
        if self._use_fallback:
            if detection.landmarks_per_hand:
                return self.recognizer.predict(detection.landmarks_per_hand[0])
            return '', 0.0

        try:
            bbox = detection.bounding_boxes[0] if detection.bounding_boxes else (0, 0, 224, 224)
            hand_crop = self.preprocessor.preprocess_for_swin(frame, bbox)
            full_frame = self.preprocessor.preprocess_for_vit(frame)

            with torch.no_grad():
                features, conf = self.hstfe(hand_crop, full_frame, [])
                _, probs = self.classifier(features)
                label, confidence = self.classifier.predict(probs)
            return label, confidence

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error("CUDA OOM in recognition — skipping frame")
            return '', 0.0
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return '', 0.0

    def _translate_and_emit(self, gloss: str):
        """Translate gloss and emit result (runs in thread)."""
        if self._ollama_available:
            result = self.cgme.translate(gloss, self._active_language)
            native = result.native_text
            roman = result.roman_text
        else:
            native = gloss
            roman = ''

        self.transcript.log_translation(
            gloss, native, roman, self._active_language
        )

        tts_cfg = self.config.get('tts', {})
        if tts_cfg.get('enabled', True):
            self.tts.speak(native, self._active_language)

        if self._translation_callback:
            self._translation_callback(native, roman, gloss, self._active_language)
