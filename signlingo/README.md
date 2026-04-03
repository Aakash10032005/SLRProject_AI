# SignLingo — Real-Time ASL Recognition + Indian Language Translation

SignLingo recognizes American Sign Language (ASL) in real time and converts gloss to Indian regional languages via a **pluggable translation layer**. **Core recognition + default translation run fully offline** (rule-based lexicon/templates). An optional **Ollama** backend can be enabled in `config/config.yaml` for richer LLM phrasing—no cloud APIs required when using Ollama locally.

**Novel contributions:**
- HSTFe: Hybrid Swin-ViT Temporal Fusion Encoder for dual-stream sign recognition
- CGME: Contextual Grammatical Morphing Engine with pluggable backends (offline rules + optional local LLM)
- Adaptive temporal gating with optical flow complexity estimation
- Local inference: no mandatory cloud dependency; optional Ollama for advanced translation

---

## Hardware Requirements

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA RTX 3050 (4GB VRAM) |
| RAM | 8GB |
| Webcam | Any USB/built-in webcam |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 12+ |

---

## Software Prerequisites

### 1. (Optional) Ollama — for `translation.backend: ollama` only
If you use the default **`translation.backend: rule_based`**, skip Ollama. To use a local LLM for translation:
```bash
# Linux/macOS: https://ollama.com
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b-instruct-q4_K_M
```
Then set in `config/config.yaml`: `translation.backend: ollama` (see **Translation backends** below).

### 2. Install espeak-ng (for TTS)
```bash
# Linux
sudo apt install espeak-ng

# macOS
brew install espeak

# Windows
# Download from: https://github.com/espeak-ng/espeak-ng/releases
# Add espeak-ng to your system PATH
```

### 3. Install Noto Sans fonts (for Indian script display)
```bash
# Linux
sudo apt install fonts-noto

# macOS/Windows: Download from https://fonts.google.com/noto
```

---

## Installation

```bash
# Clone or create project directory
cd signlingo

# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Running

```bash
cd signlingo
# If using translation.backend: ollama, run: ollama serve
python run.py
```

### Translation backends

| `translation.backend` | Behavior |
|------------------------|----------|
| `rule_based` (default) | Offline templates + lexicon in `config/language_prompts.yaml` |
| `ollama` | Uses local Ollama HTTP API (`ollama` block + legacy `ollama_*` keys) |

Extend offline coverage by editing `templates`, `lexicon`, and per-language prompts in `language_prompts.yaml`.

---

## Training (Optional)

Without weights, SignLingo uses the geometry-based fallback recognizer (~87% on letters).
To train the full HSTFe model:

```bash
# Download ASL Alphabet dataset from Kaggle first, then:
python training/train_hstfe.py --dataset_path ./data/asl_alphabet --epochs 50

# See models/weights/README.md for full dataset setup instructions
```

---

## Running Tests

```bash
python -m pytest tests/ -v

# Or run individually
python tests/test_detection.py
python tests/test_models.py
python tests/test_translation.py
python tests/test_pipeline.py
```

---

## Language Selection

Select your target language in the right panel:

| Language | Script | Voice |
|----------|--------|-------|
| Hindi | Devanagari | hi |
| Tamil | Tamil | ta |
| Telugu | Telugu | te |
| Malayalam | Malayalam | ml |
| Marathi | Devanagari | mr |

Toggle "Show Transliteration" to see Roman phonetic output alongside native script.

---

## Architecture Overview

```
Webcam → CameraManager → FramePreprocessor → HandDetector (MediaPipe)
                                                    ↓
                              LandmarkNormalizer + OpticalFlowAnalyzer
                                                    ↓
                         SwinStream (hand crop) + ViTStream (full frame)
                                                    ↓
                                      CrossAttentionBridge
                                                    ↓
                                         TemporalLSTM
                                                    ↓
                                       ClassifierHead → Sign Label
                                                    ↓
                              AdaptiveBuffer + SignBoundaryDetector
                                                    ↓
                              SentenceComposer → ASL Gloss String
                                                    ↓
                                    CGME (Ollama llama3.2:3b)
                                                    ↓
                              Native Script + Roman Transliteration
                                                    ↓
                                    TTSEngine (espeak-ng) + UI
```

---

## Troubleshooting

**CUDA out of memory**
- Close other GPU applications
- Lower `batch_size` in training
- Switch to `gemma2:2b` in config for faster Ollama responses

**Camera not detected**
- Try changing `device_id` in `config/config.yaml` (try 0, 1, 2)
- Verify with: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

**Translation looks like raw English gloss**
- You are on `rule_based`: add entries under `lexicon` in `language_prompts.yaml`, or switch to `translation.backend: ollama` with `ollama serve` running.

**Ollama connection refused** (only if `backend: ollama`)
- Run `ollama serve` in a separate terminal and keep it running

**No Indian script display**
- Install Noto Sans fonts (see prerequisites above)

**espeak-ng not found**
- Add espeak-ng to system PATH, or the system will auto-fallback to pyttsx3

**Low recognition accuracy**
- Ensure good lighting (>300 lux)
- Keep hand 40–80cm from camera
- Train HSTFe weights for best results (see `models/weights/README.md`)

**Signs not committing**
- Lower `confidence_commit_threshold` in `config/config.yaml` (default: 0.75)

---

## License

Apache 2.0 — see LICENSE file.
