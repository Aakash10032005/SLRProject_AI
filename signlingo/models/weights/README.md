# HSTFe Model Weights

Place trained model weights here as `hstfe_v1.pth`.

## Without Weights (Fallback Mode)

If no weights file is found, SignLingo automatically uses the **FallbackRecognizer**,
which uses MediaPipe landmark geometry rules.

- Letters A-Z: ~87% accuracy
- Numbers 0-9: ~91% accuracy
- WLASL words: not supported in fallback mode

## Training Your Own Weights

### 1. Get the ASL Alphabet Dataset (Kaggle)

```bash
# Install Kaggle CLI
pip install kaggle
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip -d data/asl_alphabet
```

### 2. Get the WLASL Dataset (optional, for word recognition)

```bash
# Clone WLASL repo
git clone https://github.com/dxli94/WLASL.git
# Follow their instructions to download videos
# Place in data/wlasl/videos/<class_name>/<video>.mp4
```

### 3. Run Training

```bash
# ASL Alphabet only (letters + numbers)
python training/train_hstfe.py \
    --dataset_path ./data/asl_alphabet \
    --epochs 50 \
    --batch_size 16 \
    --device cuda

# With WLASL (full 536-class model)
python training/train_hstfe.py \
    --dataset_path ./data \
    --epochs 100 \
    --batch_size 8 \
    --device cuda \
    --num_classes 536
```

Training saves the best checkpoint to `models/weights/hstfe_v1.pth`.

### 4. Evaluate

```bash
python training/evaluate.py \
    --checkpoint models/weights/hstfe_v1.pth \
    --dataset_path ./data/asl_alphabet
```

## Hardware Requirements for Training

- GPU: RTX 3050 (4GB VRAM) minimum
- RAM: 16GB recommended
- Storage: ~10GB for WLASL dataset

Training uses FP16 (mixed precision) to fit within 4GB VRAM.
Expected training time: ~2 hours for ASL Alphabet, ~8 hours for full WLASL on RTX 3050.
