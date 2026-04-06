"""
Training script for HSTFe model.
Usage: python -m training.train_hstfe --dataset_path ./data --epochs 40

Variable-length clip training (T ∈ [3, 16]) so weights match live inference.
DO NOT modify src/models/hstfe.py or src/pipeline.py.
"""
import sys
import argparse
import logging
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hstfe import HSTFe
from src.models.classifier_head import ClassifierHead
from training.dataset_loader import ASLAlphabetDataset, CombinedDataset
from training.wlasl_dataset import WLASLLazyDataset
from training.augmentations import get_train_transforms, get_val_transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Collate: pad variable-length sequences and build attention mask
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Each item from the dataset is (image_tensor, full_frame_tensor, label).
    We replicate the single image T times (T sampled in [3, 16]) to simulate
    a temporal clip, matching the variable-window inference path in pipeline.py.

    Returns:
        images      [B, T_max, C, H, W]
        full_frames [B, T_max, C, H, W]
        labels      [B]
        mask        [B, T_max]  True = valid frame
        seq_lens    [B]
    """
    T_MIN, T_MAX = 3, 16
    images, full_frames, labels, seq_lens = [], [], [], []

    for img, full, label in batch:
        t = random.randint(T_MIN, T_MAX)
        # Replicate single frame t times to form a clip
        clip = img.unsqueeze(0).expand(t, -1, -1, -1)       # [T, C, H, W]
        full_clip = full.unsqueeze(0).expand(t, -1, -1, -1)  # [T, C, H, W]
        images.append(clip)
        full_frames.append(full_clip)
        labels.append(label)
        seq_lens.append(t)

    max_len = max(seq_lens)

    # Pad each clip to max_len along the time dimension
    padded_imgs = torch.stack([
        F.pad(seq, (0, 0, 0, 0, 0, 0, 0, max_len - len(seq)))
        for seq in images
    ])  # [B, T_max, C, H, W]

    padded_full = torch.stack([
        F.pad(seq, (0, 0, 0, 0, 0, 0, 0, max_len - len(seq)))
        for seq in full_frames
    ])  # [B, T_max, C, H, W]

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long)
    mask = (
        torch.arange(max_len).unsqueeze(0).expand(len(batch), -1)
        < seq_lens_t.unsqueeze(1)
    )  # [B, T_max]  True = valid

    return padded_imgs, padded_full, torch.tensor(labels), mask, seq_lens_t


# ──────────────────────────────────────────────────────────────────────────────
# Training / validation loops
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, classifier, loader, optimizer, criterion, device, scaler):
    model.train()
    classifier.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, full_frames, labels, mask, seq_lens in tqdm(loader, desc='Train', leave=False):
        B, T = images.shape[:2]
        images = images.to(device)
        full_frames = full_frames.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            # Flatten time into batch for spatial encoding
            imgs_flat = images.view(B * T, *images.shape[2:])       # [B*T, C, H, W]
            full_flat = full_frames.view(B * T, *full_frames.shape[2:])

            spatial_out = model.encode_spatial(imgs_flat, full_flat)  # [B*T, 1024]
            spatial_seq = spatial_out.view(B, T, -1)                  # [B, T, 1024]

            # Full sequence through temporal LSTM (hidden=None for bidirectional)
            out, _ = model.forward_temporal(spatial_seq, hidden=None)  # [B, T, 512]

            # Use last valid frame per sequence for classification
            last_idx = (seq_lens - 1).clamp(min=0).to(device)         # [B]
            features = out[torch.arange(B, device=device), last_idx]  # [B, 512]

            logits, _ = classifier(features)
            loss = criterion(logits, labels)

            # Optional: mask-weighted loss (down-weight padded positions)
            # Already using last valid frame so this is a no-op, but kept for clarity
            loss = loss * mask[:, 0].float().mean()  # normalise by valid-batch ratio

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def val_epoch(model, classifier, loader, criterion, device):
    model.eval()
    classifier.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, full_frames, labels, mask, seq_lens in tqdm(loader, desc='Val', leave=False):
        B, T = images.shape[:2]
        images = images.to(device)
        full_frames = full_frames.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            imgs_flat = images.view(B * T, *images.shape[2:])
            full_flat = full_frames.view(B * T, *full_frames.shape[2:])

            spatial_out = model.encode_spatial(imgs_flat, full_flat)
            spatial_seq = spatial_out.view(B, T, -1)

            out, _ = model.forward_temporal(spatial_seq, hidden=None)

            last_idx = (seq_lens - 1).clamp(min=0).to(device)
            features = out[torch.arange(B, device=device), last_idx]

            logits, _ = classifier(features)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train HSTFe for ASL recognition')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Root path for the dataset')
    parser.add_argument('--mode', type=str, default='alphabet',
                        choices=['alphabet', 'wlasl_lite', 'wlasl', 'combined'],
                        help=(
                            'alphabet      — Kaggle ASL Alphabet images only\n'
                            'wlasl_lite    — WLASL lazy loader, lite_manifest.json subset\n'
                            'wlasl         — WLASL lazy loader, all classes\n'
                            'combined      — alphabet + wlasl_lite merged'
                        ))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_classes', type=int, default=536)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='models/weights')
    parser.add_argument('--n_frames', type=int, default=8,
                        help='Frames per WLASL clip (lazy loader). Default 8 fits 4 GB VRAM.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path('training/logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset selection ─────────────────────────────────────────────────────
    root = Path(args.dataset_path)
    mode = args.mode

    def _make_wlasl(transform, manifest=None):
        return WLASLLazyDataset(
            root, transform=transform, n_frames=args.n_frames,
            manifest=manifest,
        )

    def _make_alphabet(transform):
        return ASLAlphabetDataset(root, transform=transform)

    if mode == 'alphabet':
        base_train = _make_alphabet(get_train_transforms())
        base_val   = _make_alphabet(get_val_transforms())

    elif mode == 'wlasl_lite':
        manifest = root / 'lite_manifest.json'
        if not manifest.exists():
            logger.error(
                "lite_manifest.json not found. Run:\n"
                "  python training/prepare_wlasl.py --root %s --mode lite", root
            )
            sys.exit(1)
        base_train = _make_wlasl(get_train_transforms(), manifest=manifest)
        base_val   = _make_wlasl(get_val_transforms(),   manifest=manifest)

    elif mode == 'wlasl':
        base_train = _make_wlasl(get_train_transforms())
        base_val   = _make_wlasl(get_val_transforms())

    elif mode == 'combined':
        manifest = root / 'lite_manifest.json'
        alpha_train = _make_alphabet(get_train_transforms())
        alpha_val   = _make_alphabet(get_val_transforms())
        wlasl_train = _make_wlasl(get_train_transforms(), manifest=manifest if manifest.exists() else None)
        wlasl_val   = _make_wlasl(get_val_transforms(),   manifest=manifest if manifest.exists() else None)
        base_train = CombinedDataset(alpha_train, wlasl_train)
        base_val   = CombinedDataset(alpha_val,   wlasl_val)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    logger.info("Dataset mode=%s  train=%d  val=%d", mode, len(base_train), len(base_val))

    # For non-combined modes, split 90/10
    if mode != 'combined':
        n = len(base_train)
        val_size = max(1, int(n * 0.1))
        indices = list(range(n))
        train_ds = Subset(base_train, indices[:-val_size])
        val_ds   = Subset(base_val,   indices[-val_size:])
    else:
        train_ds = base_train
        val_ds   = base_val

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )

    # Models
    model_config = {
        'device': args.device,
        'use_fp16': False,   # FP16 disabled during training — use AMP autocast instead
        'swin_dim': 768, 'vit_dim': 512,
        'cross_attention_heads': 4,
        'cross_attention_output_dim': 1024,
        'lstm_hidden_dim': 256, 'lstm_num_layers': 2,
        'lstm_dropout': 0.1,
    }
    model = HSTFe(model_config)
    classifier = ClassifierHead(512, args.num_classes).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    best_val_acc = 0.0
    patience_counter = 0
    log_file = log_dir / 'training_log.csv'

    with open(log_file, 'w') as lf:
        lf.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                model, classifier, train_loader, optimizer, criterion, device, scaler
            )
            val_loss, val_acc = val_epoch(
                model, classifier, val_loader, criterion, device
            )
            scheduler.step()

            logger.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"Val loss={val_loss:.4f} acc={val_acc:.3f}"
            )
            lf.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
            lf.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'classifier_state': classifier.state_dict(),
                        'val_acc': val_acc,
                        'config': model_config,
                    },
                    Path(args.output_dir) / 'hstfe_v1.pth',
                )
                logger.info(f"  ✓ Saved best checkpoint (val_acc={val_acc:.3f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                    break

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.3f}")


if __name__ == '__main__':
    main()
