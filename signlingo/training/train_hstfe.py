"""
Training script for HSTFe model.
Usage: python -m training.train_hstfe --dataset_path ./data --epochs 40

RTX 3050 (4 GB VRAM) budget:
  - T_MAX=4, batch_size=4  →  B*T = 16 images through Swin+ViT per step
  - AMP autocast (float16 activations, float32 master weights)
  - use_fp16=False in model config (weights stay float32; AMP handles casting)

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

# ── VRAM budget constants ──────────────────────────────────────────────────────
# RTX 3050 4 GB: Swin-tiny ~110 MB, ViT-small ~85 MB, cross-attn+LSTM ~30 MB
# Safe working set: B*T ≤ 16 images through spatial encoder per step.
T_MIN, T_MAX = 2, 4   # keep B*T_max = batch*4 ≤ 16 → batch ≤ 4


# ──────────────────────────────────────────────────────────────────────────────
# Collate: pad variable-length sequences and build attention mask
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Replicates each image T times (T ∈ [T_MIN, T_MAX]) to simulate a clip.
    Keeps B*T_max within VRAM budget for RTX 3050.

    Returns:
        images      [B, T_max, C, H, W]
        full_frames [B, T_max, C, H, W]
        labels      [B]
        mask        [B, T_max]  True = valid frame
        seq_lens    [B]
    """
    images, full_frames, labels, seq_lens = [], [], [], []

    for img, full, label in batch:
        t = random.randint(T_MIN, T_MAX)
        clip      = img.unsqueeze(0).expand(t, -1, -1, -1).contiguous()
        full_clip = full.unsqueeze(0).expand(t, -1, -1, -1).contiguous()
        images.append(clip)
        full_frames.append(full_clip)
        labels.append(label)
        seq_lens.append(t)

    max_len = max(seq_lens)

    padded_imgs = torch.stack([
        F.pad(seq, (0, 0, 0, 0, 0, 0, 0, max_len - len(seq)))
        for seq in images
    ])  # [B, T_max, C, H, W]

    padded_full = torch.stack([
        F.pad(seq, (0, 0, 0, 0, 0, 0, 0, max_len - len(seq)))
        for seq in full_frames
    ])

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
    use_amp = device.type == 'cuda'

    for images, full_frames, labels, mask, seq_lens in tqdm(loader, desc='Train', leave=False):
        B, T = images.shape[:2]
        images      = images.to(device)
        full_frames = full_frames.to(device)
        labels      = labels.to(device)
        mask        = mask.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            imgs_flat = images.contiguous().view(B * T, *images.shape[2:])
            full_flat = full_frames.contiguous().view(B * T, *full_frames.shape[2:])

            spatial_out = model.encode_spatial(imgs_flat, full_flat)   # [B*T, 1024]
            spatial_seq = spatial_out.view(B, T, -1)                   # [B, T, 1024]

            out, _ = model.forward_temporal(spatial_seq, hidden=None)  # [B, T, 512]

            last_idx = (seq_lens - 1).clamp(min=0).to(device)
            features = out[torch.arange(B, device=device), last_idx]  # [B, 512]

            logits, _ = classifier(features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def val_epoch(model, classifier, loader, criterion, device):
    model.eval()
    classifier.eval()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = device.type == 'cuda'

    for images, full_frames, labels, mask, seq_lens in tqdm(loader, desc='Val', leave=False):
        B, T = images.shape[:2]
        images      = images.to(device)
        full_frames = full_frames.to(device)
        labels      = labels.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            imgs_flat = images.contiguous().view(B * T, *images.shape[2:])
            full_flat = full_frames.contiguous().view(B * T, *full_frames.shape[2:])

            spatial_out = model.encode_spatial(imgs_flat, full_flat)
            spatial_seq = spatial_out.view(B, T, -1)

            out, _ = model.forward_temporal(spatial_seq, hidden=None)

            last_idx = (seq_lens - 1).clamp(min=0).to(device)
            features = out[torch.arange(B, device=device), last_idx]

            logits, _ = classifier(features)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train HSTFe for ASL recognition')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--mode', type=str, default='alphabet',
                        choices=['alphabet', 'wlasl_lite', 'wlasl', 'combined'])
    parser.add_argument('--epochs',     type=int,   default=40)
    parser.add_argument('--batch_size', type=int,   default=4,
                        help='Per-step batch size. Keep batch*T_MAX ≤ 16 for 4 GB VRAM.')
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--num_classes',type=int,   default=29,
                        help='29 for alphabet-only, 536 for full WLASL combined.')
    parser.add_argument('--patience',   type=int,   default=10)
    parser.add_argument('--output_dir', type=str,   default='models/weights')
    parser.add_argument('--n_frames',   type=int,   default=4,
                        help='Frames per WLASL clip. Keep ≤ 4 for 4 GB VRAM.')
    parser.add_argument('--num_workers',type=int,   default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name}  VRAM: {props.total_memory // 1024**2} MB")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path('training/logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    root = Path(args.dataset_path)

    def _make_wlasl(transform, manifest=None):
        return WLASLLazyDataset(root, transform=transform,
                                n_frames=args.n_frames, manifest=manifest)

    def _make_alphabet(transform):
        return ASLAlphabetDataset(root, transform=transform)

    if args.mode == 'alphabet':
        full_train = _make_alphabet(get_train_transforms())
        full_val   = _make_alphabet(get_val_transforms())

    elif args.mode == 'wlasl_lite':
        manifest = root / 'lite_manifest.json'
        if not manifest.exists():
            logger.error("lite_manifest.json not found. Run prepare_wlasl.py first.")
            sys.exit(1)
        full_train = _make_wlasl(get_train_transforms(), manifest=manifest)
        full_val   = _make_wlasl(get_val_transforms(),   manifest=manifest)

    elif args.mode == 'wlasl':
        full_train = _make_wlasl(get_train_transforms())
        full_val   = _make_wlasl(get_val_transforms())

    elif args.mode == 'combined':
        manifest = root / 'lite_manifest.json'
        full_train = CombinedDataset(
            _make_alphabet(get_train_transforms()),
            _make_wlasl(get_train_transforms(), manifest=manifest if manifest.exists() else None)
        )
        full_val = CombinedDataset(
            _make_alphabet(get_val_transforms()),
            _make_wlasl(get_val_transforms(), manifest=manifest if manifest.exists() else None)
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # 90/10 split — use same indices on both transform instances
    n = len(full_train)
    val_size  = max(1, int(n * 0.1))
    indices   = list(range(n))
    train_ds  = Subset(full_train, indices[:-val_size])
    val_ds    = Subset(full_val,   indices[-val_size:])

    logger.info("mode=%s  train=%d  val=%d  batch=%d  T_MAX=%d  B*T_max=%d",
                args.mode, len(train_ds), len(val_ds),
                args.batch_size, T_MAX, args.batch_size * T_MAX)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.num_workers)

    # ── Models ────────────────────────────────────────────────────────────────
    model_config = {
        'device':                    args.device,
        'use_fp16':                  False,   # AMP handles casting; weights stay fp32
        'swin_dim':                  768,
        'vit_dim':                   512,
        'cross_attention_heads':     4,
        'cross_attention_output_dim':1024,
        'lstm_hidden_dim':           256,
        'lstm_num_layers':           2,
        'lstm_dropout':              0.1,
    }
    model      = HSTFe(model_config)
    classifier = ClassifierHead(512, args.num_classes).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc    = 0.0
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
                "Epoch %d/%d | Train loss=%.4f acc=%.3f | Val loss=%.4f acc=%.3f",
                epoch, args.epochs, train_loss, train_acc, val_loss, val_acc
            )
            lf.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
            lf.flush()

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                patience_counter = 0
                torch.save(
                    {
                        'epoch':            epoch,
                        'model_state':      model.state_dict(),
                        'classifier_state': classifier.state_dict(),
                        'val_acc':          val_acc,
                        'config':           model_config,
                        'num_classes':      args.num_classes,
                    },
                    Path(args.output_dir) / 'hstfe_v1.pth',
                )
                logger.info("  ✓ Saved best checkpoint (val_acc=%.3f)", val_acc)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

    logger.info("Training complete. Best val_acc=%.3f", best_val_acc)


if __name__ == '__main__':
    main()
