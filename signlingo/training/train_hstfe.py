"""
Training script for HSTFe model.
Usage: python training/train_hstfe.py --dataset_path ./data --epochs 50
"""
import sys
import os
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hstfe import HSTFe
from src.models.classifier_head import ClassifierHead
from training.dataset_loader import ASLAlphabetDataset, WLASLDataset, CombinedDataset
from training.augmentations import get_train_transforms, get_val_transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train HSTFe for ASL recognition')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_classes', type=int, default=536)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='models/weights')
    return parser.parse_args()


def train_epoch(model, classifier, loader, optimizer, criterion, device, scaler):
    model.train()
    classifier.train()
    total_loss, correct, total = 0.0, 0, 0

    for hand_crop, full_frame, labels in tqdm(loader, desc='Train', leave=False):
        hand_crop = hand_crop.to(device)
        full_frame = full_frame.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            features, _ = model(hand_crop, full_frame, [])
            logits, probs = classifier(features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def val_epoch(model, classifier, loader, criterion, device):
    model.eval()
    classifier.eval()
    total_loss, correct, total = 0.0, 0, 0

    for hand_crop, full_frame, labels in tqdm(loader, desc='Val', leave=False):
        hand_crop = hand_crop.to(device)
        full_frame = full_frame.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            features, _ = model(hand_crop, full_frame, [])
            logits, _ = classifier(features)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path('training/logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = ASLAlphabetDataset(args.dataset_path, transform=get_train_transforms())
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset.transform = get_val_transforms()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Models
    model_config = {
        'device': args.device,
        'use_fp16': device.type == 'cuda',
        'swin_dim': 768, 'vit_dim': 512,
        'cross_attention_heads': 4,
        'cross_attention_output_dim': 1024,
        'lstm_hidden_dim': 256, 'lstm_num_layers': 2
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
    log_file = log_dir / 'training_log.txt'

    with open(log_file, 'w') as lf:
        lf.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                model, classifier, train_loader, optimizer, criterion, device, scaler
            )
            val_loss, val_acc = val_epoch(model, classifier, val_loader, criterion, device)
            scheduler.step()

            logger.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f} acc={val_acc:.3f}"
            )
            lf.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
            lf.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                checkpoint = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'classifier_state': classifier.state_dict(),
                    'val_acc': val_acc,
                    'config': model_config
                }
                torch.save(checkpoint, Path(args.output_dir) / 'hstfe_v1.pth')
                logger.info(f"Saved best model (val_acc={val_acc:.3f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.3f}")


if __name__ == '__main__':
    main()
