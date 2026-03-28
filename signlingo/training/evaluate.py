"""
Evaluate a trained HSTFe checkpoint.
Usage: python training/evaluate.py --checkpoint models/weights/hstfe_v1.pth --dataset_path ./data
"""
import sys
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hstfe import HSTFe
from src.models.classifier_head import ClassifierHead
from training.dataset_loader import ASLAlphabetDataset
from training.augmentations import get_val_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = HSTFe(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    classifier = ClassifierHead(512, 536).to(device)
    classifier.load_state_dict(checkpoint['classifier_state'])
    classifier.eval()

    dataset = ASLAlphabetDataset(args.dataset_path, transform=get_val_transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for hand_crop, full_frame, labels in loader:
            hand_crop, full_frame, labels = (
                hand_crop.to(device), full_frame.to(device), labels.to(device)
            )
            features, _ = model(hand_crop, full_frame, [])
            logits, _ = classifier(features)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    logger.info(f"Accuracy: {correct/total:.4f} ({correct}/{total})")


if __name__ == '__main__':
    main()
