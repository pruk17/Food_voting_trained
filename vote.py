"""
vote.py  —  Dual Backbone Siamese Network predictor with TTA
=============================================================
Loads trained model and predicts which image is more attractive for each pair.
Uses Test Time Augmentation (TTA) to improve prediction confidence.

How TTA works:
    Instead of predicting once with the original image,
    we predict N times with slightly different versions of the same image,
    then average the probabilities for a more reliable result.

Usage:
    python vote.py --test_csv test.csv --img_dir "Test Images"
    python vote.py --test_csv test.csv --img_dir "Test Images" --tta 8
"""

import os
import argparse
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ==================== CONFIG ====================
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_siamese.pth")

# ==================== TRANSFORMS ====================
# Base transform: no augmentation (always applied)
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# TTA transforms: slight variations of the same image
tta_transforms = [
    # Original (no change)
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Slightly brighter
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Slightly darker (brightness tuple: factor drawn from [0.6, 0.8])
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=(0.6, 0.8)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Slight rotation clockwise
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(8, 12)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Slight rotation counter-clockwise
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(-12, -8)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Higher contrast
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Horizontal flip + brightness
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]

# ==================== MODEL ====================
# Must match architecture in train.py exactly
class DualBackboneSiameseNet(nn.Module):
    def __init__(self):
        super().__init__()

        b0 = models.efficientnet_b0(weights=None)
        self.backbone_b0 = nn.Sequential(*list(b0.children())[:-1])

        b3 = models.efficientnet_b3(weights=None)
        self.backbone_b3 = nn.Sequential(*list(b3.children())[:-1])

        feat_dim  = 1280 + 1536  # 2816 per image
        input_dim = feat_dim * 3  # feat1 + feat2 + diff = 8448

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def extract(self, x):
        f_b0 = self.backbone_b0(x).flatten(1)
        f_b3 = self.backbone_b3(x).flatten(1)
        return torch.cat([f_b0, f_b3], dim=1)

    def forward(self, x1, x2):
        feat1    = self.extract(x1)
        feat2    = self.extract(x2)
        diff     = feat1 - feat2
        combined = torch.cat([feat1, feat2, diff], dim=1)
        return self.classifier(combined).squeeze(1)

# ==================== PREDICT ====================
def predict_winner(model, img1_path, img2_path, n_tta=8):
    """
    Predict which image is more attractive using TTA.

    For each TTA version:
        - Apply the same transform to BOTH images
        - Get probability that image 1 wins
    Then average all probabilities and decide.

    Args:
        n_tta: number of TTA transforms to use (1 = no TTA, max = 8)
    """
    img1_pil = Image.open(img1_path).convert("RGB")
    img2_pil = Image.open(img2_path).convert("RGB")

    probs = []
    selected_transforms = tta_transforms[:n_tta]

    for tfm in selected_transforms:
        img1 = tfm(img1_pil).unsqueeze(0).to(DEVICE)
        img2 = tfm(img2_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = model(img1, img2).item()
        probs.append(prob)

    avg_prob = sum(probs) / len(probs)
    return 1 if avg_prob > 0.5 else 2

# ==================== MAIN ====================
def main(test_csv, img_dir, n_tta, output_csv):
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"TTA    : {n_tta} augmentations per image pair")

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = DualBackboneSiameseNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # Load test pairs
    df = pd.read_csv(test_csv)
    print(f"Total pairs to predict: {len(df)}\n")

    winners = []
    errors  = 0

    for idx, row in df.iterrows():
        p1 = os.path.join(img_dir, str(row["Image 1"]))
        p2 = os.path.join(img_dir, str(row["Image 2"]))

        if not os.path.exists(p1) or not os.path.exists(p2):
            print(f"[WARNING] Image not found at row {idx}: {row['Image 1']} or {row['Image 2']}")
            winners.append(1)
            errors += 1
            continue

        winners.append(predict_winner(model, p1, p2, n_tta=n_tta))

        if (idx + 1) % 10 == 0:
            print(f"  Predicted {idx + 1}/{len(df)}")

    # Write results to output file
    df["Winner"] = winners
    df.to_csv(output_csv, index=False)

    print(f"\nResults saved to : {output_csv}")
    print(f"  Predicted 1    : {winners.count(1)} pairs")
    print(f"  Predicted 2    : {winners.count(2)} pairs")
    if errors:
        print(f"  [WARNING] Missing images: {errors} pairs (defaulted to 1)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", default="test.csv",
                        help="Path to input test CSV file")
    parser.add_argument("--output",   default=None,
                        help="Path to save results CSV (default: overwrite test_csv)")
    parser.add_argument("--img_dir",  default="Test Images",
                        help="Folder containing test images")
    parser.add_argument("--tta",      default=8, type=int,
                        help="Number of TTA augmentations (1=no TTA, max=8)")
    args = parser.parse_args()
    output_path = args.output if args.output else args.test_csv
    main(args.test_csv, args.img_dir, args.tta, output_path)