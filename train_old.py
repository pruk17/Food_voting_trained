"""
train.py  —  PyTorch + EfficientNetB0 + GPU
============================================
Siamese Network เปรียบเทียบความน่าดึงดูดของภาพอาหาร

โครงสร้างโฟลเดอร์:
ComVision_train/
├── images/
│   ├── instragram_photos/Intragram Images [Original]/
│   │   └── Burger/, Dessert/, Pizza/, Ramen/, Sushi/
│   └── Questionair_Images/
├── data_from_intragram.csv
├── data_from_questionaire.csv
└── train.py
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

# ==================== CONFIG ====================
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
INSTAGRAM_DIR     = os.path.join(BASE_DIR, "images", "instragram_photos", "Intragram Images [Original]")
QUESTIONNAIRE_DIR = os.path.join(BASE_DIR, "images", "Questionair_Images")
MODEL_SAVE_PATH   = os.path.join(BASE_DIR, "model_siamese.pth")

# ==================== TRANSFORMS ====================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ==================== HELPERS ====================
def find_image(filename, food_type=None):
    path = os.path.join(QUESTIONNAIRE_DIR, str(filename))
    if os.path.exists(path):
        return path
    if food_type:
        path = os.path.join(INSTAGRAM_DIR, str(food_type), str(filename))
        if os.path.exists(path):
            return path
    for sub in ["Burger", "Dessert", "Pizza", "Ramen", "Sushi"]:
        path = os.path.join(INSTAGRAM_DIR, sub, str(filename))
        if os.path.exists(path):
            return path
    return None

def load_all_pairs():
    pairs = []
    skipped = 0

    df_q = pd.read_csv(os.path.join(BASE_DIR, "data_from_questionaire.csv"))
    for _, row in df_q.iterrows():
        p1 = find_image(row["Image 1"])
        p2 = find_image(row["Image 2"])
        if p1 and p2:
            pairs.append((p1, p2, 1.0 if int(row["Winner"]) == 1 else 0.0))
        else:
            skipped += 1

    df_i = pd.read_csv(os.path.join(BASE_DIR, "data_from_intragram.csv"))
    for _, row in df_i.iterrows():
        p1 = find_image(row["Image 1"], row["Menu"])
        p2 = find_image(row["Image 2"], row["Menu"])
        if p1 and p2:
            pairs.append((p1, p2, 1.0 if int(row["Winner"]) == 1 else 0.0))
        else:
            skipped += 1

    print(f"โหลดสำเร็จ: {len(pairs)} คู่  |  ข้าม: {skipped} คู่")
    return pairs

# ==================== DATASET ====================
class FoodPairDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs     = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)

# ==================== MODEL ====================
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(eff.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(1280 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        return self.backbone(x).flatten(1)

    def forward(self, x1, x2):
        f1   = self.forward_one(x1)
        f2   = self.forward_one(x2)
        diff = f1 - f2
        return self.classifier(torch.cat([f1, f2, diff], dim=1)).squeeze(1)

    def unfreeze_backbone(self, last_n_blocks=3):
        blocks = list(self.backbone[0].children())
        for block in blocks[-last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"Unfrozen last {last_n_blocks} blocks of backbone")

# ==================== TRAIN / VAL LOOP ====================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    bar = tqdm(loader, desc="  Train", leave=False, ncols=80)
    for img1, img2, labels in bar:
        img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        preds = model(img1, img2)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += ((preds > 0.5).float() == labels).sum().item()
        total      += len(labels)
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total

def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    bar = tqdm(loader, desc="  Val  ", leave=False, ncols=80)
    with torch.no_grad():
        for img1, img2, labels in bar:
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
            preds = model(img1, img2)
            loss  = criterion(preds, labels)
            total_loss += loss.item() * len(labels)
            correct    += ((preds > 0.5).float() == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total

# ==================== MAIN ====================
def train():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 55)

    print("โหลดข้อมูล...")
    pairs = load_all_pairs()
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=0.15, random_state=42,
        stratify=[p[2] for p in pairs]
    )
    print(f"Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    # num_workers=0 เพื่อป้องกัน print ซ้ำบน Windows
    train_dl = DataLoader(FoodPairDataset(train_pairs, train_transform),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(FoodPairDataset(val_pairs,   val_transform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = SiameseNet().to(DEVICE)
    criterion = nn.BCELoss()
    best_val_acc     = 0.0
    patience_counter = 0
    PATIENCE         = 8

    # ===== Phase 1: Frozen backbone =====
    print("\n--- Phase 1: Frozen backbone ---")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    for epoch in range(1, 16):
        tr_loss, tr_acc = train_epoch(model, train_dl, optimizer, criterion)
        v_loss,  v_acc  = val_epoch(model, val_dl, criterion)
        scheduler.step(v_loss)
        print(f"Epoch {epoch:02d}/15 | train_acc={tr_acc:.4f} loss={tr_loss:.4f} | val_acc={v_acc:.4f} loss={v_loss:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ บันทึก model (val_acc={v_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("  Early stopping (Phase 1)")
                break

    # ===== Phase 2: Fine-tune backbone =====
    print("\n--- Phase 2: Fine-tune backbone ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.unfreeze_backbone(last_n_blocks=3)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR / 10)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, optimizer, criterion)
        v_loss,  v_acc  = val_epoch(model, val_dl, criterion)
        scheduler.step(v_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_acc={tr_acc:.4f} loss={tr_loss:.4f} | val_acc={v_acc:.4f} loss={v_loss:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ บันทึก model (val_acc={v_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("  Early stopping (Phase 2)")
                break

    print(f"\n=== เสร็จสิ้น! Best Val Accuracy: {best_val_acc*100:.2f}% ===")
    print(f"Model บันทึกที่: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()