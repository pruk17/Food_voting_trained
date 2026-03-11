"""
train.py  —  Dual Backbone Siamese Network (EfficientNet B0 + B3)
=================================================================
Improvements over previous version:
  1. Pair swapping: doubles training data by flipping each pair
     (A, B) label=1  →  also add  (B, A) label=0
  2. Label smoothing: prevents overconfidence
     label 1.0 → 0.9,  label 0.0 → 0.1

Directory structure:
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
IMG_SIZE       = 224
BATCH_SIZE     = 16       # reduced to fit dual backbone in 4GB VRAM
EPOCHS         = 50
LR             = 1e-4
WEIGHT_DECAY   = 1e-4     # L2 regularization
PATIENCE       = 5        # early stopping patience
PHASE1_EPOCHS  = 10       # frozen backbone epochs
LABEL_SMOOTH   = 0.1      # label smoothing: 1.0→0.9, 0.0→0.1
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
INSTAGRAM_DIR     = os.path.join(BASE_DIR, "images", "instragram_photos", "Intragram Images [Original]")
QUESTIONNAIRE_DIR = os.path.join(BASE_DIR, "images", "Questionair_Images")
MODEL_SAVE_PATH   = os.path.join(BASE_DIR, "model_siamese.pth")

# ==================== TRANSFORMS ====================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
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
    """Search for image file across known directories."""
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

def smooth_label(label, epsilon=LABEL_SMOOTH):
    """
    Apply label smoothing.
    Converts hard labels to soft labels to prevent overconfidence.
        1.0  →  1.0 - epsilon  (e.g. 0.9)
        0.0  →  0.0 + epsilon  (e.g. 0.1)
    """
    return label * (1 - epsilon) + (1 - label) * epsilon

def load_all_pairs():
    """
    Load image pairs from both CSV files.
    Also creates swapped pairs (B, A) from each original (A, B)
    to double the training data while keeping val set unswapped.

    Original:  (img1, img2, label=1.0)  →  img1 is better
    Swapped:   (img2, img1, label=0.0)  →  img1 (was img2) is worse
    """
    pairs   = []
    skipped = 0

    # --- Questionnaire data (500 pairs) ---
    df_q = pd.read_csv(os.path.join(BASE_DIR, "data_from_questionaire.csv"))
    for _, row in df_q.iterrows():
        p1 = find_image(row["Image 1"])
        p2 = find_image(row["Image 2"])
        if p1 and p2:
            label = 1.0 if int(row["Winner"]) == 1 else 0.0
            pairs.append((p1, p2, label))
        else:
            skipped += 1

    # --- Instagram data (265 pairs) ---
    df_i = pd.read_csv(os.path.join(BASE_DIR, "data_from_intragram.csv"))
    for _, row in df_i.iterrows():
        p1 = find_image(row["Image 1"], row["Menu"])
        p2 = find_image(row["Image 2"], row["Menu"])
        if p1 and p2:
            label = 1.0 if int(row["Winner"]) == 1 else 0.0
            pairs.append((p1, p2, label))
        else:
            skipped += 1

    print(f"Original pairs loaded : {len(pairs)}  |  Skipped: {skipped}")
    return pairs

def augment_with_swaps(train_pairs):
    """
    Double the training data by adding swapped versions of each pair.
    (img1, img2, 1.0)  →  also add  (img2, img1, 0.0)
    (img1, img2, 0.0)  →  also add  (img2, img1, 1.0)

    Note: only applied to TRAIN set, not val set.
    """
    swapped = [(p2, p1, 1.0 - label) for p1, p2, label in train_pairs]
    combined = train_pairs + swapped
    print(f"After swap augmentation: {len(train_pairs)} → {len(combined)} train pairs")
    return combined

# ==================== DATASET ====================
class FoodPairDataset(Dataset):
    def __init__(self, pairs, transform=None, use_label_smooth=False):
        self.pairs             = pairs
        self.transform         = transform
        self.use_label_smooth  = use_label_smooth

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Apply label smoothing only during training
        if self.use_label_smooth:
            label = smooth_label(label)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# ==================== MODEL ====================
class DualBackboneSiameseNet(nn.Module):
    """
    Dual Backbone Siamese Network.

    Each image passes through TWO backbones (shared weights):
        B0 (feat=1280) + B3 (feat=1536) → combined feat=2816 per image

    Then compared:
        combined = [feat1 | feat2 | feat1-feat2]  →  size 8448
        → Dense → sigmoid → P(image1 wins)
    """
    def __init__(self):
        super().__init__()

        b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone_b0 = nn.Sequential(*list(b0.children())[:-1])

        b3 = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.backbone_b3 = nn.Sequential(*list(b3.children())[:-1])

        # Freeze both backbones initially (Phase 1)
        for param in self.backbone_b0.parameters():
            param.requires_grad = False
        for param in self.backbone_b3.parameters():
            param.requires_grad = False

        feat_dim  = 1280 + 1536  # 2816 per image
        input_dim = feat_dim * 3  # 8448

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
        """Extract and concatenate features from both backbones."""
        f_b0 = self.backbone_b0(x).flatten(1)  # (B, 1280)
        f_b3 = self.backbone_b3(x).flatten(1)  # (B, 1536)
        return torch.cat([f_b0, f_b3], dim=1)  # (B, 2816)

    def forward(self, x1, x2):
        feat1    = self.extract(x1)
        feat2    = self.extract(x2)
        diff     = feat1 - feat2
        combined = torch.cat([feat1, feat2, diff], dim=1)
        return self.classifier(combined).squeeze(1)

    def unfreeze_last_blocks(self, n_blocks=3):
        """Unfreeze last N blocks of both backbones for fine-tuning."""
        for name, backbone in [("B0", self.backbone_b0), ("B3", self.backbone_b3)]:
            blocks = list(backbone[0].children())
            for block in blocks[-n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
        print(f"Unfrozen last {n_blocks} blocks of both backbones (B0 + B3)")

# ==================== TRAIN / VAL LOOP ====================
def run_epoch(model, loader, optimizer, criterion, training=True):
    """Run one full epoch of training or validation."""
    model.train() if training else model.eval()
    total_loss, correct, total = 0, 0, 0
    tag = "Train" if training else "Val  "

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        bar = tqdm(loader, desc=f"  {tag}", leave=False, ncols=80)
        for img1, img2, labels in bar:
            img1   = img1.to(DEVICE)
            img2   = img2.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            preds = model(img1, img2)
            loss  = criterion(preds, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            # Accuracy uses hard threshold 0.5 (not smoothed labels)
            hard_labels = (labels > 0.5).float()
            correct     += ((preds > 0.5).float() == hard_labels).sum().item()
            total       += len(labels)
            bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total

# ==================== TRAINING PHASE ====================
def run_phase(model, train_dl, val_dl, criterion,
              lr, max_epochs, phase_name, best_val_acc):
    """Generic training loop for both phases."""
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler        = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    patience_counter = 0

    print(f"\n--- {phase_name} ---")
    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, optimizer, criterion, training=True)
        v_loss,  v_acc  = run_epoch(model, val_dl,   optimizer, criterion, training=False)
        scheduler.step(v_loss)

        print(f"Epoch {epoch:02d}/{max_epochs} | "
              f"train_acc={tr_acc:.4f} loss={tr_loss:.4f} | "
              f"val_acc={v_acc:.4f} loss={v_loss:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ Saved model (val_acc={v_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping ({phase_name})")
                break

    return best_val_acc

# ==================== MAIN ====================
def train():
    print(f"Device        : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"Label smooth  : {LABEL_SMOOTH}  (1.0→{1-LABEL_SMOOTH}, 0.0→{LABEL_SMOOTH})")
    print("=" * 55)

    # Load original pairs
    print("Loading data...")
    pairs = load_all_pairs()

    # Split BEFORE swapping (val must stay clean / unswapped)
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=0.15, random_state=42,
        stratify=[p[2] for p in pairs]
    )

    # Double training data by swapping pairs
    train_pairs = augment_with_swaps(train_pairs)
    print(f"Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    train_dl = DataLoader(
        FoodPairDataset(train_pairs, train_transform, use_label_smooth=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_dl = DataLoader(
        # Val uses hard labels (no smoothing) for accurate accuracy measurement
        FoodPairDataset(val_pairs, val_transform, use_label_smooth=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model     = DualBackboneSiameseNet().to(DEVICE)
    criterion = nn.BCELoss()

    if DEVICE.type == "cuda":
        print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Phase 1: train classifier only (backbones frozen)
    best_val_acc = run_phase(
        model, train_dl, val_dl, criterion,
        lr=LR, max_epochs=PHASE1_EPOCHS,
        phase_name="Phase 1: Frozen Backbones",
        best_val_acc=0.0
    )

    # Phase 2: fine-tune last blocks of both backbones
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.unfreeze_last_blocks(n_blocks=3)

    best_val_acc = run_phase(
        model, train_dl, val_dl, criterion,
        lr=LR / 10, max_epochs=EPOCHS,
        phase_name="Phase 2: Fine-tune Backbones",
        best_val_acc=best_val_acc
    )

    print(f"\n{'='*55}")
    print(f"Training complete!  Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to    : {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()