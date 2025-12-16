import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# -------------------------
# Config
# -------------------------
DATA_DIR = Path("/home/ricardo/proyectoVisionComputador/VisionPorComputadorSDNET-main/data")          # data/Negative, data/Positive
OUT_DIR  = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
VAL_SPLIT = 0.2
NUM_WORKERS = 2
IMG_SIZE = 224  # EfficientNet-B0 default is 224

# -------------------------
# Repro
# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Dataset + split
# -------------------------
base_ds = datasets.ImageFolder(str(DATA_DIR))
print("class_to_idx:", base_ds.class_to_idx)  # verify mapping (Negative/Positive)

n = len(base_ds)
idxs = list(range(n))
random.shuffle(idxs)

val_n = int(n * VAL_SPLIT)
val_idxs = idxs[:val_n]
train_idxs = idxs[val_n:]

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(str(DATA_DIR), transform=train_tf)
val_ds   = datasets.ImageFolder(str(DATA_DIR), transform=val_tf)

train_ds = Subset(train_ds, train_idxs)
val_ds   = Subset(val_ds, val_idxs)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

# -------------------------
# Model: EfficientNet-B0
# -------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 2 classes
model = model.to(device)

# Freeze backbone (good if you have small dataset)
for name, p in model.named_parameters():
    if not name.startswith("classifier."):
        p.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

@torch.no_grad()
def evaluate():
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)

def train_one_epoch():
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)

best_acc = -1.0
out_pt = OUT_DIR / "crack_effnetb0.pt"

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch()
    va_loss, va_acc = evaluate()
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
          f"val loss {va_loss:.4f} acc {va_acc:.3f}")

    if va_acc > best_acc:
        best_acc = va_acc
        torch.save(model, out_pt)

print("Done. Best val acc:", best_acc)
print("Saved:", out_pt)
