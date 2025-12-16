import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# -------------------------
# Config
# -------------------------
DATA_DIR = Path("/home/ricardo/proyectoVisionComputador/VisionPorComputadorSDNET-main/data")              # expects data/Negative and data/Positive
OUT_DIR  = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
VAL_SPLIT = 0.2
NUM_WORKERS = 2

IMG_SIZE = 224  # ResNet default

# -------------------------
# Reproducibility
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
# Transforms
# -------------------------
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

# -------------------------
# Dataset + split
# -------------------------
# We load twice so train and val can have different transforms cleanly.
base_ds = datasets.ImageFolder(str(DATA_DIR))
print("class_to_idx:", base_ds.class_to_idx)  # IMPORTANT: verify mapping

n = len(base_ds)
idxs = list(range(n))
random.shuffle(idxs)

val_n = int(n * VAL_SPLIT)
val_idxs = idxs[:val_n]
train_idxs = idxs[val_n:]

train_ds = datasets.ImageFolder(str(DATA_DIR), transform=train_tf)
val_ds   = datasets.ImageFolder(str(DATA_DIR), transform=val_tf)

train_ds = Subset(train_ds, train_idxs)
val_ds   = Subset(val_ds, val_idxs)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

# -------------------------
# Model (ResNet18)
# -------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)  # 2 classes: Negative, Positive
model = model.to(device)

# Optional: freeze backbone for faster/safer training on small datasets
for name, p in model.named_parameters():
    if not name.startswith("fc."):
        p.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(m):
    m.eval()
    total, correct = 0, 0
    total_loss = 0.0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = m(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

def train_one_epoch(m):
    m.train()
    total, correct = 0, 0
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = m(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

best_val_acc = -1.0

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model)
    va_loss, va_acc = evaluate(model)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
          f"val loss {va_loss:.4f} acc {va_acc:.3f}")

    if va_acc > best_val_acc:
        best_val_acc = va_acc

        # Save full model (for your Streamlit torch.load demo)
        torch.save(model, OUT_DIR / "crack_resnet18.pt")

        # Save TorchScript (portable)
        model_cpu = model.to("cpu").eval()
        example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        scripted = torch.jit.trace(model_cpu, example)
        scripted.save(str(OUT_DIR / "crack_resnet18_scripted.pt"))
        model.to(device)

print("Done. Best val acc:", best_val_acc)
print("Saved:", OUT_DIR / "crack_resnet18.pt")
print("Saved:", OUT_DIR / "crack_resnet18_scripted.pt")
