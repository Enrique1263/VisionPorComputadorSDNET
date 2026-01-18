# ======================
# IMPORTS
# ======================
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from tqdm import tqdm

from dataset import CrackDataset, build_samples
from cnn_crack_custom import CrackCNNCustom
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))



def main():
    # ======================
    # CONFIG
    # ======================
    BATCH_SIZE = 32
    NUM_EPOCHS = 15          # more epochs for scratch training
    LR = 3e-4                # slightly higher LR
    NUM_WORKERS = 1
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ======================
    # DEVICE
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    # ======================
    # TRANSFORMS (CUSTOM CNN)
    # ======================
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ======================
    # DATASET
    # ======================
    BASE_DIR = PROJECT_ROOT
    DATA_DIR = BASE_DIR / "data"

    samples = build_samples(DATA_DIR)
    random.shuffle(samples)

    print(f"Total samples found: {len(samples)}")
    if len(samples) == 0:
        raise RuntimeError("❌ No samples found. Check dataset path and structure.")

    n_total = len(samples)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    train_dataset = CrackDataset(train_samples, transform=train_transforms)
    val_dataset = CrackDataset(val_samples, transform=val_test_transforms)
    test_dataset = CrackDataset(test_samples, transform=val_test_transforms)

    # ======================
    # DATALOADERS
    # ======================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # ======================
    # MODEL (CUSTOM CNN)
    # ======================
    model = CrackCNNCustom().to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    # ======================
    # TRAIN
    # ======================
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [TRAIN]"
        )

        for images, labels in train_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        # ======================
        # VALIDATION
        # ======================
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images).squeeze(1)
                preds = (torch.sigmoid(outputs) >= 0.5).float()

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    # ======================
    # TEST
    # ======================
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images).squeeze(1)
            preds = (torch.sigmoid(outputs) >= 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total if total > 0 else 0.0
    print(f"\n🎯 Test accuracy: {test_acc:.4f}")

    # ======================
    # SAVE MODEL
    # ======================
    MODEL_PATH = BASE_DIR / "models/crack_cnn_custom.pth"
    torch.save(model, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
