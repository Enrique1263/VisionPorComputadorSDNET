import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from pathlib import Path
import random
from tqdm import tqdm

from dataset import CrackDataset, build_samples


def main():
    # ======================
    # CONFIG
    # ======================
    BATCH_SIZE = 32
    NUM_EPOCHS = 2
    LR = 1e-4
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
    # TRANSFORMS
    # ======================
    weights = models.EfficientNet_B0_Weights.DEFAULT
    base_transforms = weights.transforms()

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        base_transforms
    ])

    val_test_transforms = base_transforms

    # ======================
    # DATASET
    # ======================
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"

    samples = build_samples(DATA_DIR)
    random.shuffle(samples)

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
    # MODELO
    # ======================
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 1)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ======================
    # TRAIN
    # ======================
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [TRAIN]")

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

        # VALIDATION
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

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_acc={correct/total:.4f}")

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

    print(f"\n🎯 Test accuracy: {correct/total:.4f}")

    MODEL_PATH = BASE_DIR / "models/efficientnet_crack_best.pth"
    torch.save(model, MODEL_PATH)
    print(f"✅ Pesos guardados en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
