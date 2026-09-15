import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# =========================
# CONFIG
# =========================

TRAIN_DIR = "data/combined_v2/train"
VAL_DIR = "data/combined_v2/val"

MODEL_PATH = "models/signalscope_efficientnet_b0_v2.pth"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU Memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

# =========================
# DATA AUGMENTATION
# =========================

train_transform = transforms.Compose([
    transforms.Resize(256),

    transforms.RandomResizedCrop(
        224,
        scale=(0.70, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.05
    ),

    # Blur only some images
    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.5)
            )
        ],
        p=0.20
    ),

    transforms.ToTensor(),

    transforms.RandomErasing(
        p=0.15,
        scale=(0.02, 0.15)
    ),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASETS
# =========================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# =========================
# DATALOADERS
# =========================

print("\nCreating DataLoaders...")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

# =========================
# TEST FIRST BATCH
# =========================

print("\nTesting first training batch...")

start = time.time()

images, labels = next(iter(train_loader))

print(
    "First batch loaded:",
    images.shape,
    "Labels:",
    labels.shape
)

print(
    "First batch loading time:",
    round(time.time() - start, 2),
    "seconds"
)

# =========================
# MODEL
# =========================

print("\nLoading EfficientNet-B0...")

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model = model.to(device)

for param in model.parameters():
    param.requires_grad = True

print("Model loaded.")

if device.type == "cuda":
    torch.cuda.empty_cache()

# =========================
# LOSS / OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# =========================
# TRAINING
# =========================

best_val_acc = 0.0

print("\n==============================")
print("STARTING V2 TRAINING")
print("==============================\n")

for epoch in range(EPOCHS):

    epoch_start = time.time()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        # Progress every 100 batches
        if (
            batch_idx == 0
            or (batch_idx + 1) % 100 == 0
            or batch_idx + 1 == len(train_loader)
        ):

            current_acc = (
                100.0 * correct / total
            )

            print(
                f"  Batch "
                f"{batch_idx + 1}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f} "
                f"| Acc: {current_acc:.2f}%"
            )

    train_acc = (
        100.0 * correct / total
    )

    # =====================
    # VALIDATION
    # =====================

    print("  Starting validation...")

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_acc = (
        100.0 * val_correct / val_total
    )

    scheduler.step()

    epoch_time = time.time() - epoch_start

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}] "
        f"Complete"
    )

    print(
        f"Loss: {running_loss / len(train_loader):.4f}"
    )

    print(
        f"Train Acc: {train_acc:.2f}%"
    )

    print(
        f"Val Acc: {val_acc:.2f}%"
    )

    print(
        f"Time: {epoch_time / 60:.1f} minutes"
    )

    if device.type == "cuda":

        memory_used = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"Peak GPU memory: "
            f"{memory_used:.2f} GB"
        )

        torch.cuda.reset_peak_memory_stats()

    # =====================
    # SAVE BEST MODEL
    # =====================

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"Best model saved!"
        )

        print(
            f"Validation accuracy: "
            f"{val_acc:.2f}%"
        )

# =========================
# COMPLETE
# =========================

print("\n==============================")
print("V2 TRAINING COMPLETE")
print("==============================")

print(
    "Best validation accuracy:",
    f"{best_val_acc:.2f}%"
)

print(
    "Saved:",
    MODEL_PATH
)