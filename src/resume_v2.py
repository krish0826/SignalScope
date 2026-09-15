import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==========================================
# SignalScope V2 — Continue Training
# ==========================================

DATA_DIR = "data/combined_v2"
MODEL_PATH = "models/signalscope_efficientnet_b0_v2.pth"
OUTPUT_PATH = "models/signalscope_efficientnet_b0_v2_final.pth"

BATCH_SIZE = 32
EPOCHS = 3
LR = 0.00005

print("==========================================")
print("SignalScope V2 — Resume Training")
print("==========================================")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU Memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

# ==========================================
# Transforms
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(
        224,
        scale=(0.75, 1.0)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.RandomApply([
        transforms.GaussianBlur(
            kernel_size=3
        )
    ], p=0.20),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.15
    ),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# Dataset
# ==========================================

print()
print("Loading datasets...")

train_dataset = datasets.ImageFolder(
    DATA_DIR + "/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR + "/val",
    transform=val_transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

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

# ==========================================
# Model
# ==========================================

print()
print("Loading previous V2 model...")

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

print("Previous V2 checkpoint loaded!")

# ==========================================
# Training
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

best_val_acc = 0.0

print()
print("==========================================")
print("STARTING CONTINUED TRAINING")
print("==========================================")

for epoch in range(EPOCHS):

    start_time = time.time()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:

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

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / total:.2f}%"
        )

    train_loss = running_loss / len(train_loader)

    train_acc = 100 * correct / total

    # ======================================
    # Validation
    # ======================================

    print()
    print("Starting validation...")

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

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    elapsed = (time.time() - start_time) / 60

    print()
    print("------------------------------------------")
    print(
        f"Epoch {epoch + 1}/{EPOCHS} completed"
    )
    print(
        f"Train Loss: {train_loss:.4f}"
    )
    print(
        f"Train Acc : {train_acc:.2f}%"
    )
    print(
        f"Val Acc   : {val_acc:.2f}%"
    )
    print(
        f"Time      : {elapsed:.1f} minutes"
    )

    if torch.cuda.is_available():
        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"Peak GPU memory: {peak_memory:.2f} GB"
        )

        torch.cuda.reset_peak_memory_stats()

    # ======================================
    # Save best checkpoint
    # ======================================

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            OUTPUT_PATH
        )

        print()
        print("BEST MODEL SAVED!")
        print(
            f"Validation accuracy: {val_acc:.2f}%"
        )

print()
print("==========================================")
print("CONTINUED TRAINING COMPLETE")
print("==========================================")
print("Best validation accuracy:", f"{best_val_acc:.2f}%")
print("Saved model:", OUTPUT_PATH)