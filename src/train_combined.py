import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# =====================================
# SETTINGS
# =====================================

DATA_DIR = "data/combined"

BATCH_SIZE = 32
EPOCHS = 5
LR = 5e-5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# =====================================
# TRANSFORMS
# =====================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        224,
        scale=(0.75, 1.0)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.15,
        hue=0.03
    ),

    transforms.RandomApply([
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.5)
        )
    ], p=0.2),

    transforms.ToTensor(),

    transforms.RandomErasing(
        p=0.15,
        scale=(0.02, 0.10),
        ratio=(0.3, 3.3)
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

# =====================================
# DATASETS
# =====================================

train_dataset = datasets.ImageFolder(
    f"{DATA_DIR}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    f"{DATA_DIR}/val",
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

# =====================================
# MODEL
# =====================================

model = models.efficientnet_b0(
    weights="DEFAULT"
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

# Start from our robust model
previous_model = (
    "models/"
    "signalscope_efficientnet_b0_robust.pth"
)

model.load_state_dict(
    torch.load(
        previous_model,
        map_location=DEVICE
    )
)

print("Robust model loaded!")

# Fine-tune the entire network
for param in model.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# =====================================
# LOSS / OPTIMIZER
# =====================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# =====================================
# TRAINING
# =====================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

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

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    scheduler.step()

    train_accuracy = (
        100 * correct / total
    )

    # =================================
    # VALIDATION
    # =================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = (
        100 * val_correct / val_total
    )

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: "
        f"{running_loss / len(train_loader):.4f} "
        f"Train Acc: "
        f"{train_accuracy:.2f}% "
        f"Val Acc: "
        f"{val_accuracy:.2f}%"
    )

# =====================================
# SAVE
# =====================================

output_path = (
    "models/"
    "signalscope_efficientnet_b0_custom_real.pth"
)

torch.save(
    model.state_dict(),
    output_path
)

print("\n====================================")
print("TRAINING COMPLETE")
print("====================================")
print("Saved:", output_path)