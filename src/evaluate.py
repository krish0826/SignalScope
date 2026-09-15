import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("================================")
print("SignalScope V2 Final Evaluation")
print("================================")

# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# ==========================================
# Test data
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    "data/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print()
print("Test images:", len(test_dataset))
print("Classes:", test_dataset.classes)

# ==========================================
# Load FINAL V2 model
# ==========================================

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

MODEL_PATH = (
    "models/"
    "signalscope_efficientnet_b0_v2_final.pth"
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print()
print("Final V2 model loaded!")
print("Model:", MODEL_PATH)

# ==========================================
# Prediction
# ==========================================

all_labels = []
all_probabilities = []
all_predictions = []

print()
print("Starting prediction...")

with torch.no_grad():

    for batch_number, (images, labels) in enumerate(
        test_loader,
        start=1
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        # ImageFolder:
        # 0 = FAKE
        # 1 = REAL

        all_labels.extend(
            labels.numpy()
        )

        # REAL probability
        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions
            .cpu()
            .numpy()
        )

        if batch_number % 100 == 0:
            processed = min(
                batch_number * 32,
                len(test_dataset)
            )

            print(
                f"Processed: "
                f"{processed}/"
                f"{len(test_dataset)}"
            )

# ==========================================
# Metrics
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro"
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

# ==========================================
# False Positive Rate
# ==========================================

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)

# ==========================================
# Results
# ==========================================

print()
print("================================")
print("FINAL V2 TEST RESULTS")
print("================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Macro-F1 : {macro_f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"FPR      : {fpr:.4f}"
)

print()
print("Confusion Matrix:")
print(cm)

print()
print("================================")
print("Evaluation Completed!")
print("================================")