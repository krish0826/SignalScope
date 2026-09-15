import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

# =========================================
# SignalScope V2 — Unseen Generator Test
# =========================================

print("==========================================")
print("SignalScope V2 — Unseen Generator Test")
print("==========================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================
# Load model
# =========================================

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model.load_state_dict(
    torch.load(
        "models/signalscope_efficientnet_b0_v2.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("V2 model loaded!")

# =========================================
# Image preprocessing
# =========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================
# Evaluate one generator
# =========================================

def evaluate_generator(config):

    print()
    print("==========================================")
    print("Generator:", config)
    print("==========================================")

    dataset = load_dataset(
        "nebula/GenImage-arrow",
        config,
        split="test",
        streaming=True
    )

    labels = []
    probabilities = []
    predictions = []

    count = 0

    with torch.no_grad():

        for item in dataset:

            image = Image.open(
                io.BytesIO(item["image"])
            ).convert("RGB")

            image = transform(image).unsqueeze(0).to(device)

            output = model(image)

            prob = torch.softmax(output, dim=1)

            prediction = torch.argmax(prob, dim=1).item()

            # GenImage label:
            # 0 = REAL
            # 1 = FAKE
            label = item["label"]

            # Our model:
            # 0 = FAKE
            # 1 = REAL
            fake_probability = prob[0][0].item()

            labels.append(label)
            probabilities.append(fake_probability)

            # Convert our prediction to GenImage convention
            genimage_prediction = 1 if prediction == 0 else 0

            predictions.append(genimage_prediction)

            count += 1

            if count % 1000 == 0:
                print("Images processed:", count)

    accuracy = accuracy_score(labels, predictions)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    print()
    print("Results:", config)
    print("------------------------------------------")
    print(f"Images   : {count}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print()
    print("Confusion Matrix")
    print(cm)


# =========================================
# UNSEEN GENERATORS
# =========================================

evaluate_generator("sd14-test")

evaluate_generator("midjourney-test")

print()
print("==========================================")
print("UNSEEN GENERATOR TEST COMPLETE")
print("==========================================")