import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/signalscope_efficientnet_b0_custom_real.pth"
IMAGE_DIR = "data/real_world_test"# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# -----------------------------
# Load model
# -----------------------------
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
model.eval()

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Find images
# -----------------------------
valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)

images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(valid_extensions)
]

print("Images found:", len(images))
print("-" * 60)

real_count = 0
fake_count = 0

# -----------------------------
# Predict each image
# -----------------------------
for filename in sorted(images):

    image_path = os.path.join(IMAGE_DIR, filename)

    try:
        image = Image.open(image_path).convert("RGB")

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        fake_probability = probabilities[0][0].item()
        real_probability = probabilities[0][1].item()

        if fake_probability > real_probability:
            prediction = "AI-GENERATED"
            confidence = fake_probability
            fake_count += 1
        else:
            prediction = "REAL"
            confidence = real_probability
            real_count += 1

        print(
            f"{filename:<35} "
            f"{prediction:<15} "
            f"{confidence * 100:.2f}%"
        )

    except Exception as e:
        print(f"{filename} -> ERROR: {e}")

# -----------------------------
# Summary
# -----------------------------
total = real_count + fake_count

print("\n" + "=" * 60)
print("REAL-WORLD TEST RESULTS")
print("=" * 60)

print("Total images       :", total)
print("Predicted REAL     :", real_count)
print("Predicted AI       :", fake_count)

if total > 0:
    print(
        "REAL prediction rate:",
        f"{real_count / total * 100:.2f}%"
    )

    print(
        "False-positive rate :",
        f"{fake_count / total * 100:.2f}%"
    )

print("=" * 60)