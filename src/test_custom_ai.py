import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "models/signalscope_efficientnet_b0_v2_final.pth"
IMAGE_DIR = "data/custom_ai_test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("=" * 60)
print("SignalScope - Custom AI Image Test")
print("=" * 60)

files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

print(f"Images found: {len(files)}")
print()

for filename in files:

    path = os.path.join(IMAGE_DIR, filename)

    try:
        image = Image.open(path).convert("RGB")

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        fake_probability = probabilities[0][0].item()
        real_probability = probabilities[0][1].item()

        if fake_probability > real_probability:
            prediction = "AI-GENERATED"
            confidence = fake_probability
        else:
            prediction = "REAL"
            confidence = real_probability

        print(f"Image      : {filename}")
        print(f"Prediction : {prediction}")
        print(f"REAL       : {real_probability * 100:.6f}%")
        print(f"AI         : {fake_probability * 100:.6f}%")
        print(f"Confidence : {confidence * 100:.6f}%")
        print("-" * 60)

    except Exception as e:
        print(f"ERROR: {filename}")
        print(e)
        print("-" * 60)

print()
print("TEST COMPLETE")