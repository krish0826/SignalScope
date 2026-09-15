import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ==============================
# SETTINGS
# ==============================

IMAGE_PATH = "data/test/REAL/0000 (10).jpg"
MODEL_PATH = "models/signalscope_efficientnet_b0_robust.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# LOAD MODEL
# ==============================

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


# ==============================
# IMAGE
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(IMAGE_PATH).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(device)


# ==============================
# PREDICTION
# ==============================

with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

confidence = probabilities[0][predicted_class].item()

class_names = ["FAKE", "REAL"]

print("================================")
print("SignalScope Grad-CAM Test")
print("================================")

print("Prediction:", class_names[predicted_class])
print(f"Confidence: {confidence * 100:.2f}%")


# ==============================
# GRAD-CAM
# ==============================

# Last convolutional layer
target_layers = [
    model.features[-1]
]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

targets = [
    ClassifierOutputTarget(predicted_class)
]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)

grayscale_cam = grayscale_cam[0]


# ==============================
# CREATE HEATMAP
# ==============================

import numpy as np

rgb_image = np.array(
    image.resize((224, 224))
).astype(np.float32) / 255.0

visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)

output_path = "models/gradcam_result.jpg"

Image.fromarray(visualization).save(
    output_path
)

print()
print("Grad-CAM generated successfully!")
print("Saved:", output_path)

print("================================")