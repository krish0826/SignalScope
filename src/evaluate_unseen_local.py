import os
import io
import gc
import torch
import torch.nn as nn
from torchvision import models, transforms
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================================
# SignalScope V2 Final
# Unseen Generator Evaluation
# ==========================================

MODEL_PATH = "models/signalscope_efficientnet_b0_v2_final.pth"

REPO_ID = "nebula/GenImage-arrow"
REVISION = "3f4b9f921a673be09a93b335ed728cea0c6ecf33"

GENERATORS = {
    "SD1.4": (
        "stable_diffusion_v_1_4",
        8
    ),
    "Midjourney": (
        "Midjourney",
        19
    )
}

# ==========================================
# Device
# ==========================================

print("==========================================")
print("SignalScope V2 Final")
print("Unseen Generator Evaluation")
print("==========================================")

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
# Load model
# ==========================================

model = models.efficientnet_b0(
    weights=None
)

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

print("Final V2 model loaded!")

# ==========================================
# Transform
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# Evaluate generator
# ==========================================

def evaluate_generator(
    display_name,
    generator,
    shard_count
):

    print()
    print("==========================================")
    print("Testing:", display_name)
    print("==========================================")

    all_labels = []
    all_predictions = []
    all_fake_probabilities = []

    total_images = 0

    for shard in range(shard_count):

        filename = (
            f"data/test/{generator}/"
            f"data-{shard:05d}-of-{shard_count:05d}.arrow"
        )

        print()
        print(
            f"Shard {shard + 1}/{shard_count}"
        )
        print(filename)

        # --------------------------------------
        # Download one shard
        # --------------------------------------

        try:

            local_file = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                revision=REVISION
            )

        except Exception as e:

            print()
            print("DOWNLOAD ERROR:")
            print(e)

            print()
            print(
                "Skipping this shard and continuing..."
            )

            continue

        print("Shard downloaded.")

        # --------------------------------------
        # Load Arrow dataset
        # --------------------------------------

        try:

            dataset = load_dataset(
                "arrow",
                data_files=local_file,
                split="train"
            )

        except Exception as e:

            print()
            print("ARROW LOAD ERROR:")
            print(e)

            continue

        print(
            "Images in shard:",
            len(dataset)
        )

        # --------------------------------------
        # Process images
        # --------------------------------------

        for item in dataset:

            try:

                image = Image.open(
                    io.BytesIO(item["image"])
                ).convert("RGB")

                image_tensor = transform(
                    image
                ).unsqueeze(0).to(device)

                with torch.no_grad():

                    output = model(
                        image_tensor
                    )

                    probabilities = torch.softmax(
                        output,
                        dim=1
                    )

                # Our model:
                # 0 = FAKE
                # 1 = REAL

                fake_probability = (
                    probabilities[0][0].item()
                )

                prediction = torch.argmax(
                    probabilities,
                    dim=1
                ).item()

                # GenImage:
                # 0 = REAL
                # 1 = FAKE

                genimage_prediction = (
                    1 if prediction == 0 else 0
                )

                label = int(item["label"])

                all_labels.append(label)

                all_predictions.append(
                    genimage_prediction
                )

                all_fake_probabilities.append(
                    fake_probability
                )

                total_images += 1

                if total_images % 1000 == 0:

                    print(
                        "Images processed:",
                        total_images
                    )

            except Exception as e:

                print(
                    "Image processing error:",
                    e
                )

        # --------------------------------------
        # Free shard memory
        # --------------------------------------

        del dataset

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"Finished shard {shard + 1}/{shard_count}"
        )

    # ==========================================
    # Final metrics
    # ==========================================

    if len(all_labels) == 0:

        print()
        print(
            "No images were successfully evaluated."
        )

        return

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
        all_fake_probabilities
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print()
    print("==========================================")
    print("RESULTS:", display_name)
    print("==========================================")

    print(
        f"Images   : {len(all_labels)}"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Macro-F1 : {macro_f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(cm)


# ==========================================
# Run unseen tests
# ==========================================

for display_name, values in GENERATORS.items():

    generator, shard_count = values

    evaluate_generator(
        display_name,
        generator,
        shard_count
    )

print()
print("==========================================")
print("UNSEEN GENERATOR EVALUATION COMPLETE")
print("==========================================")