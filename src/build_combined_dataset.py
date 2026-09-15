import os
import shutil
import random

# -----------------------------------
# SOURCE DATA
# -----------------------------------

CIFAKE_TRAIN_REAL = "data/train/REAL"
CIFAKE_TRAIN_FAKE = "data/train/FAKE"

CIFAKE_VAL_REAL = "data/val/REAL"
CIFAKE_VAL_FAKE = "data/val/FAKE"

CUSTOM_TRAIN_REAL = "data/custom_real_split/train"
CUSTOM_VAL_REAL = "data/custom_real_split/val"

# -----------------------------------
# OUTPUT DATASET
# -----------------------------------

OUTPUT_TRAIN_REAL = "data/combined/train/REAL"
OUTPUT_TRAIN_FAKE = "data/combined/train/FAKE"

OUTPUT_VAL_REAL = "data/combined/val/REAL"
OUTPUT_VAL_FAKE = "data/combined/val/FAKE"

for folder in [
    OUTPUT_TRAIN_REAL,
    OUTPUT_TRAIN_FAKE,
    OUTPUT_VAL_REAL,
    OUTPUT_VAL_FAKE
]:
    os.makedirs(folder, exist_ok=True)


def get_images(folder):
    extensions = (".jpg", ".jpeg", ".png", ".webp")

    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ]


def copy_images(source, destination, prefix):
    images = get_images(source)

    for i, filename in enumerate(images):
        source_path = os.path.join(source, filename)

        # Prefix prevents filename collisions
        new_name = f"{prefix}_{i}_{filename}"

        destination_path = os.path.join(
            destination,
            new_name
        )

        shutil.copy2(source_path, destination_path)

    return len(images)


# -----------------------------------
# BUILD TRAINING DATA
# -----------------------------------

print("Building combined TRAIN dataset...")

train_cifake_real = copy_images(
    CIFAKE_TRAIN_REAL,
    OUTPUT_TRAIN_REAL,
    "cifake_real"
)

train_custom_real = copy_images(
    CUSTOM_TRAIN_REAL,
    OUTPUT_TRAIN_REAL,
    "phone_real"
)

train_fake = copy_images(
    CIFAKE_TRAIN_FAKE,
    OUTPUT_TRAIN_FAKE,
    "cifake_fake"
)


# -----------------------------------
# BUILD VALIDATION DATA
# -----------------------------------

print("Building combined VALIDATION dataset...")

val_cifake_real = copy_images(
    CIFAKE_VAL_REAL,
    OUTPUT_VAL_REAL,
    "cifake_val_real"
)

val_custom_real = copy_images(
    CUSTOM_VAL_REAL,
    OUTPUT_VAL_REAL,
    "phone_val_real"
)

val_fake = copy_images(
    CIFAKE_VAL_FAKE,
    OUTPUT_VAL_FAKE,
    "cifake_val_fake"
)


# -----------------------------------
# RESULTS
# -----------------------------------

print()
print("=" * 50)
print("COMBINED DATASET CREATED")
print("=" * 50)

print("\nTRAIN")
print("CIFAKE REAL :", train_cifake_real)
print("PHONE REAL  :", train_custom_real)
print("TOTAL REAL  :", train_cifake_real + train_custom_real)
print("FAKE        :", train_fake)

print("\nVALIDATION")
print("CIFAKE REAL :", val_cifake_real)
print("PHONE REAL  :", val_custom_real)
print("TOTAL REAL  :", val_cifake_real + val_custom_real)
print("FAKE        :", val_fake)

print("=" * 50)