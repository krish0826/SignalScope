import os
import shutil
import random

random.seed(42)

# -----------------------------
# Paths
# -----------------------------
output = "data/combined_v2"

cifake_train = "data/train"
cifake_val = "data/val"

custom_train = "data/custom_real_split/train"
custom_val = "data/custom_real_split/val"

genimage_fake = "data/genimage_fake"

# -----------------------------
# Create folders
# -----------------------------
for split in ["train", "val"]:
    for label in ["FAKE", "REAL"]:
        os.makedirs(
            os.path.join(output, split, label),
            exist_ok=True
        )

# -----------------------------
# Helper
# -----------------------------
def copy_images(source, destination, limit=None, prefix=""):
    files = [
        f for f in os.listdir(source)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    random.shuffle(files)

    if limit:
        files = files[:limit]

    for i, filename in enumerate(files):
        src = os.path.join(source, filename)
        dst = os.path.join(
            destination,
            f"{prefix}{i:06d}_{filename}"
        )
        shutil.copy2(src, dst)

    return len(files)


# ============================================================
# TRAINING DATA
# ============================================================

print("\n=== BUILDING TRAINING DATA ===")

# CIFAKE FAKE
n = copy_images(
    os.path.join(cifake_train, "FAKE"),
    os.path.join(output, "train", "FAKE"),
    prefix="cifake_fake_"
)
print("CIFAKE FAKE:", n)

# CIFAKE REAL
n = copy_images(
    os.path.join(cifake_train, "REAL"),
    os.path.join(output, "train", "REAL"),
    prefix="cifake_real_"
)
print("CIFAKE REAL:", n)

# Custom REAL
n = copy_images(
    custom_train,
    os.path.join(output, "train", "REAL"),
    prefix="phone_real_"
)
print("Phone REAL:", n)

# GenImage FAKE
limits = {
    "BigGAN": 5000,
    "GLIDE": 1592,
    "ADM": 855,
    "WUKONG": 641,
    "SD15": 549,
    "VQDM": 294
}

for generator, limit in limits.items():

    source = os.path.join(
        genimage_fake,
        generator
    )

    if not os.path.exists(source):
        print("Missing:", generator)
        continue

    n = copy_images(
        source,
        os.path.join(output, "train", "FAKE"),
        limit=limit,
        prefix=f"{generator.lower()}_"
    )

    print(f"{generator} FAKE:", n)


# ============================================================
# VALIDATION DATA
# ============================================================

print("\n=== BUILDING VALIDATION DATA ===")

# CIFAKE validation FAKE
n = copy_images(
    os.path.join(cifake_val, "FAKE"),
    os.path.join(output, "val", "FAKE"),
    prefix="cifake_fake_"
)
print("Validation FAKE:", n)

# CIFAKE validation REAL
n = copy_images(
    os.path.join(cifake_val, "REAL"),
    os.path.join(output, "val", "REAL"),
    prefix="cifake_real_"
)
print("Validation REAL:", n)

# Custom phone validation REAL
n = copy_images(
    custom_val,
    os.path.join(output, "val", "REAL"),
    prefix="phone_real_"
)
print("Validation phone REAL:", n)


# ============================================================
# SUMMARY
# ============================================================

print("\n=== V2 DATASET COMPLETE ===")

for split in ["train", "val"]:
    for label in ["FAKE", "REAL"]:

        path = os.path.join(output, split, label)

        count = len([
            f for f in os.listdir(path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            )
        ])

        print(f"{split}/{label}: {count}")