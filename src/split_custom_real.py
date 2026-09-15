import os
import random
import shutil

SOURCE = "data/custom_real"

TRAIN = "data/custom_real_split/train"
VAL = "data/custom_real_split/val"
TEST = "data/custom_real_split/test"

random.seed(42)

extensions = (".jpg", ".jpeg", ".png", ".webp")

images = [
    f for f in os.listdir(SOURCE)
    if f.lower().endswith(extensions)
]

random.shuffle(images)

total = len(images)

train_count = int(total * 0.80)
val_count = int(total * 0.10)

train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]

for folder in [TRAIN, VAL, TEST]:
    os.makedirs(folder, exist_ok=True)

def copy_images(files, destination):
    for filename in files:
        shutil.copy2(
            os.path.join(SOURCE, filename),
            os.path.join(destination, filename)
        )

copy_images(train_images, TRAIN)
copy_images(val_images, VAL)
copy_images(test_images, TEST)

print("====================================")
print("CUSTOM REAL DATASET SPLIT")
print("====================================")
print("Total:", total)
print("Train:", len(train_images))
print("Validation:", len(val_images))
print("Test:", len(test_images))
print("====================================")