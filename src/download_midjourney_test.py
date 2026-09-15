from datasets import load_dataset
from pathlib import Path
from PIL import Image
import io

OUTPUT = Path("data/genimage_midjourney_test")
REAL_DIR = OUTPUT / "REAL"
FAKE_DIR = OUTPUT / "FAKE"

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_REAL = 1000
TARGET_FAKE = 1000

# Count files already downloaded
real_count = len(list(REAL_DIR.glob("*.jpg")))
fake_count = len(list(FAKE_DIR.glob("*.jpg")))

print("Starting Midjourney test download...")
print(f"Existing REAL: {real_count}/{TARGET_REAL}")
print(f"Existing FAKE: {fake_count}/{TARGET_FAKE}")

if real_count >= TARGET_REAL and fake_count >= TARGET_FAKE:
    print("\nAlready complete!")
    exit()

ds = load_dataset(
    "nebula/GenImage-arrow",
    "midjourney-test",
    split="test",
    streaming=True
)

last_real_report = real_count
last_fake_report = fake_count

try:
    for sample in ds:

        label = sample["label"]

        # REAL
        if label == 0 and real_count < TARGET_REAL:

            image = Image.open(
                io.BytesIO(sample["image"])
            ).convert("RGB")

            image.save(
                REAL_DIR / f"real_{real_count:04d}.jpg",
                quality=95
            )

            real_count += 1

            if real_count % 100 == 0 and real_count != last_real_report:
                print(f"REAL: {real_count}/{TARGET_REAL}")
                last_real_report = real_count

        # FAKE
        elif label == 1 and fake_count < TARGET_FAKE:

            image = Image.open(
                io.BytesIO(sample["image"])
            ).convert("RGB")

            image.save(
                FAKE_DIR / f"fake_{fake_count:04d}.jpg",
                quality=95
            )

            fake_count += 1

            if fake_count % 100 == 0 and fake_count != last_fake_report:
                print(f"FAKE: {fake_count}/{TARGET_FAKE}")
                last_fake_report = fake_count

        # Finished
        if real_count >= TARGET_REAL and fake_count >= TARGET_FAKE:
            break

except Exception as e:
    print("\nDownload interrupted because of network/stream error.")
    print("You can safely run this script again.")
    print(f"Current REAL: {real_count}/{TARGET_REAL}")
    print(f"Current FAKE: {fake_count}/{TARGET_FAKE}")
    print("\nError:", e)

print("\n==============================")
print("DOWNLOAD STATUS")
print("==============================")
print(f"REAL: {real_count}/{TARGET_REAL}")
print(f"FAKE: {fake_count}/{TARGET_FAKE}")
print(f"Folder: {OUTPUT}")