from datasets import Dataset
from PIL import Image
from io import BytesIO
import os

arrow_path = r"C:\Users\krish panchal\.cache\huggingface\hub\datasets--nebula--GenImage-arrow\snapshots\3f4b9f921a673be09a93b335ed728cea0c6ecf33\data\train\VQDM\data-00073-of-00074.arrow"

output_dir = r"data\genimage_fake\VQDM"
os.makedirs(output_dir, exist_ok=True)

print("Loading VQDM shard...")
ds = Dataset.from_file(arrow_path)

count = 0

for item in ds:
    if item["label"] == 1:
        image = Image.open(BytesIO(item["image"])).convert("RGB")
        image.save(
            os.path.join(output_dir, f"vqdm_{count:05d}.jpg"),
            quality=95
        )

        count += 1

        if count >= 5000:
            break

print(f"VQDM FAKE images extracted: {count}")