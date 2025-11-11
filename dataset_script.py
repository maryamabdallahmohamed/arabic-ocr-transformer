# from datasets import load_dataset
# import base64
# import json
# from pathlib import Path

# # Load dataset
# dataset = load_dataset("riotu-lab/SARD", split="Amiri")

# # Create output folders
# output_dir = Path("SARD_Amiri")
# images_dir = output_dir / "images"
# images_dir.mkdir(parents=True, exist_ok=True)

# ground_truth = []

# for item in dataset:
#     img_name = item['image_name']
#     img_base64 = item['image_base64'].strip('"')
#     img_bytes = base64.b64decode(img_base64)
    
#     # Save image
#     with open(images_dir / img_name, "wb") as f:
#         f.write(img_bytes)
    
#     # Save ground truth info
#     ground_truth.append({
#         "image_name": img_name,
#         "text": item["chunk"],
#         "font": item["font_name"]
#     })

# # Save ground truth JSON
# with open(output_dir / "ground_truth.json", "w", encoding="utf-8") as f:
#     json.dump(ground_truth, f, ensure_ascii=False, indent=2)


from datasets import load_dataset
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Load dataset with streaming enabled
ds = load_dataset("riotu-lab/SARD-Extended", streaming=True)
print(ds)

# Iterate over a specific font dataset (e.g., Amiri)
for sample in ds["Amiri"]:
    image_name = sample["image_name"]
    chunk = sample["chunk"]  # Arabic text transcription
    font_name = sample["font_name"]
    
    # Decode Base64 image
    image_data = base64.b64decode(sample["image_base64"])
    image = Image.open(BytesIO(image_data))

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Font: {font_name}")
    plt.show()

    # Print the details
    print(f"Image Name: {image_name}")
    print(f"Font Name: {font_name}")
    print(f"Text Chunk: {chunk}")
    
    # Break after one sample for testing
    break
