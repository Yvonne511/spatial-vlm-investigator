from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image, ImageChops
import imagehash
from tqdm import tqdm

# Step 1: Load datasets
val_a = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_70K_Complex", split="train")
depth = load_dataset("erkam/clevr-with-depth")

# Step 2: Hash map builder
def get_hash_map(dataset, resize=(320, 320)):
    hash_map = {}
    for example in tqdm(dataset, desc="Hashing images"):
        img = example['image'].convert("RGB").resize(resize)
        img_hash = str(imagehash.phash(img))
        hash_map[img_hash] = example
    return hash_map

# Step 3: Build hash map across all splits
depth_hashes = {}
for split in ["train", "test", "val"]:
    depth_hashes.update(get_hash_map(depth[split]))

# Step 4: Fuse datasets with hash match and image verification
def images_are_identical(img1, img2):
    # Compares pixels directly; assumes both are RGB
    diff = ImageChops.difference(img1, img2)
    return not diff.getbbox()  # True if no difference

fused_data = []
for item in tqdm(val_a, desc="Matching and fusing"):
    img = item["image"].convert("RGB")
    hash_val = str(imagehash.phash(img))

    if hash_val in depth_hashes:
        matched = depth_hashes[hash_val]
        matched_img = matched["image"].convert("RGB")

        if images_are_identical(img, matched_img):
            fused_data.append({
                "image": img,
                "verify_image": matched_img,  # Or keep only one if redundant
                "question": item["question"],
                "answer": item["answer"],
                "prompt": matched["prompt"],
                "depth_image": matched["depth"]
            })

# Step 5: Save and upload to Hugging Face Hub
print("Total fused examples:", len(fused_data))
dataset = Dataset.from_list(fused_data)
dataset_dict = DatasetDict({"train": dataset})
dataset_dict.push_to_hub("Yvonne511/Clevr_CoGenT_ValA_w_depth_prompt")

