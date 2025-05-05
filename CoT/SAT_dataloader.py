from datasets import load_dataset
import matplotlib.pyplot as plt

# Load dataset
split = "val"
dataset = load_dataset("array/SAT", split="val", cache_dir=r"N:\uni_work\LLVM_project\weights")

# Get the 10th example
example = dataset[split][3000]

# Print the metadata
print("Question:", example['question'])
print("Answers:", example['answers'])
print("Correct Answer:", example['correct_answer'])

# Save the images
for idx, img in enumerate(example['image_bytes']):
    filename = f"image_{idx+1}.jpg"
    img.save(filename)
    print(f"Saved {filename}")

# Display the images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for idx, img in enumerate(example['image_bytes']):
    axes[idx].imshow(img)
    axes[idx].axis('off')
    axes[idx].set_title(f"Image {idx+1}")

plt.suptitle(example['question'])
plt.show()
