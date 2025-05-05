# Download the dataset parquet and rename it
# wget -O SAT_val.parquet "https://huggingface.co/datasets/array/SAT/resolve/main/SAT_val.parquet?download=true"

# Create the dataset directory
mkdir -p SAT_images_val

# Process the dataset
python process_dataset.py -- fold val