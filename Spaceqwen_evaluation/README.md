# 🚀 SpaceQwen VQA Batch Evaluation

This script is used to batch evaluate the Qwen2.5-VL-3B-Instruct vision-language model on different datasets (CLEVR or SAT). It supports batch inference, result statistics, and saving results to JSON files.

## 🛠️ Requirements

- 🐍 Python 3.8+
- 🔥 torch
- 🤗 transformers
- ⏳ tqdm
- 🖼️ PIL (Pillow)
- 📚 datasets
- 🛠️ (Custom) `data/process_dataset.py` and related dataset files

Install dependencies (if not already installed):

```bash
pip install torch transformers tqdm pillow datasets
```

Install Qwen2.5 package for huggingface

```bash
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]==0.0.8
```

## 🚦 Usage

### 1️⃣ Prepare Data

- 📥 For CLEVR: The script will automatically download the dataset from HuggingFace.
- 📂 For SAT: Place your SAT JSON and image files in the `data/SAT/` directory, or specify their paths with arguments.

### 2️⃣ Run the Script

```bash
python test_spaceqwen_vqa_batch.py [OPTIONS]
```

#### ⚙️ Main Options

| Option                | Description                                                                                  | Default         |
|-----------------------|----------------------------------------------------------------------------------------------|-----------------|
| `--dataset`           | Dataset to test: `CLEVR` or `SAT`                                                            | CLEVR           |
| `--samples`           | Number of samples to test                                                                    | 20              |
| `--batch_size`        | Batch size for inference                                                                     | 1               |
| `--threshold`         | Threshold for small error in counting questions                                              | 1               |
| `--counting_only`     | Only test counting questions (flag, no value needed)                                         | False           |
| `--start_sample`      | Start testing from which sample (index)                                                      | 0               |
| `--sat_json`          | Path to SAT dataset JSON file (if not provided, uses `data/SAT/SAT_train_15000.json`)        | None            |
| `--sat_image_dir`     | Path to SAT image directory (if not provided, uses `data/SAT/`)                              | None            |

#### 🧪 Example: Test CLEVR dataset

```bash
python test_spaceqwen_vqa_batch.py --dataset CLEVR --samples 100 --batch_size 4
```

#### 🧪 Example: Test SAT dataset

```bash
python test_spaceqwen_vqa_batch.py --dataset SAT --sat_json data/SAT/SAT_train_15000.json --sat_image_dir data/SAT/ --samples 50
```

### 3️⃣ Output

- 📊 Results and statistics will be saved in the `results/` directory as JSON files.
- 💾 Interim results (every 10 samples) are saved in `results/interim/`.

### 4️⃣ Notes

- 🗂️ The script will automatically create `models/`, `data/`, and `results/` directories if they do not exist.
- 🧩 Make sure your custom dataset loader and processing functions are available in `data/process_dataset.py`.
- 🤗 The script uses the HuggingFace model `"Qwen/Qwen2.5-VL-3B-Instruct"` by default.

---

## 📁 File Structure

```
SpaceQwen_evaluation/
  test_spaceqwen_vqa_batch.py
  data/
    process_dataset.py
    SAT/
      SAT_train_15000.json
      [SAT images...]
  models/
  results/
```

---

## 📬 Contact

For questions or issues, please open an issue in this repository.
