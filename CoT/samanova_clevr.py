from datasets import load_dataset
from PIL
import re
import json
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Configure model and processor
MODEL_NAME = "sambanova/llama-vision-instruct-3.2-11b"  # Update with the actual model repo name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME).to(device)

batch_size = 5  # Used only for progress tracking

dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB", split="train")

def encode_image_to_bytes(image):
    """Convert a PIL image to bytes"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()

def call_llama_vision_api(image, prompt):
    """Call LLaMA Vision model with an image and prompt"""
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    try:
        outputs = model.generate(**inputs, max_new_tokens=256)
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        print(f"Error during LLaMA Vision API call: {e}")
        return None

def extract_answer(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text)
    return match.group(1).strip().lower() if match else text.strip().lower()

def normalize_answer(text):
    text = text.lower().strip()
    if "\n" in text:
        text = text.split("\n")[-1].strip()

    numbers = re.findall(r"\d+", text)
    if numbers:
        return numbers[0]
    if "yes" in text:
        return "yes"
    elif "no" in text:
        return "no"
    return text

def compute_accuracy(responses, ground_truths, questions=None, output_file=None):
    correct = 0
    results = []
    for i, (pred, gt) in enumerate(zip(responses, ground_truths)):
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(extract_answer(gt))
        is_correct = pred_norm == gt_norm
        if is_correct:
            correct += 1

        item = {
            "id": i,
            "prediction": pred,
            "prediction_normalized": pred_norm,
            "ground_truth": gt,
            "ground_truth_normalized": gt_norm,
            "is_correct": is_correct,
        }
        if questions:
            item["question"] = questions[i]
        results.append(item)
        print(item)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "accuracy": correct / len(responses),
                "total": len(responses),
                "correct": correct,
                "results": results
            }, f, indent=2, ensure_ascii=False)
    return correct / len(responses)

# Evaluation loop
responses, ground_truths, problems = [], [], []
QUESTION_TEMPLATE = "{Question}. Final answer should be a single number, \"yes\", or \"no\" between <Answer> </Answer> tags"
save_path = './checkpoints/internvl3_llama_vision_eval_CoGenT_ValB.json'

for i in tqdm(range(0, len(dataset))):
    sample = dataset[i]
    image = sample["image"].convert("RGB")
    question = QUESTION_TEMPLATE.format(Question=sample["problem"])
    gt_answer = sample["solution"].lower()

    # Call LLaMA Vision model
    pred = call_llama_vision_api(image, question)

    responses.append(pred)
    ground_truths.append(gt_answer)
    problems.append(question)

    # Save intermediate results every batch_size samples
    if (i + 1) % batch_size == 0:
        accuracy = compute_accuracy(responses, ground_truths, problems, save_path)
        print(f"Processed {i+1}/{len(dataset)} samples. Current accuracy: {accuracy:.2%}")

accuracy = compute_accuracy(responses, ground_truths, problems, save_path)
print(f"Final accuracy: {accuracy:.2%}")
