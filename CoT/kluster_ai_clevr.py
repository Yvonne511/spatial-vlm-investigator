from datasets import load_dataset
from PIL import Image
import re
import json
from tqdm import tqdm
import os
import base64
from io import BytesIO
import time
import requests
from openai import OpenAI
from getpass import getpass

# Load the dataset
dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB", split="train")

# Configure batch processing parameters
BATCH_SIZE = 5  # Number of samples to process in each batch
SAVE_PATH = './checkpoints/llama4_scout_eval_CoGenT_ValB.json'
QUESTION_TEMPLATE = "{Question}"

def encode_image_to_base64(image):
    """Convert a PIL image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

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

def extract_answer(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE)
    return match.group(1).strip().lower() if match else text.strip().lower()

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

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "accuracy": correct / len(responses),
                "total": len(responses),
                "correct": correct,
                "results": results
            }, f, indent=2, ensure_ascii=False)
    return correct / len(responses)

def main():
    # Get API key from user input
    api_key = "a5e0b5c4-dee5-402f-acaf-ced3114d5b3c"
    
    # Initialize OpenAI client pointing to kluster.ai API
    client = OpenAI(
        base_url="https://api.kluster.ai/v1",
        api_key=api_key,
    )
    
    responses = []
    ground_truths = []
    problems = []
    
    # Process dataset in batches
    for batch_start in range(0, len(dataset), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(dataset))
        current_batch = dataset[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//BATCH_SIZE + 1}, samples {batch_start} to {batch_end-1}")
        
        # Prepare batch requests
        batch_requests = []
        batch_ground_truths = []
        batch_questions = []
        
        for i, sample in enumerate(current_batch):
            image = sample["image"].convert("RGB")
            question = QUESTION_TEMPLATE.format(Question=sample["problem"])
            gt_answer = sample["solution"].lower()
            
            # Store ground truth and question
            batch_ground_truths.append(gt_answer)
            batch_questions.append(question)
            
            # Create request for this sample
            batch_requests.append({
                "custom_id": f"request-{batch_start + i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Final answer should be a single number, \"yes\" or \"no\" between <answer> </answer> tags."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(image)}"}}
                            ]
                        }
                    ],
                    "max_completion_tokens": 1024
                }
            })
        
        # Save batch requests to a JSONL file
        batch_file_name = f"batch_request_{batch_start}_{batch_end}.jsonl"
        with open(batch_file_name, "w") as file:
            for request in batch_requests:
                file.write(json.dumps(request) + "\n")
        
        # Upload batch job file
        try:
            batch_input_file = client.files.create(
                file=open(batch_file_name, "rb"),
                purpose="batch"
            )
            
            # Submit batch job
            batch_request = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            
            print(f"Submitted batch job with ID: {batch_request.id}")
            
            # Poll the batch status until it's complete
            while True:
                batch_status = client.batches.retrieve(batch_request.id)
                print(f"Batch status: {batch_status.status}")
                print(
                    f"Completed tasks: {batch_status.request_counts.completed} / {batch_status.request_counts.total}"
                )
                if batch_status.status.lower() in ["completed", "failed", "cancelled"]:
                    break
                time.sleep(30)  # Wait before checking again
            
            # Process batch results
            if batch_status.status.lower() == "completed":
                # Retrieve the results
                result_file_id = batch_status.output_file_id
                results_content = client.files.content(result_file_id).content
                
                # Save results to a file
                result_file_name = f"batch_results_{batch_start}_{batch_end}.jsonl"
                with open(result_file_name, "wb") as file:
                    file.write(results_content)
                
                # Parse results
                batch_responses = []
                results_list = [json.loads(line) for line in results_content.decode().splitlines()]
                
                # Sort results by custom_id to maintain order
                results_list.sort(key=lambda x: int(x["custom_id"].split("-")[1]))
                
                for result in results_list:
                    if "body" in result and "choices" in result["body"]:
                        answer = result["body"]["choices"][0]["message"]["content"]
                        batch_responses.append(answer)
                    else:
                        print(f"Missing data in result: {result}")
                        batch_responses.append("error")
                
                # Add batch results to overall results
                responses.extend(batch_responses)
                ground_truths.extend(batch_ground_truths)
                problems.extend(batch_questions)
                
                # Calculate intermediate accuracy
                accuracy = compute_accuracy(responses, ground_truths, problems, SAVE_PATH)
                print(f"Processed {len(responses)}/{len(dataset)} samples. Current accuracy: {accuracy:.2%}")
            else:
                print(f"Batch failed with status: {batch_status.status}")
                # Add error placeholders for this batch
                responses.extend(["error"] * len(batch_requests))
                ground_truths.extend(batch_ground_truths)
                problems.extend(batch_questions)
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Add error placeholders for this batch
            responses.extend(["error"] * len(batch_requests))
            ground_truths.extend(batch_ground_truths)
            problems.extend(batch_questions)
        
        # Clean up batch files
        try:
            os.remove(batch_file_name)
        except:
            pass
    
    # Calculate final accuracy
    final_accuracy = compute_accuracy(responses, ground_truths, problems, SAVE_PATH)
    print(f"Final accuracy: {final_accuracy:.2%}")

if __name__ == "__main__":
    main()