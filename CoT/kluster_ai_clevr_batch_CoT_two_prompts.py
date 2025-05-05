from datasets import load_dataset
from PIL import Image
import re
import json
from tqdm import tqdm
import os
import base64
from io import BytesIO
import requests
import time
from openai import OpenAI
from getpass import getpass

# Get API key from user input (or set it directly)
KLUSTER_API_KEY = "6db1741a-baa8-4d4a-a26b-730579f558e5"

# Initialize OpenAI client pointing to kluster.ai API
client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=KLUSTER_API_KEY,
)

# Load dataset
dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB", split="train")

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# Configuration
batch_size = 1000  # Number of samples to process in each batch
max_batches = None  # Set to a number to limit the batches processed, None for all
save_path = f'./checkpoints/{MODEL_NAME.split("/")[1]}_CoT_two_prompts.json'

# Create checkpoints directory if it doesn't exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

def encode_image_to_base64(image):
    """Convert a PIL image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def extract_answer(text):
    """Extract answer from between <answer> tags"""
    match = re.search(r"<[aA]nswer>\s*(.*?)\s*</[aA]nswer>", text, re.IGNORECASE)
    return match.group(1).strip().lower() if match else text.strip().lower()

def normalize_answer(text):
    """Normalize answer to handle different formats"""
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

def compute_accuracy(results):
    """Compute accuracy from results"""
    correct = sum(1 for item in results if item["is_correct"])
    return correct / len(results) if results else 0

def process_batch(batch_index, start_idx, end_idx):
    """Process a batch of samples and return as batch request for the two-step approach"""
    first_step_requests = []
    ground_truths = []
    problems = []
    # Note: We'll create second_step_requests later after getting descriptions
    
    # Prepare batch requests with tqdm
    for i in tqdm(range(start_idx, min(end_idx, len(dataset))), 
                  desc=f"Preparing batch {batch_index+1}", 
                  unit="sample"):
        sample = dataset[i]
        # Using dataset index as the ID
        sample_id = i
        image = sample["image"].convert("RGB")
        question = sample["problem"]
        gt_answer = sample["solution"].lower()
        
        # Encode image to base64
        image_base64 = encode_image_to_base64(image)
        
        # First step: Describe objects in the image
        first_step_request = {
            "custom_id": f"sample-{sample_id}-step1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": f"{MODEL_NAME}",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can see and describe images accurately."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the objects in the image: their size, color, shape, location, relative location, reflectance (glossy or matte), and material properties (glossy is metallic, matte is rubber)"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                "max_completion_tokens": 3000,
                "temperature": 0.0
            }
        }
        
        first_step_requests.append(first_step_request)
        ground_truths.append(gt_answer)
        problems.append(question)
    
    return first_step_requests, ground_truths, problems

def update_results_file(all_results, file_path):
    """Update the results JSON file with current results"""
    accuracy = compute_accuracy(all_results)
    data = {
        "accuracy": accuracy,
        "total": len(all_results),
        "correct": sum(1 for item in all_results if item["is_correct"]),
        "results": all_results,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processed_batches": len(all_results) // batch_size + (1 if len(all_results) % batch_size else 0)
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return accuracy

# Main execution
def main():
    total_samples = len(dataset)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    
    print(f"Processing {total_samples} samples in {num_batches} batches using two-step approach")
    print(f"Step 1: Describe image objects with properties (size, color, shape, etc.)")
    print(f"Step 2: Answer question based on description")
    print(f"Results will be saved to {save_path}")
    
    # Initialize or load existing results
    all_results = []
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if "results" in existing_data:
                    all_results = existing_data["results"]
                    print(f"Loaded {len(all_results)} existing results from {save_path}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    
    # Calculate starting batch based on existing results
    already_processed = len(all_results)
    start_batch = already_processed // batch_size
    
    # Use tqdm for overall batch progress
    batch_progress = tqdm(range(start_batch, num_batches), 
                        desc="Overall progress", 
                        unit="batch",
                        position=0)
    
    for batch_idx in batch_progress:
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)
        
        batch_progress.set_description(f"Processing batch {batch_idx+1}/{num_batches}")
        
        # Process batch - First Step
        first_step_requests, ground_truths, problems = process_batch(batch_idx, start_idx, end_idx)
        
        try:
            # Create batch content for first step in memory
            batch_content = BytesIO()
            for request in first_step_requests:
                batch_content.write((json.dumps(request) + "\n").encode('utf-8'))
            
            batch_content.seek(0)
            
            # Upload batch job file for first step
            print(f"Uploading batch data for first step (description)...")
            batch_input_file = client.files.create(
                file=batch_content,
                purpose="batch"
            )
            
            # Submit batch job for first step
            print("Submitting batch job for first step...")
            batch_request = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            
            # Poll the batch status until it's complete
            print(f"Monitoring batch job {batch_request.id} for first step...")
            status_progress = tqdm(desc="First step batch processing", position=1)
            
            while True:
                time.sleep(5)
                batch_status = client.batches.retrieve(batch_request.id)
                completed = batch_status.request_counts.completed
                total = batch_status.request_counts.total
                
                status_progress.total = total
                status_progress.n = completed
                status_progress.set_description(f"Status: {batch_status.status}")
                status_progress.set_postfix(completed=f"{completed}/{total}")
                status_progress.update(0)
                
                if batch_status.status.lower() in ["completed", "failed", "cancelled"]:
                    break
            
            status_progress.close()
            
            # Check if the first step Batch completed successfully
            if batch_status.status.lower() == "completed":
                # Retrieve the first step results
                result_file_id = batch_status.output_file_id
                raw_results = client.files.content(result_file_id).content
                
                # Process first step results
                lines = raw_results.decode('utf-8').splitlines()
                image_descriptions = []
                
                for i, line in enumerate(tqdm(lines, desc="Processing first step results", position=1)):
                    result_json = json.loads(line)
                    sample_idx = start_idx + i
                    
                    try:
                        if "error" in result_json:
                            print(f"Error in first step for sample {sample_idx}: {result_json['error']}")
                            image_descriptions.append("Error getting image description.")
                        elif "response" in result_json and "body" in result_json["response"] and "choices" in result_json["response"]["body"]:
                            # Extract content from the correct JSON structure
                            description = result_json["response"]["body"]["choices"][0]["message"]["content"]
                            image_descriptions.append(description)
                        else:
                            print(f"Unexpected response format for first step sample {sample_idx}: {result_json}")
                            image_descriptions.append("Error getting image description.")
                    except Exception as e:
                        print(f"Exception processing first step result for sample {sample_idx}: {str(e)}")
                        image_descriptions.append("Error getting image description.")
                
                # Now prepare second step requests with the descriptions from first step
                second_step_requests = []
                
                for i in tqdm(range(len(image_descriptions)), desc="Preparing second step requests", position=1):
                    sample_idx = start_idx + i
                    sample = dataset[sample_idx]
                    # Using dataset index as the ID
                    sample_id = sample_idx
                    image = sample["image"].convert("RGB")
                    question = sample["problem"]
                    image_description = image_descriptions[i]
                    
                    # Encode image to base64 again
                    image_base64 = encode_image_to_base64(image)
                    
                    # Second step: Ask the original question with the description context
                    second_step_request = {
                        "custom_id": f"sample-{sample_id}-step2",
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
                                        {"type": "text", "text": "Describe the objects in the image: their size, color, shape, location, relative location, reflectance (glossy or matte), and material properties (glossy is metallic, matte is rubber)"},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                                    ]
                                },
                                {
                                    "role": "assistant",
                                    "content": image_description
                                },
                                {
                                    "role": "user",
                                    "content": question
                                }
                            ],
                            "max_completion_tokens": 3000
                        }
                    }
                    
                    second_step_requests.append(second_step_request)
                
                # Create batch content for second step in memory
                batch_content = BytesIO()
                for request in second_step_requests:
                    batch_content.write((json.dumps(request) + "\n").encode('utf-8'))
                
                batch_content.seek(0)
                
                # Upload batch job file for second step
                print(f"Uploading batch data for second step (question answering)...")
                batch_input_file = client.files.create(
                    file=batch_content,
                    purpose="batch"
                )
                
                # Submit batch job for second step
                print("Submitting batch job for second step...")
                batch_request = client.batches.create(
                    input_file_id=batch_input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                
                # Poll the batch status until it's complete
                print(f"Monitoring batch job {batch_request.id} for second step...")
                status_progress = tqdm(desc="Second step batch processing", position=1)
                
                while True:
                    time.sleep(5)
                    batch_status = client.batches.retrieve(batch_request.id)
                    completed = batch_status.request_counts.completed
                    total = batch_status.request_counts.total
                    
                    status_progress.total = total
                    status_progress.n = completed
                    status_progress.set_description(f"Status: {batch_status.status}")
                    status_progress.set_postfix(completed=f"{completed}/{total}")
                    status_progress.update(0)
                    
                    if batch_status.status.lower() in ["completed", "failed", "cancelled"]:
                        break
                
                status_progress.close()
                
                # Check if the second step Batch completed successfully
                if batch_status.status.lower() == "completed":
                    # Retrieve the second step results
                    result_file_id = batch_status.output_file_id
                    raw_results = client.files.content(result_file_id).content
                    
                    # Process second step results
                    lines = raw_results.decode('utf-8').splitlines()
                    batch_results = []
                    
                    for i, line in enumerate(tqdm(lines, desc="Processing second step results", position=1)):
                        result_json = json.loads(line)
                        sample_idx = start_idx + i
                        
                        try:
                            if "error" in result_json:
                                print(f"Error in second step for sample {sample_idx}: {result_json['error']}")
                                prediction = "error"
                            elif "response" in result_json and "body" in result_json["response"] and "choices" in result_json["response"]["body"]:
                                # Extract content from the correct JSON structure
                                prediction = result_json["response"]["body"]["choices"][0]["message"]["content"]
                            else:
                                print(f"Unexpected response format for second step sample {sample_idx}: {result_json}")
                                prediction = "error"
                        except Exception as e:
                            print(f"Exception processing second step result for sample {sample_idx}: {str(e)}")
                            prediction = "error"
                        
                        # Extract and normalize answers
                        pred_norm = normalize_answer(extract_answer(prediction))
                        gt_norm = normalize_answer(extract_answer(ground_truths[i]))
                        is_correct = pred_norm == gt_norm
                        
                        result_item = {
                            "id": sample_idx,
                            "prediction": prediction,
                            "prediction_normalized": pred_norm,
                            "ground_truth": ground_truths[i],
                            "ground_truth_normalized": gt_norm,
                            "is_correct": is_correct,
                            "question": problems[i],
                            "image_description": image_descriptions[i]
                        }
                        
                        batch_results.append(result_item)
                        all_results.append(result_item)
                    
                    # Update results file after each batch
                    current_accuracy = update_results_file(all_results, save_path)
                    batch_progress.set_postfix(accuracy=f"{current_accuracy:.2%}")
                else:
                    print(f"Second step batch {batch_idx+1} failed with status: {batch_status.status}")
            else:
                print(f"First step batch {batch_idx+1} failed with status: {batch_status.status}")
        
        finally:
            # No cleanup needed since we're not writing temporary files to disk
            pass
    
    # Save final results
    final_accuracy = update_results_file(all_results, save_path)
    print(f"\nEvaluation complete! Final accuracy: {final_accuracy:.2%}")
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()