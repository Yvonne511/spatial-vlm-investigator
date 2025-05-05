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
KLUSTER_API_KEY = "54835178-1c63-4339-95b7-02b3f1529b44"  # Replace with your key if needed

# Initialize OpenAI client pointing to kluster.ai API
client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=KLUSTER_API_KEY,
)

# Load CV Bench dataset
dataset = load_dataset("nyu-visionx/CV-Bench", split="test")

# Model configuration
MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
# Configuration
batch_size = 5000  # Number of samples to process in each batch
max_batches = None  # Set to a number to limit the batches processed, None for all
save_path = f'./checkpoints/{MODEL_NAME.split("/")[1]}_cv_bench.json'

# Create checkpoints directory if it doesn't exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

def encode_image_to_base64(image):
    """Convert a PIL image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def extract_answer(text):
    """Extract answer from text, handling CV Bench format"""
    # First check for <answer> tags
    match = re.search(r"<[aA]nswer>\s*(.*?)\s*</[aA]nswer>", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Look for lettered choices (A), (B), (C), (D)
    match = re.search(r"\(([A-D])\)", text)
    if match:
        return match.group(1).strip()
    
    # Just return the text as is
    return text.strip()

def normalize_answer(text, gt_answer, choices=None):
    """Normalize answer to handle different formats"""
    text = text.lower().strip()
    
    # Handle multi-line responses
    if "\n" in text:
        lines = text.split("\n")
        for line in lines:
            if line.strip():
                text = line.strip()
                break
    
    # Handle multiple choice answers
    if choices:
        # Check if the answer is a letter (A, B, C, D)
        if re.match(r"^[a-d]$", text):
            # Convert to 0-indexed (A->0, B->1, etc.)
            index = ord(text) - ord('a')
            if 0 <= index < len(choices):
                return text.upper()
        
        # Check if the answer is a number (1, 2, 3, 4)
        if re.match(r"^\d+$", text):
            index = int(text) - 1
            if 0 <= index < len(choices):
                # Convert back to letter format
                return chr(index + ord('A'))
        
        # Check if the answer matches any of the choices directly
        for i, choice in enumerate(choices):
            if text == choice.lower().strip():
                return chr(i + ord('A'))
    
    # Handle yes/no questions
    if "yes" in text:
        return "YES"
    elif "no" in text:
        return "NO"
    
    # If we're dealing with A, B, C, D answers format
    if re.match(r"^[a-d]$", gt_answer.lower()):
        for letter in "abcd":
            if letter in text.lower():
                return letter.upper()
    
    return text.upper()

def compute_accuracy(results):
    """Compute accuracy from results"""
    correct = sum(1 for item in results if item["is_correct"])
    return correct / len(results) if results else 0

def process_batch(batch_index, start_idx, end_idx):
    """Process a batch of samples and return as batch request"""
    requests = []
    ground_truths = []
    problems = []
    choices_list = []
    
    # Prepare batch requests with tqdm
    for i in tqdm(range(start_idx, min(end_idx, len(dataset))), 
                  desc=f"Preparing batch {batch_index+1}", 
                  unit="sample"):
        sample = dataset[i]
        image = sample["image"]
        question = sample["question"]
        gt_answer = sample["answer"]
        choices = sample["choices"] if "choices" in sample and sample["choices"] else None
        
        # Create prompt with choices if available
        full_prompt = question
        if choices:
            choice_text = " ".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            full_prompt = f"{question} Select from the following choices. {choice_text}"
        
        # Encode image to base64
        image_base64 = encode_image_to_base64(image)
        
        # Create request structure
        request = {
            "custom_id": f"sample-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": f"{MODEL_NAME}",
                "messages": [
                    {
                        "role": "system",
                        "content": "Final answer should be provided between <answer> </answer> tags. All questions are multiple choice, just provide the letter (A, B, C, or D) of the correct answer."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                "max_completion_tokens": 3000,
                "temperature": 0.0
            }
        }
        
        requests.append(request)
        ground_truths.append(gt_answer)
        problems.append(full_prompt)
        choices_list.append(choices)
    
    return requests, ground_truths, problems, choices_list

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
    
    print(f"Processing {total_samples} samples in {num_batches} batches")
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
    
    # Track task-specific performance
    task_results = {}
    
    for batch_idx in batch_progress:
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)
        
        batch_progress.set_description(f"Processing batch {batch_idx+1}/{num_batches}")
        
        # Process batch
        requests, ground_truths, problems, choices_list = process_batch(batch_idx, start_idx, end_idx)
        
        try:
            # Create batch content in memory using BytesIO instead of writing to disk
            batch_content = BytesIO()
            for request in requests:
                batch_content.write((json.dumps(request) + "\n").encode('utf-8'))
            
            # Reset position to beginning of buffer
            batch_content.seek(0)
            
            # Upload batch job file directly from memory buffer
            print(f"Uploading batch data...")
            batch_input_file = client.files.create(
                file=batch_content,
                purpose="batch"
            )
            
            # Submit batch job
            print("Submitting batch job...")
            batch_request = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            
            # Poll the batch status until it's complete
            print(f"Monitoring batch job {batch_request.id}...")
            status_progress = tqdm(desc="Batch processing", position=1)
            
            while True:
                time.sleep(5)  # Wait for 5 seconds before checking again
                batch_status = client.batches.retrieve(batch_request.id)
                completed = batch_status.request_counts.completed
                total = batch_status.request_counts.total
                
                # Update progress bar
                status_progress.total = total
                status_progress.n = completed
                status_progress.set_description(f"Status: {batch_status.status}")
                status_progress.set_postfix(completed=f"{completed}/{total}")
                status_progress.update(0)  # Force refresh
                
                if batch_status.status.lower() in ["completed", "failed", "cancelled"]:
                    break
            
            status_progress.close()
            
            # Check if the Batch completed successfully
            if batch_status.status.lower() == "completed":
                # Retrieve the results
                result_file_id = batch_status.output_file_id
                raw_results = client.files.content(result_file_id).content
                
                # Process results with tqdm
                lines = raw_results.decode('utf-8').splitlines()
                batch_results = []
                
                for i, line in enumerate(tqdm(lines, desc="Processing results", position=1)):
                    result_json = json.loads(line)
                    sample_idx = start_idx + i
                    sample = dataset[sample_idx]
                    
                    if "error" in result_json:
                        print(f"Error in sample {sample_idx}: {result_json['error']}")
                        prediction = "error"
                    elif "response" in result_json and "body" in result_json["response"] and "choices" in result_json["response"]["body"]:
                        # Extract content from the correct JSON structure
                        prediction = result_json["response"]["body"]["choices"][0]["message"]["content"]
                    else:
                        print(f"Unexpected response format for sample {sample_idx}: {result_json}")
                        prediction = "error"
                    
                    # Extract and normalize answers
                    pred_extracted = extract_answer(prediction)
                    pred_norm = normalize_answer(pred_extracted, ground_truths[i], choices_list[i])
                    pred_norm = "(" + pred_norm + ")"
                    gt_norm = ground_truths[i].upper()
                    is_correct = pred_norm == gt_norm
                    
                    # Get task type for per-task analysis
                    task_type = sample["task"] if "task" in sample else "unknown"
                    
                    result_item = {
                        "id": sample_idx,
                        "prediction": prediction,
                        "prediction_extracted": pred_extracted,
                        "prediction_normalized": pred_norm,
                        "ground_truth": ground_truths[i],
                        "ground_truth_normalized": gt_norm,
                        "is_correct": is_correct,
                        "question": problems[i],
                        "task": task_type,
                        "source": sample.get("source", "unknown"),
                        "source_dataset": sample.get("source_dataset", "unknown")
                    }
                    
                    # Track task-specific results
                    if task_type not in task_results:
                        task_results[task_type] = {"correct": 0, "total": 0}
                    
                    task_results[task_type]["total"] += 1
                    if is_correct:
                        task_results[task_type]["correct"] += 1
                    
                    batch_results.append(result_item)
                    all_results.append(result_item)
                
                # Update results file after each batch
                current_accuracy = update_results_file(all_results, save_path)
                batch_progress.set_postfix(accuracy=f"{current_accuracy:.2%}")
                
                # Print task-specific accuracy
                print("\nTask-specific accuracy:")
                for task, stats in task_results.items():
                    task_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                    print(f"  {task}: {task_acc:.2%} ({stats['correct']}/{stats['total']})")
            else:
                print(f"Batch {batch_idx+1} failed with status: {batch_status.status}")
        
        finally:
            # No cleanup needed since we're not writing temporary files to disk
            pass
    
    # Save final results
    final_accuracy = update_results_file(all_results, save_path)
    print(f"\nEvaluation complete! Final accuracy: {final_accuracy:.2%}")
    print(f"Results saved to {save_path}")
    
    # Print final task-specific results
    print("\nFinal task-specific accuracy:")
    for task, stats in task_results.items():
        task_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {task}: {task_acc:.2%} ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    main()