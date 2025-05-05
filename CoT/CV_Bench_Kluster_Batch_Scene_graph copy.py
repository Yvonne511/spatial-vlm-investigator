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
KLUSTER_API_KEY = "55af2ad1-6455-4303-9db3-e601d819a614"  # Replace with your key if needed

# Initialize OpenAI client pointing to kluster.ai API
client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=KLUSTER_API_KEY,
)

# Load CV Bench dataset
dataset = load_dataset("nyu-visionx/CV-Bench", split="test")

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# Configuration
batch_size = 1000 # Number of samples to process in each batch
max_batches = None  # Set to a number to limit the batches processed, None for all
save_path = f'./checkpoints/{MODEL_NAME.split("/")[1]}_cv_bench_ccot.json'

# Create checkpoints directory if it doesn't exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# CCoT Prompting Strategy
# First prompt: Scene Graph Generation
scene_graph_prompt = '''
For the provided image and its associated question, think and generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question
Enclose the final scene graph in ```\nscene_graph_here\n```
'''

# Second prompt: Response Generation
response_prompt = '''
Use the image and scene graph as context and answer the following question:
'''

# Final instruction
answer_instruction = "Answer with the option's letter from the given choices directly."

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
    
    # Step 1: Create scene graph generation requests
    sg_requests = []
    image_base64_list = []  # Store image encodings for reuse
    
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
            full_prompt = f"{question} {choice_text}"
        
        # Encode image to base64
        image_base64 = encode_image_to_base64(image)
        image_base64_list.append(image_base64)
        
        # Step 1: Create scene graph generation request
        sg_request = {
            "custom_id": f"sg-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": f"{MODEL_NAME}",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant that generates scene graphs based on images and questions."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{full_prompt}\n\n{scene_graph_prompt}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                "max_completion_tokens": 4000,
                "temperature": 0.0
            }
        }
        
        sg_requests.append(sg_request)
        ground_truths.append(gt_answer)
        problems.append(full_prompt)
        choices_list.append(choices)
    
    return sg_requests, ground_truths, problems, choices_list, image_base64_list

def extract_json_from_response(full_response):
    """
    Attempts to extract valid JSON from a response string using multiple strategies.
    
    Args:
        full_response (str): The string that potentially contains JSON.
        
    Returns:
        str: The extracted JSON string, or "{}" if extraction fails.
    """
    try:
        
        # Strategy 4: Extract content from code blocks
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', full_response, re.IGNORECASE)
        if code_block_match:
            potential_json = code_block_match.group(1).strip()
            try:
                json.loads(potential_json)
                return potential_json
            except:
                pass
            
        # Strategy 1: Try to parse the entire response as JSON
        try:
            json.loads(full_response)
            return full_response
        except json.JSONDecodeError:
            pass
            
        # Strategy 2: Look for JSON object pattern with regex
        json_match = re.search(r'({[\s\S]*?})(?:\s*$|\n)', full_response)
        if json_match:
            json_str = json_match.group(1)
            try:
                json.loads(json_str)
                return json_str
            except:
                pass
            
        # Strategy 3: Look for content after "Scene Graph:" label
        sg_match = re.search(r'Scene Graph:?\s*([\s\S]*)', full_response, re.IGNORECASE)
        if sg_match:
            potential_json = sg_match.group(1).strip()
            try:
                json.loads(potential_json)
                return potential_json
            except:
                pass
                
        
        # If all strategies fail, return empty JSON object
        return "{}"
        
    except Exception as e:
        print(f"Unexpected error during JSON extraction: {str(e)}")
        return "{}"


def process_scene_graphs(sg_responses, image_base64_list, problems, ground_truths, choices_list, start_idx):
    """Process scene graph responses and create response generation requests"""
    answer_requests = []
    full_responses_1 = []
    scene_graphs = []
    
    for i, response_line in enumerate(sg_responses):
        sample_idx = start_idx + i
        
        if "error" in response_line:
            print(f"Error in scene graph generation for sample {sample_idx}: {response_line['error']}")
            scene_graph = "{}"  # Empty scene graph in case of error
            full_response_1 = "Error: " + str(response_line.get('error', 'Unknown error'))
        else:
            full_response_1 = response_line["response"]["body"]["choices"][0]["message"]["content"]
            scene_graph = extract_json_from_response(full_response_1)
        
        # Step 2: Create response generation request with scene graph context
        answer_request = {
            "custom_id": f"answer-{sample_idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": f"{MODEL_NAME}",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant proficient in Visual and Spatial reasoning with tasks involving counting, relations, depth, distances, etc. Think and then answer. Final answer should be provided between <answer> </answer> tags. All questions are multiple choice, just provide the letter (A, B, C, or D) of the correct answer."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Scene Graph: {scene_graph}\n\n{response_prompt}\n{problems[i]}\n\n{answer_instruction}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_list[i]}"}}
                        ]
                    }
                ],
                "max_completion_tokens": 4000,
                "temperature": 0.0
            }
        }
        
        answer_requests.append(answer_request)
        full_responses_1.append(full_response_1)
        scene_graphs.append(scene_graph)
    
    return answer_requests, full_responses_1, scene_graphs

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

def run_batch_job(requests, job_description):
    """Submit a batch job and wait for completion"""
    # Create batch content in memory
    batch_content = BytesIO()
    for request in requests:
        batch_content.write((json.dumps(request) + "\n").encode('utf-8'))
    
    # Reset position to beginning of buffer
    batch_content.seek(0)
    
    # Upload batch job file
    print(f"Uploading {job_description} batch data...")
    batch_input_file = client.files.create(
        file=batch_content,
        purpose="batch"
    )
    
    # Submit batch job
    print(f"Submitting {job_description} batch job...")
    batch_request = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    
    # Poll the batch status until it's complete
    print(f"Monitoring {job_description} batch job {batch_request.id}...")
    status_progress = tqdm(desc=f"{job_description} processing", position=1)
    
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
        results = [json.loads(line) for line in raw_results.decode('utf-8').splitlines()]
        return results, True
    else:
        print(f"{job_description} batch failed with status: {batch_status.status}")
        return [], False

# Main execution
def main():
    total_samples = len(dataset)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    
    print(f"Processing {total_samples} samples in {num_batches} batches using CCoT prompting")
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
        
        # Process batch - Step 1: Scene Graph Generation
        sg_requests, ground_truths, problems, choices_list, image_base64_list = process_batch(batch_idx, start_idx, end_idx)
        
        try:
            # Run scene graph generation batch job
            sg_results, sg_success = run_batch_job(sg_requests, "Scene Graph Generation")
            
            if not sg_success:
                print(f"Skipping batch {batch_idx+1} due to scene graph generation failure")
                continue
            
            # Process scene graphs and create answer requests
            answer_requests, full_responses_1, scene_graphs = process_scene_graphs(sg_results, image_base64_list, problems, ground_truths, choices_list, start_idx)
            
            # Run answer generation batch job
            answer_results, answer_success = run_batch_job(answer_requests, "Response Generation")
            
            if not answer_success:
                print(f"Skipping batch {batch_idx+1} due to answer generation failure")
                continue
            
            # Process results
            batch_results = []
            
            for i, result_json in enumerate(tqdm(answer_results, desc="Processing results", position=1)):
                sample_idx = start_idx + i
                sample = dataset[sample_idx]
                
                if "error" in result_json:
                    print(f"Error in answer for sample {sample_idx}: {result_json['error']}")
                    prediction = "error"
                    full_response = "Error: " + str(result_json.get('error', 'Unknown error'))
                elif "response" in result_json and "body" in result_json["response"] and "choices" in result_json["response"]["body"]:
                    # Extract content from the correct JSON structure
                    full_response = result_json["response"]["body"]["choices"][0]["message"]["content"]
                    prediction = full_response
                else:
                    print(f"Unexpected response format for sample {sample_idx}: {result_json}")
                    prediction = "error"
                    full_response = "Error: Unexpected response format"
                
                # Extract and normalize answers
                pred_extracted = extract_answer(prediction)
                pred_norm = normalize_answer(pred_extracted, ground_truths[i], choices_list[i])
                pred_norm = "(" + pred_norm + ")"
                gt_norm = ground_truths[i].upper()
                is_correct = pred_norm == gt_norm
                
                # Get task type for per-task analysis
                task_type = sample["task"] if "task" in sample else "unknown"
                
                # Store the full results for both prompts
                result_item = {
                    "id": sample_idx,
                    "task": task_type,
                    "question": problems[i],
                    "full_prompt_1_response": full_responses_1[i],  # Full response from first prompt (scene graph)
                    "scene_graph": scene_graphs[i],  # Extracted scene graph
                    "full_prompt_2_response": full_response,  # Full response from second prompt
                    "is_correct": is_correct,
                    "prediction_extracted": pred_extracted,
                    "prediction_normalized": pred_norm,
                    "ground_truth": ground_truths[i],
                    "ground_truth_normalized": gt_norm,
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
        
        except Exception as e:
            print(f"Error processing batch {batch_idx+1}: {e}")
            # Continue with next batch
    
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