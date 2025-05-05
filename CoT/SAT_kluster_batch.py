from datasets import load_dataset
from PIL import Image
import re
import json
from tqdm import tqdm
import os
import base64
from io import BytesIO
import time
from openai import OpenAI
import traceback

# Get API key
KLUSTER_API_KEY = "a5e0b5c4-dee5-402f-acaf-ced3114d5b3c"

# Initialize OpenAI client pointing to kluster.ai API
client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=KLUSTER_API_KEY,
)

# Configuration
batch_size = 5  # Number of samples to process in each batch
max_batches = None  # Set to a number to limit the batches processed, None for all
save_path = './checkpoints/llama4_scout_eval_SAT.json'

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
    
    # For SAT, answers are usually single letters (A, B, C, D, E)
    # Extract just the letter if present
    match = re.search(r'\b([a-e])\b', text)
    if match:
        return match.group(1)
    
    return text

def compute_accuracy(results):
    """Compute accuracy from results"""
    correct = sum(1 for item in results if item["is_correct"])
    return correct / len(results) if results else 0

def format_answer_options(answers):
    """Format answer options as a string"""
    options = ""
    for i, answer in enumerate(answers):
        letter = chr(65 + i)  # Convert to A, B, C, etc.
        options += f"{letter}. {answer}\n"
    return options.strip()

# Load dataset with custom download configurations
def load_sat_dataset():
    """
    Load the SAT dataset with error handling and retry logic
    """
    try:
        # First try: attempt direct loading with specific config
        print("Attempting to load dataset with specific configuration...")
        dataset = load_dataset(
            "array/SAT", 
            split="val",
            cache_dir="./dataset_cache",  # Use relative path for better compatibility
            trust_remote_code=True,
            use_auth_token=True,
        )
        return dataset
    except Exception as e:
        print(f"First attempt failed: {e}")
        
        try:
            # Second try: use streaming mode which handles some conversion issues
            print("Attempting to load dataset in streaming mode...")
            dataset = load_dataset(
                "array/SAT", 
                split="val",
                streaming=True,
                cache_dir="./dataset_cache",
                trust_remote_code=True,
            )
            # Convert to regular dataset by taking the first N examples
            # Adjust this number based on your needs
            return list(dataset.take(500))
        except Exception as e:
            print(f"Second attempt failed: {e}")
            
            # Fallback: Create a minimal dataset manually
            # This is if both loading methods fail
            print("Creating a minimal test dataset...")
            return create_minimal_dataset()

def create_minimal_dataset():
    """
    Create a minimal test dataset with a few examples to allow testing
    of the evaluation pipeline even when the dataset loading fails
    """
    # Example structure - you would need to fill this with real data
    # for actual testing
    return [
        {
            "question": "Example SAT question 1",
            "answers": ["Option A", "Option B", "Option C", "Option D", "Option E"],
            "correct_answer": 0,  # Index of correct answer (0 = A)
            "image_bytes": [Image.new('RGB', (300, 300), color='white')]  # Placeholder image
        },
        {
            "question": "Example SAT question 2",
            "answers": ["Option A", "Option B", "Option C", "Option D", "Option E"],
            "correct_answer": 1,  # Index of correct answer (1 = B)
            "image_bytes": [Image.new('RGB', (300, 300), color='white')]  # Placeholder image
        }
    ]

def process_batch(batch_index, samples):
    """Process a batch of samples and return as batch request"""
    requests = []
    ground_truths = []
    problems = []
    
    # Prepare batch requests with tqdm
    for i, sample in enumerate(tqdm(samples, desc=f"Preparing batch {batch_index+1}", unit="sample")):
        question = sample["question"]
        answers = sample["answers"]
        correct_answer_idx = sample["correct_answer"]
        correct_answer_letter = chr(65 + correct_answer_idx)  # Convert to A, B, C, etc.
        
        # Format answer options
        answer_options = format_answer_options(answers)
        
        # Handle either one or two images
        image_content = []
        for img in sample["image_bytes"]:
            image_base64 = encode_image_to_base64(img)
            image_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        # Create the full prompt with question and answer options
        prompt = f"{question}\n\n{answer_options}"
        
        # Create request structure
        request = {
            "custom_id": f"sample-{batch_index * batch_size + i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are solving a spatial aptitude test. Analyze the image(s) carefully and select the best answer from the given options. Your final answer should be just the letter (A, B, C, D, or E) between <answer> </answer> tags."
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}] + image_content
                    }
                ],
                "max_completion_tokens": 3000,
                "temperature": 0.0
            }
        }
        
        requests.append(request)
        ground_truths.append(correct_answer_letter)
        problems.append({
            "question": question,
            "answers": answers,
            "correct_answer_idx": correct_answer_idx,
            "correct_answer_letter": correct_answer_letter
        })
    
    return requests, ground_truths, problems

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

def batch_iterator(dataset, batch_size):
    """Iterator that yields batches from the dataset"""
    total_samples = len(dataset)
    for i in range(0, total_samples, batch_size):
        yield i // batch_size, dataset[i:min(i + batch_size, total_samples)]

# Main execution
def main():
    try:
        # First try to load the dataset
        print("Loading SAT dataset...")
        dataset = load_sat_dataset()
        
        if not dataset:
            print("Failed to load dataset. Please check your network connection and dataset credentials.")
            return
            
        total_samples = len(dataset)
        print(f"Successfully loaded {total_samples} samples from the SAT dataset.")
        
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
        
        # Process batches
        for batch_idx in batch_progress:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            current_batch = dataset[start_idx:end_idx]
            
            batch_progress.set_description(f"Processing batch {batch_idx+1}/{num_batches}")
            
            # Process batch
            requests, ground_truths, problems = process_batch(batch_idx, current_batch)
            
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
                        pred_norm = normalize_answer(extract_answer(prediction))
                        gt_norm = normalize_answer(ground_truths[i])
                        is_correct = pred_norm == gt_norm
                        
                        result_item = {
                            "id": sample_idx,
                            "prediction": prediction,
                            "prediction_normalized": pred_norm,
                            "ground_truth": ground_truths[i],
                            "ground_truth_normalized": gt_norm,
                            "is_correct": is_correct,
                            "question": problems[i]["question"],
                            "answers": problems[i]["answers"],
                            "correct_answer_idx": problems[i]["correct_answer_idx"],
                            "num_images": len(current_batch[i]["image_bytes"])  # Track number of images
                        }
                        
                        batch_results.append(result_item)
                        all_results.append(result_item)
                    
                    # Update results file after each batch
                    current_accuracy = update_results_file(all_results, save_path)
                    batch_progress.set_postfix(accuracy=f"{current_accuracy:.2%}")
                else:
                    print(f"Batch {batch_idx+1} failed with status: {batch_status.status}")
            
            except Exception as e:
                print(f"Error processing batch {batch_idx+1}: {e}")
                traceback.print_exc()
        
        # Save final results
        final_accuracy = update_results_file(all_results, save_path)
        print(f"\nEvaluation complete! Final accuracy: {final_accuracy:.2%}")
        print(f"Results saved to {save_path}")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        traceback.print_exc()

# Utility function to explore the dataset without running the full evaluation
def explore_example(dataset, index=0):
    """Visualize an example from the dataset"""
    try:
        import matplotlib.pyplot as plt
        
        if not dataset or index >= len(dataset):
            print("Invalid dataset or index")
            return None
        
        example = dataset[index]
        question = example["question"]
        answers = example["answers"]
        correct_idx = example["correct_answer"]
        
        # Format answer options with letter indicators
        formatted_answers = []
        for i, ans in enumerate(answers):
            letter = chr(65 + i)  # Convert to A, B, C, etc.
            formatted_answers.append(f"{letter}. {ans}")
        
        # Create figure based on number of images
        num_images = len(example["image_bytes"])
        fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
        
        # Handle single image case
        if num_images == 1:
            axes = [axes]
        
        # Display images
        for i, img in enumerate(example["image_bytes"]):
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"Image {i+1}")
        
        # Set overall title
        fig.suptitle(f"Question: {question}\n\n" + "\n".join(formatted_answers) + 
                    f"\n\nCorrect Answer: {chr(65 + correct_idx)}", fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)  # Make room for title
        plt.show()
        
        return {
            "question": question,
            "answers": answers,
            "correct_answer": chr(65 + correct_idx),
            "num_images": num_images
        }
    except Exception as e:
        print(f"Error exploring example: {e}")
        return None

if __name__ == "__main__":
    # Load the dataset first 
    try:
        print("Attempting to load dataset for exploration...")
        dataset = load_sat_dataset()
        if dataset:
            # Uncomment to explore examples before running full evaluation
            # explore_example(dataset, 0)  # View first example
            # explore_example(dataset, 10)  # View example at index 10
            # explore_example(dataset, 20)  # View example at index 20
            pass
    except Exception as e:
        print(f"Couldn't load dataset for exploration: {e}")
    
    # Run the main evaluation
    main()