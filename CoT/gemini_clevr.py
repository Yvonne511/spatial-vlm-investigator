from datasets import load_dataset
from PIL import Image
import re
import json
from tqdm import tqdm
import os
import asyncio
import aiohttp
import base64
from io import BytesIO
import time
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor


class AsyncGeminiClient:
    """
    Asynchronous client for Gemini API that uses multiple API keys in parallel
    """
    
    def __init__(self, api_keys, model_name="models/gemini-2.0-flash-lite", requests_per_min=30):
        """
        Initialize the async Gemini client
        
        Args:
            api_keys (list): List of API keys to use in parallel
            model_name (str): Gemini model to use
            requests_per_min (int): Rate limit per key per minute
        """
        self.api_keys = api_keys
        self.model_name = model_name
        self.requests_per_min = requests_per_min
        self.key_usage = {key: [] for key in api_keys}
        self.endpoint = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent"
    
    def get_headers(self, api_key):
        """Get the request headers for a specific API key"""
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
    
    def should_throttle(self, api_key):
        """Check if we should throttle requests for this key"""
        now = time.time()
        # Clean up timestamps older than 1 minute
        self.key_usage[api_key] = [t for t in self.key_usage[api_key] if now - t < 60]
        
        # Return True if we're at the rate limit
        return len(self.key_usage[api_key]) >= self.requests_per_min
    
    def record_usage(self, api_key):
        """Record usage of an API key"""
        now = time.time()
        self.key_usage[api_key].append(now)
    
    async def wait_for_capacity(self, api_key):
        """Wait until the key has capacity"""
        while self.should_throttle(api_key):
            delay = 2  # Wait in small increments to check capacity
            await asyncio.sleep(delay)
    
    async def generate_content(self, session, api_key, image_bytes, text_prompt, max_retries=3):
        """
        Generate content asynchronously using Gemini API
        
        Args:
            session: aiohttp ClientSession
            api_key: API key to use
            image_bytes: Image data
            text_prompt: Text prompt for the model
            max_retries: Maximum number of retries on failure
        
        Returns:
            The model's response text
        """
        # Wait if we're at capacity for this key
        await self.wait_for_capacity(api_key)
        
        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare request payload
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": text_prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}
                    ]
                }
            ]
        }
        
        # Record usage of this key
        self.record_usage(api_key)
        
        # Short key identifier for logging
        key_display = f"...{api_key[-4:]}"
        print(f"Using API key {key_display}")
        
        for attempt in range(max_retries + 1):
            try:
                async with session.post(
                    self.endpoint,
                    headers=self.get_headers(api_key),
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    elif response.status == 429:  # Rate limit exceeded
                        error_text = await response.text()
                        print(f"Rate limit hit for {key_display}: {error_text}")
                        
                        if attempt < max_retries:
                            # Add artificial rate limit by filling usage
                            self.key_usage[api_key] = [time.time()] * self.requests_per_min
                            await asyncio.sleep(5)  # Wait before retry
                        continue
                    else:
                        error_text = await response.text()
                        print(f"API error ({response.status}) with {key_display}: {error_text}")
                        
                        if attempt < max_retries:
                            await asyncio.sleep(2)  # Wait before retry
                        continue
            
            except Exception as e:
                print(f"Exception with {key_display}: {str(e)}")
                
                if attempt < max_retries:
                    await asyncio.sleep(2)  # Wait before retry
                else:
                    return None
        
        return None


def encode_image(image):
    """Convert a PIL image to bytes"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()


def extract_answer(text):
    """Extract answer from text between <answer> tags"""
    if text is None:
        return ""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text)
    return match.group(1).strip().lower() if match else text.strip().lower()


def normalize_answer(text):
    """Normalize the answer for consistent comparison"""
    if text is None:
        return ""
    
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


async def process_batch(client, session, batch_items, results, pbar):
    """Process a batch of items concurrently"""
    # Process items in the batch concurrently
    tasks = []
    for item in batch_items:
        idx, image, question, gt_answer = item
        image_bytes = encode_image(image)
        
        # Choose a random API key for better distribution
        api_key = random.choice(client.api_keys)
        
        # Create task
        task = asyncio.create_task(
            client.generate_content(session, api_key, image_bytes, question)
        )
        tasks.append((idx, task, gt_answer, question))
    
    # Wait for all tasks to complete
    for idx, task, gt_answer, question in tasks:
        try:
            response = await task
            
            # Process and store the result
            if response:
                pred_norm = normalize_answer(response)
                gt_norm = normalize_answer(extract_answer(gt_answer))
                is_correct = pred_norm == gt_norm
                
                result = {
                    "id": idx,
                    "prediction": response,
                    "prediction_normalized": pred_norm,
                    "ground_truth": gt_answer,
                    "ground_truth_normalized": gt_norm,
                    "is_correct": is_correct,
                    "question": question
                }
                
                results[idx] = result
            else:
                # Handle failed requests
                results[idx] = {
                    "id": idx,
                    "prediction": None,
                    "prediction_normalized": "",
                    "ground_truth": gt_answer,
                    "ground_truth_normalized": normalize_answer(extract_answer(gt_answer)),
                    "is_correct": False,
                    "question": question
                }
        
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            results[idx] = {
                "id": idx,
                "prediction": None,
                "prediction_normalized": "",
                "ground_truth": gt_answer,
                "ground_truth_normalized": normalize_answer(extract_answer(gt_answer)),
                "is_correct": False,
                "question": question,
                "error": str(e)
            }
        
        # Update progress bar
        pbar.update(1)


def compute_current_accuracy(results):
    """Compute current accuracy from results"""
    completed = [r for r in results.values() if r is not None]
    if not completed:
        return 0.0
    correct = sum(1 for r in completed if r.get("is_correct", False))
    return correct / len(completed)


def save_results(results, save_path):
    """Save results to a JSON file"""
    completed_results = [r for r in results.values() if r is not None]
    
    # Calculate accuracy
    if completed_results:
        correct = sum(1 for r in completed_results if r.get("is_correct", False))
        accuracy = correct / len(completed_results)
    else:
        correct = 0
        accuracy = 0
    
    # Create output data
    output_data = {
        "accuracy": accuracy,
        "total": len(completed_results),
        "correct": correct,
        "results": sorted(completed_results, key=lambda x: x["id"])
    }
    
    # Save to file
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return accuracy


async def main():
    # Define your API keys here
    API_KEYS = [
        "AIzaSyDbkyqQytkQmrNh-t9rDcQn1n8g3xPaWAY",
        "AIzaSyCh4319f7ABATd9AOTuD_iXPEisJg9Tvso",
        "AIzaSyBo60lElBpXd-FwFAQs7IwiKtCllwqd0gI",
        "AIzaSyAyy5G2x9w6fLLj_t-JwjgbLjYTpGUcfSg",
        "AIzaSyBWMsZfHsNpkzxGM8MKJBy3mCg3bzXh5EA",
        "AIzaSyAVb9ig6Q5pBr6dJBA2Qo6FZkJQ3BqiJmY"
    ]
    
    # Parameters
    BATCH_SIZE = 24  # Process more items at once
    SAVE_FREQUENCY = 50  # Save results every 50 samples
    QUESTION_TEMPLATE = "{Question} Final answer should be a single number, \"yes\" or \"no\" between <answer> </answer> tags."
    SAVE_PATH = 'checkpoints_CLEVR_CoGenT/gemini-2.0-flash-CoGenT.json'
    
    # Ensure the checkpoint directory exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB", split="train")
    
    # Create the async Gemini client
    client = AsyncGeminiClient(API_KEYS)
    
    # Dictionary to store results by index
    results = {}
    
    # Set up progress bar
    pbar = tqdm(total=len(dataset))
    
    # Initialize HTTP session
    conn = aiohttp.TCPConnector(limit=20)  # Limit concurrent connections
    async with aiohttp.ClientSession(connector=conn) as session:
        # Process dataset in batches
        for start_idx in range(0, len(dataset), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(dataset))
            batch_items = []
            
            # Prepare batch
            for i in range(start_idx, end_idx):
                sample = dataset[i]
                image = sample["image"].convert("RGB")
                question = QUESTION_TEMPLATE.format(Question=sample["problem"])
                gt_answer = sample["solution"].lower()
                batch_items.append((i, image, question, gt_answer))
            
            # Process batch
            await process_batch(client, session, batch_items, results, pbar)
            
            # Save results at specified intervals
            if end_idx % SAVE_FREQUENCY == 0 or end_idx == len(dataset):
                accuracy = save_results(results, SAVE_PATH)
                print(f"\nProcessed {end_idx}/{len(dataset)} samples. Current accuracy: {accuracy:.2%}")
    
    # Final save
    final_accuracy = save_results(results, SAVE_PATH)
    print(f"\nEvaluation complete. Final accuracy: {final_accuracy:.2%}")


if __name__ == "__main__":
    # Set up event loop policy for Windows if needed
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the async main function
    asyncio.run(main())