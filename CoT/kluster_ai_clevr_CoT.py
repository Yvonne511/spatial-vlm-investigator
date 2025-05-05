from datasets import load_dataset
from PIL import Image
import re
import json
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
import os
import base64
from io import BytesIO

# Configure Kluster.ai Llama 4 Scout API
KLUSTER_API_KEY = "a5e0b5c4-dee5-402f-acaf-ced3114d5b3c"  # Replace with your actual API key
KLUSTER_API_URL = "https://api.kluster.ai/v1/chat/completions"  # Adjust if different

# Control concurrency with this parameter - adjust based on your API rate limits
MAX_CONCURRENT_REQUESTS = 5

def encode_image_to_base64(image):
    """Convert a PIL image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

async def call_llama4_scout_api_chat(session, image, first_prompt, second_prompt):
    """Call Kluster.ai's Llama 4 Scout API in a chat-like environment with two prompts"""
    image_base64 = encode_image_to_base64(image)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KLUSTER_API_KEY}"
    }
    
    # First API call - ask for image description
    first_payload = {
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that can see and describe images accurately."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": first_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 1024
    }
    
    try:
        # Make first API call to get image description
        async with session.post(KLUSTER_API_URL, headers=headers, json=first_payload) as response:
            response.raise_for_status()
            first_response_json = await response.json()
            image_description = first_response_json["choices"][0]["message"]["content"]
        
        # Second API call - ask the actual question using the description
        second_payload = {
            "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "Final answer should be a single number, \"yes\" or \"no\" between <answer> </answer> tags."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": first_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                },
                {
                    "role": "assistant",
                    "content": image_description
                },
                {
                    "role": "user",
                    "content": second_prompt
                }
            ],
            "max_tokens": 1024
        }
        
        async with session.post(KLUSTER_API_URL, headers=headers, json=second_payload) as response:
            response.raise_for_status()
            second_response_json = await response.json()
            return second_response_json["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error during Llama 4 Scout API call: {e}")
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

async def process_batch(session, batch, first_prompt):
    """Process a batch of samples concurrently"""
    tasks = []
    for sample in batch:
        image = sample["image"].convert("RGB")
        second_prompt = sample["problem"]
        task = call_llama4_scout_api_chat(session, image, first_prompt, second_prompt)
        tasks.append(task)
    
    return await asyncio.gather(*tasks)

async def save_checkpoint(responses, ground_truths, problems, save_path, total_processed, total_samples):
    """Save checkpoint and print progress"""
    accuracy = compute_accuracy(responses, ground_truths, problems, save_path)
    print(f"Processed {total_processed}/{total_samples} samples. Current accuracy: {accuracy:.2%}")

async def main():
    dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB", split="train")
    batch_size = MAX_CONCURRENT_REQUESTS  # Process in batches equal to concurrency limit
    
    responses, ground_truths, problems = [], [], []
    FIRST_PROMPT = "Describe the objects in this image: their color, shape, size, location, relative location, reflectance and material (glossy is metallic, matte is rubber)."
    save_path = './checkpoints/llama4_scout_eval_CoGenT_ValB.json'
    
    # Create directory for checkpoints if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Configure TCP connector with keep-alive
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, keepalive_timeout=60)
    
    # Configure timeout
    timeout = aiohttp.ClientTimeout(total=120)  # 2-minute timeout for entire request
    
    # Initialize session with connector and timeout
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Process dataset in batches
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            batch_questions = [sample["problem"] for sample in batch]
            batch_answers = [sample["solution"].lower() for sample in batch]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}")
            
            # Process batch concurrently
            batch_responses = await process_batch(session, batch, FIRST_PROMPT)
            
            # Add results to overall lists
            responses.extend(batch_responses)
            ground_truths.extend(batch_answers)
            problems.extend(batch_questions)
            
            # Save checkpoint after each batch
            total_processed = min(i + batch_size, len(dataset))
            await save_checkpoint(responses, ground_truths, problems, save_path, total_processed, len(dataset))
    
    # Final accuracy calculation
    accuracy = compute_accuracy(responses, ground_truths, problems, save_path)
    print(f"Final accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())