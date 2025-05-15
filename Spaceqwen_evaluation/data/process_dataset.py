import re
from typing import List, Tuple, Union, Optional
from PIL import Image
import os
import json
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from hashlib import sha256
import requests
from io import BytesIO

# Input an already opened json file
class SATDataset:
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
    
# New: Dataset class specifically for processing local SAT data
class SATLocalDataset(Dataset):
    def __init__(self, json_file, image_dir, transform=None, max_samples=None):
        """
        Args:
            json_file (str): Path to the JSON file with annotations
            image_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on images
            max_samples (int, optional): Maximum number of samples to load (for debugging)
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Limit sample count (if needed)
        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data[:max_samples]
            
        # Preprocessing: Validate all image paths exist
        self._validate_images()
        
    def _validate_images(self):
        """Validate image files exist and fix paths"""
        valid_data = []
        for item in self.data:
            if 'images' in item and len(item['images']) > 0:
                # Check image path
                image_path = item['images'][0]
                # Replace Windows path separators with current OS separator
                image_path = image_path.replace('\\', os.path.sep)
                # Build full path
                full_path = os.path.join(self.image_dir, image_path)
                
                # If path contains SAT_images_train but it's already part of the full path, remove redundant part
                if "SAT_images_train" in full_path and "SAT_images_train" in self.image_dir:
                    full_path = os.path.join(self.image_dir, os.path.basename(image_path))
                
                # Check if file exists
                if os.path.exists(full_path):
                    # Update image path to absolute path
                    item['images'][0] = full_path
                    valid_data.append(item)
                else:
                    print(f"Warning: Cannot find image file {full_path}")
            else:
                print(f"Warning: Data item missing image path: {item}")
        
        self.data = valid_data
        print(f"Valid samples after validation: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            dict: Contains the following key-value pairs:
                - 'image': Processed image (PIL Image or tensor)
                - 'question': Question text
                - 'answer': Answer text
                - 'is_counting': Boolean indicating if it's a counting question
                - 'number_answer': Extracted numeric answer if counting question
        """
        item = self.data[idx]
        
        # Get question (from user message)
        user_message = item['messages'][0]['content']
        question = user_message.replace("<image>", "").strip()
        
        # Get answer (from assistant message)
        answer = item['messages'][1]['content']
        
        # Load image
        image_path = item['images'][0]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform (if any)
        if self.transform:
            image = self.transform(image)
            
        # Check if counting question
        is_counting = False
        number_answer = None
        number_in_answer = extract_number_from_answer(answer)
        if number_in_answer is not None:
            is_counting = True
            number_answer = number_in_answer
            
        return {
            'image': image,
            'question': question,
            'answer': answer,
            'wrapped_answer': f"<answer>{answer}</answer>",
            'is_counting': is_counting,
            'number_answer': number_answer,
            'problem': question,  # Interface consistent with other datasets
            'solution': f"<answer>{answer}</answer>"  # Interface consistent with other datasets
        }

# Create DataLoader for SAT dataset
def create_sat_dataloader(json_file, image_dir, batch_size=8, shuffle=True, num_workers=4, max_samples=None):
    """
    Create DataLoader for SAT dataset
    
    Args:
        json_file (str): Path to JSON data file
        image_dir (str): Path to image directory
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker threads for loading data
        max_samples (int, optional): Maximum number of samples (for debugging)
        
    Returns:
        DataLoader: PyTorch data loader
    """
    # Create dataset
    dataset = SATLocalDataset(json_file, image_dir, max_samples=max_samples)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sat_collate_fn
    )
    
    return dataloader

# Collate function for handling images of different sizes
def sat_collate_fn(batch):
    """
    Custom collate function for handling images of different sizes and text data
    
    Returns:
        dict: Dictionary containing batched data
    """
    # Group data of same type
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    wrapped_answers = [item['wrapped_answer'] for item in batch]
    is_counting = [item['is_counting'] for item in batch]
    number_answers = [item['number_answer'] for item in batch]
    problems = [item['problem'] for item in batch]
    solutions = [item['solution'] for item in batch]
    
    # Return batched data
    return {
        'image': images,
        'question': questions,
        'answer': answers,
        'wrapped_answer': wrapped_answers,
        'is_counting': is_counting,
        'number_answer': number_answers,
        'problem': problems,
        'solution': solutions
    }

# Load local SAT data example
def load_local_sat_data(json_file, image_dir, batch_size=8, max_samples=None):
    """Load local SAT data and return DataLoader"""
    print(f"Loading SAT dataset from local file: {json_file}")
    print(f"Image directory: {image_dir}")
    
    # Create data loader
    dataloader = create_sat_dataloader(
        json_file=json_file,
        image_dir=image_dir,
        batch_size=batch_size,
        max_samples=max_samples
    )
    
    print(f"Dataset loaded, {len(dataloader.dataset)} samples, {len(dataloader)} batches")
    return dataloader

def evaluate_response(model_response, sample, dataset_name, threshold=2, batch_idx=0):
    """Evaluate model response and return result information"""
    result = {
        "index": batch_idx,
        "question": sample["problem"],
        "model_response": model_response,
        "ground_truth": sample["solution"]
    }
    
    if dataset_name == "CLEVR":
        ground_truth = sample["solution"]
        
        # First determine question type
        if is_boolean_answer(ground_truth):
            result["question_type"] = "boolean"
            # Try to extract answer from response
            match = re.search(r'<answer>(.*?)</answer>', model_response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip().lower()
                result["pred_answer"] = extracted_answer
                is_correct = compare_answers(extracted_answer, ground_truth)
                result["correct"] = is_correct
                
                if is_correct:
                    print(f"Answer evaluation: Boolean question answered correctly")
                    return result, "boolean", True
                else:
                    print(f"Answer evaluation: Boolean question answered incorrectly (model: {extracted_answer}, correct: {ground_truth})")
                    return result, "boolean", False
            else:
                print("Answer evaluation: Invalid answer format (missing <answer> tags)")
                result["question_type"] = "invalid"
                return result, "invalid", None
                
        # Check if numeric answer
        elif extract_number_from_answer(ground_truth) is not None:
            result["question_type"] = "counting"
            # Try to extract answer from response
            match = re.search(r'<answer>(.*?)</answer>', model_response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip()
                result["pred_answer"] = extracted_answer
                
                try:
                    pred_number = extract_number_from_answer(extracted_answer)
                    true_number = extract_number_from_answer(ground_truth)
                    
                    if pred_number is not None and true_number is not None:
                        diff = abs(pred_number - true_number)
                        result["diff"] = diff
                        is_completely_correct = (diff == 0)
                        is_within_threshold = (diff <= threshold)
                        result["completely_correct"] = is_completely_correct
                        result["correct_within_threshold"] = is_within_threshold
                        result["correct"] = is_completely_correct
                        
                        if is_completely_correct:
                            print(f"Answer evaluation: Counting question answered correctly ({pred_number})")
                            return result, "counting", True
                        elif is_within_threshold:
                            print(f"Answer evaluation: Counting question within error margin (model: {pred_number}, correct: {true_number}, diff: {diff})")
                            return result, "counting", True
                        else:
                            print(f"Answer evaluation: Counting question answered incorrectly (model: {pred_number}, correct: {true_number}, diff: {diff})")
                            return result, "counting", False
                    else:
                        print("Answer evaluation: Cannot extract valid number from answer")
                        result["question_type"] = "invalid"
                        return result, "invalid", None
                except:
                    print("Answer evaluation: Numeric processing error")
                    result["question_type"] = "invalid"
                    return result, "invalid", None
            else:
                print("Answer evaluation: Invalid answer format (missing <answer> tags)")
                result["question_type"] = "invalid"
                return result, "invalid", None
                
        # Other types of questions
        else:
            result["question_type"] = "other"
            # Try to extract answer from response
            match = re.search(r'<answer>(.*?)</answer>', model_response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip()
                result["pred_answer"] = extracted_answer
                is_correct = compare_answers(extracted_answer, ground_truth)
                result["correct"] = is_correct
                
                if is_correct:
                    print("Answer evaluation: Other question answered correctly")
                    return result, "other", True
                else:
                    print(f"Answer evaluation: Other question answered incorrectly (model: {extracted_answer}, correct: {ground_truth})")
                    return result, "other", False
            else:
                print("Answer evaluation: Invalid answer format (missing <answer> tags)")
                result["question_type"] = "invalid"
                return result, "invalid", None
    
    elif dataset_name == "SAT":
        # For SAT dataset, check if counting question
        if "is_counting" in sample and sample["is_counting"]:
            result["question_type"] = "counting"
            
            # Try to extract answer from response
            match = re.search(r'<answer>(.*?)</answer>', model_response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip()
                result["pred_answer"] = extracted_answer
                
                try:
                    pred_number = extract_number_from_answer(extracted_answer)
                    true_number = sample.get("number_answer")
                    
                    if pred_number is not None and true_number is not None:
                        diff = abs(pred_number - true_number)
                        result["diff"] = diff
                        is_completely_correct = (diff == 0)
                        is_within_threshold = (diff <= threshold)
                        result["completely_correct"] = is_completely_correct
                        result["correct_within_threshold"] = is_within_threshold
                        result["correct"] = is_completely_correct
                        
                        if is_completely_correct:
                            print(f"Answer evaluation: Counting question completely correct ({pred_number})")
                            return result, "counting", True
                        elif is_within_threshold:
                            print(f"Answer evaluation: Counting question within error margin (model: {pred_number}, correct: {true_number}, diff: {diff})")
                            return result, "counting", True
                        else:
                            print(f"Answer evaluation: Counting question outside error margin (model: {pred_number}, correct: {true_number}, diff: {diff})")
                            return result, "counting", False
                    else:
                        print("Answer evaluation: Cannot extract valid number from answer")
                        result["question_type"] = "invalid"
                        return result, "invalid", None
                except Exception as e:
                    print(f"Answer evaluation: Numeric processing error - {e}")
                    result["question_type"] = "invalid"
                    return result, "invalid", None
            else:
                print("Answer evaluation: Invalid answer format (missing <answer> tags)")
                result["question_type"] = "invalid"
                return result, "invalid", None
        else:
            # Non-counting questions processed as "other" type
            result["question_type"] = "other"
            # Try to extract answer from response
            match = re.search(r'<answer>(.*?)</answer>', model_response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip()
                result["pred_answer"] = extracted_answer
                
                # For non-counting questions, just record result without affecting counting statistics
                print("Answer evaluation: Non-counting question")
                return result, "other", None
            else:
                print("Answer evaluation: Invalid answer format (missing <answer> tags)")
                result["question_type"] = "invalid"
                return result, "invalid", None
    
    elif dataset_name == "pixmo":
        # pixmo dataset processing - all are counting questions
        result["question_type"] = "counting"
        
        # Try to extract answer from response
        match = re.search(r'<answer>(.*?)</answer>', model_response, re.IGNORECASE | re.DOTALL)
        if match:
            extracted_answer = match.group(1).strip()
            result["pred_answer"] = extracted_answer
            
            try:
                pred_number = extract_number_from_answer(extracted_answer)
                true_number = sample.get("number_answer")
                
                if pred_number is not None and true_number is not None:
                    diff = abs(pred_number - true_number)
                    result["diff"] = diff
                    is_completely_correct = (diff == 0)
                    is_within_threshold = (diff <= threshold)
                    result["completely_correct"] = is_completely_correct
                    result["correct_within_threshold"] = is_within_threshold
                    result["correct"] = is_completely_correct
                    
                    if is_completely_correct:
                        print(f"Answer evaluation: Counting question completely correct ({pred_number})")
                        return result, "counting", True
                    elif is_within_threshold:
                        print(f"Answer evaluation: Counting question within error margin (model: {pred_number}, correct: {true_number}, diff: {diff})")
                        return result, "counting", True
                    else:
                        print(f"Answer evaluation: Counting question outside error margin (model: {pred_number}, correct: {true_number}, diff: {diff})")
                        return result, "counting", False
                else:
                    print("Answer evaluation: Cannot extract valid number from answer")
                    result["question_type"] = "invalid"
                    return result, "invalid", None
            except Exception as e:
                print(f"Answer evaluation: Numeric processing error - {e}")
                result["question_type"] = "invalid"
                return result, "invalid", None
        else:
            print("Answer evaluation: Invalid answer format (missing <answer> tags)")
            result["question_type"] = "invalid"
            return result, "invalid", None
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def process_vision_info(
    conversations: List[dict] | List[List[dict]]
) -> Tuple[Union[List[Image.Image], Image.Image, None], None]:
    """Simplified function to process vision info, handles only images, not videos
    
    Args:
        conversations: List of conversations or nested list of conversations
        
    Returns:
        tuple: (image_inputs, None) where image_inputs can be single image, list of images, or None
    """
    # Ensure conversations is in nested list format
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    
    # Extract image information
    image_inputs = []
    
    for conversation in conversations:
        for message in conversation:
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    # Only process image type content
                    if item.get("type") == "image" and "image" in item:
                        image_inputs.append(item["image"])
    
    # Standardize output format
    if len(image_inputs) == 0:
        image_inputs = None
    elif len(image_inputs) == 1:
        image_inputs = image_inputs[0]
    
    # Return image inputs and None (indicating no video inputs)
    return image_inputs, None

def process_sat_sample(sample):
    """Process SAT dataset sample (Hugging Face format)"""
    try:
        question = sample['problem']
        solution = f"<answer>{sample['solution']}</answer>"
        ground_truth = sample.get('answer', sample.get('solution', ""))
        image = sample.get('image', None)
        
        # For samples without image, try loading from image_path
        if image is None and 'image_path' in sample:
            from PIL import Image
            try:
                image = Image.open(sample['image_path']).convert("RGB")
            except Exception as e:
                print(f"Cannot load image {sample['image_path']}: {e}")
                return None
        
        # Check if counting question (by checking if solution contains number)
        is_counting = False
        number_in_solution = extract_number_from_answer(ground_truth)
        if number_in_solution is not None:
            is_counting = True
            
        # Return processed sample
        return {
            'problem': question,
            'solution': solution,
            'image': image,
            'is_counting': is_counting,  # New field marking if counting question
            'number_answer': number_in_solution  # New field storing counting answer
        }
    except Exception as e:
        print(f"Error processing SAT sample: {e}")
        return None

def extract_number_from_answer(ans):
    """Extract number from answer"""
    # Try to find clearly marked numbers
    if ans is None:
        return None
    
    # Extract numbers from text, supporting multiple formats
    number_patterns = [
        r"<answer>(\d+(\.\d+)?)</answer>",  # <answer>5</answer>
        r"the answer is (\d+(\.\d+)?)",  # the answer is 5
        r"answer: (\d+(\.\d+)?)",  # answer: 5
        r"(\d+(\.\d+)?)",  # any number
    ]
    
    for pattern in number_patterns:
        match = re.search(pattern, ans, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    # If no numbers found, check for English number words
    word_to_number = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }
    
    for word, number in word_to_number.items():
        if re.search(r'\b' + word + r'\b', ans, re.IGNORECASE):
            return float(number)
    
    return None

def is_boolean_answer(ans):
    """Determine if answer is boolean"""
    # Check for explicit boolean answer patterns
    bool_patterns = [
        r"<answer>(yes|no|true|false)</answer>",
        r"\b(yes|no|true|false)\b",
    ]
    
    for pattern in bool_patterns:
        if re.search(pattern, ans.lower()):
            return True
    
    return False

def compare_answers(pred_answer, ground_truth):
    """Compare predicted answer with ground truth"""
    if pred_answer is None or ground_truth is None:
        return False
    
    # Convert answers to strings and normalize
    pred_str = str(pred_answer).strip().lower()
    truth_str = str(ground_truth).strip().lower()
    
    # Check if numeric answer
    try:
        pred_num = float(pred_str.replace(',', ''))
        truth_num = float(truth_str.replace(',', ''))
        # Allow small error margin
        return abs(pred_num - truth_num) < 0.01
    except (ValueError, TypeError):
        pass
    
    # For descriptive answers, check if key information is included
    # Using simple substring matching, can be extended to more complex logic if needed
    return pred_str in truth_str or truth_str in pred_str

def process_batch(dataset_name, batch, data_dir=None):
    if dataset_name == "CLEVR":
        messages_list = []
        for sample in batch:
            image = sample['image']
            question = sample['problem']
            # Explicitly request <answer> tags
            prompt = f"{question} Please answer with the answer, wrapped in <answer> </answer> tags, e.g. <answer> 3 </answer>."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages_list.append(messages)
        return messages_list

    elif dataset_name == "SAT":
        processed_samples = []
            
        for sample in batch:
            # Check if local file format
            if 'messages' in sample and 'images' in sample:
                # Local file format
                user_message = sample['messages'][0]['content']
                if "<image>" in user_message:
                    user_message = user_message.replace("<image>", "").strip()
                
                ground_truth = sample['messages'][1]['content']
                
                # Load image
                image_paths = sample['images']
                if not image_paths:
                    print(f"Sample has no image path, skipping")
                    continue
                    
                try:
                    image_path = os.path.join(data_dir, "SAT", image_paths[0])
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"Cannot load image {image_path}: {e}")
                    # Try relative path
                    try:
                        image = Image.open(image_paths[0]).convert("RGB")
                    except:
                        print(f"Cannot load image, skipping")
                        continue
                
                # Check if counting question (by checking if solution contains number)
                is_counting = False
                number_in_solution = extract_number_from_answer(ground_truth)
                if number_in_solution is not None:
                    is_counting = True
                
                # Processed sample
                processed_sample = {
                    'problem': user_message,
                    'solution': f"<answer>{ground_truth}</answer>",
                    'image': image,
                    'is_counting': is_counting,  # Mark if counting question
                    'number_answer': number_in_solution  # Store counting answer
                }
                
            else:
                # Hugging Face format
                processed_sample = process_sat_sample(sample)
            
            if processed_sample:
                processed_samples.append(processed_sample)
        
        # Construct messages
        messages_list = []
        for sample in processed_samples:
            image = sample['image']
            question = sample['problem']
            # Explicitly request <answer> tags
            prompt_suffix = ""
            if sample.get('is_counting', False):
                prompt_suffix = " Please make sure to include the numeric answer."
            
            prompt = f"{question}{prompt_suffix} Please respond with the answer wrapped in <answer> </answer> tags, e.g. <answer>3</answer>; <answer>42</answer>; <answer>7.5</answer>."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages_list.append(messages)
        return messages_list, processed_samples
    
    elif dataset_name == "pixmo":
        messages_list = []
        processed_samples = []
        
        for i, example in enumerate(batch):
            try:
                # Download image
                image_bytes = requests.get(example["image_url"]).content
                byte_hash = sha256(image_bytes).hexdigest()
                # Convert to PIL image
                image = Image.open(BytesIO(image_bytes))
                
                # Get label and answer
                label = example["label"]
                ground_truth = int(example["count"])
                
                # Build prompt
                prompt = f"how many {label} are in this image, Please answer with the answer, wrapped in <answer> </answer> tags, e.g. <answer> 3 </answer>."
                
                # Create message structure
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                messages_list.append(messages)
                
                # Create processed sample, add necessary fields for consistency
                processed_sample = {
                    'problem': prompt,
                    'solution': f"<answer>{ground_truth}</answer>",
                    'image': image,
                    'is_counting': True,  # Explicitly set to True
                    'number_answer': ground_truth
                }
                processed_samples.append(processed_sample)
                
                print(f"Processed pixmo example {i+1}/{len(batch)}: Hash={byte_hash[:8]}..., Label={label}, Count={ground_truth}")
            except Exception as e:
                print(f"Error processing pixmo example {i}: {e}")
                # Skip on processing failure
                continue
                
        return messages_list, processed_samples


def load_data(dataset_name, split="train", data_dir=None):
    # 3. Load dataset
    if dataset_name == "CLEVR":
        print("Loading CLEVR dataset...")
        dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB", split=split, cache_dir=data_dir)
        print(f"Dataset loaded, {len(dataset)} samples")

    elif dataset_name == "SAT":
        print("Loading SAT dataset...")
        # Try loading from local file
        sat_local_path = os.path.join(data_dir, "SAT", "SAT_train_15000.json")
        if os.path.exists(sat_local_path):
            print(f"Loading SAT dataset from local file: {sat_local_path}")
            with open(sat_local_path, "r", encoding="utf-8") as f:
                sat_data = json.load(f)

            dataset = SATDataset(sat_data)
            print(f"Dataset loaded, {len(dataset)} samples")
        else:
            # If local file not found, try loading from Hugging Face
            print("Local SAT dataset not found, trying to load from Hugging Face...")
            dataset = load_dataset("array/SAT", split=split, cache_dir=data_dir)
            print(f"Dataset loaded, {len(dataset)} samples")
        
    elif dataset_name == "pixmo":
        print("Loading pixmo dataset...")
        dataset = load_dataset("allenai/pixmo-count", split=split, cache_dir=data_dir)
        print(f"Dataset loaded, {len(dataset)} samples")
    elif dataset_name == "CVBENCH":
        print("Loading CVBENCH dataset...")
        dataset = load_dataset("nyu-visionx/CV-Bench", "2D")
        print(f"Dataset loaded, {len(dataset)} samples")
    return dataset
