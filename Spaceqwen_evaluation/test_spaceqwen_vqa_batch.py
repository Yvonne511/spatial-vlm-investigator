import argparse
import os
import json
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
import re
import sys

# Add data directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "data"))

# Import our custom modules
from data.process_dataset import (
    SATLocalDataset, create_sat_dataloader, evaluate_response, 
    process_vision_info, extract_number_from_answer
)

def main():
    # Set command line arguments
    parser = argparse.ArgumentParser(description="Test Qwen model performance on different datasets")
    parser.add_argument('--dataset', type=str, default='CLEVR', choices=['CLEVR', 'SAT'], 
                        help="Choose the dataset to test: CLEVR or SAT")
    parser.add_argument('--samples', type=int, default=20, help="Number of samples to test")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch inference size")
    parser.add_argument('--threshold', type=int, default=1, help="Small error threshold")
    parser.add_argument('--counting_only', action='store_true', 
                        help="Test only counting questions")
    parser.add_argument('--start_sample', type=int, default=0,
                        help="Start testing from which sample")
    parser.add_argument('--sat_json', type=str, default=None,
                        help="SAT dataset JSON file path (if not provided, will use default path)")
    parser.add_argument('--sat_image_dir', type=str, default=None,
                        help="SAT image directory path (if not provided, will use default path)")
    args = parser.parse_args()
    
    # 1. Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    data_dir = os.path.join(current_dir, "data")
    results_dir = os.path.join(current_dir, "results")

    # Ensure folders exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Set environment variables to force local cache
    os.environ["TRANSFORMERS_CACHE"] = models_dir
    os.environ["HF_HOME"] = models_dir
    os.environ["HF_HUB_CACHE"] = models_dir
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_DATASETS_CACHE"] = data_dir

    # 2. Load model
    #model_name = "remyxai/SpaceQwen2.5-VL-3B-Instruct"
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"Loading Qwen vision model: {model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    # 3. Load dataset
    if args.dataset == "CLEVR":
        print("Loading CLEVR dataset...")
        from datasets import load_dataset
        dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB", split="train", cache_dir=data_dir)
        print(f"Dataset loaded, total {len(dataset)} samples")

        # CLEVR batch inference processing function
        def process_batch(batch):
            # Construct messages
            messages_list = []
            for sample in batch:
                image = sample['image']
                question = sample['problem']
                # Explicitly require output <answer> tag
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
            return messages_list, batch  # Return messages_list and original batch

    elif args.dataset == "SAT":
        print("Loading SAT dataset...")
        
        # Set SAT dataset path
        sat_json = args.sat_json if args.sat_json else os.path.join(data_dir, "SAT", "SAT_train_15000.json")
        sat_image_dir = args.sat_image_dir if args.sat_image_dir else os.path.join(data_dir, "SAT")
        
        # Use our custom data loader to load SAT dataset
        if os.path.exists(sat_json):
            print(f"Using custom data loader to load SAT dataset: {sat_json}")
            
            # Create a DataLoader, but we don't use it directly for iteration
            # Instead, we only use its dataset attribute to access data
            dataloader = create_sat_dataloader(
                json_file=sat_json,
                image_dir=sat_image_dir,
                batch_size=args.batch_size,
                num_workers=0,  # Single process loading to avoid multi-process memory issues
                shuffle=False   # Do not shuffle data order
            )
            
            # Get dataset from DataLoader
            dataset = dataloader.dataset
            print(f"Dataset loaded, total {len(dataset)} samples")
            
            # SAT batch inference processing function
            def process_batch(batch):
                # The batch here is already a list of samples from the dataset
                # Each sample already contains all necessary fields
                
                # Construct messages
                messages_list = []
                processed_batch = []
                
                for sample in batch:
                    # Get image and question
                    image = sample['image']
                    question = sample['question']
                    
                    # Construct prompt
                    prompt_suffix = ""
                    if sample.get('is_counting', False):
                        prompt_suffix = " Please make sure to include the numeric answer."
                    
                    prompt = f"{question}{prompt_suffix} Please respond with the answer wrapped in <answer> </answer> tags, e.g. <answer>3</answer>; <answer>42</answer>; <answer>7.5</answer>."
                    
                    # Construct message
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
                    processed_batch.append(sample)
                
                return messages_list, processed_batch
                
        else:
            print(f"Error: SAT dataset JSON file not found: {sat_json}")
            return

    # 4. Batch inference parameters
    batch_size = args.batch_size  # Number of samples per inference
    start_idx = args.start_sample  # Start testing from which sample
    #max_samples = min(args.samples, len(dataset))  # Total number of samples to infer
    max_samples = len(dataset)
    small_diff_threshold = args.threshold

    results = []
    
    # Statistics variables
    total_samples = 0
    invalid_answers = 0
    counting_questions = 0
    correct_counting = 0     # Number of completely correct counting questions
    correct_within_threshold = 0  # Number of counting questions correct within threshold
    wrong_counting = 0
    num_diff_LE_smt = 0
    num_diff_grthan_smt = 0
    other_questions = 0

    print(f"\nStart batch testing {max_samples} samples, batch size {batch_size} ...")
    print(f"Start from sample {start_idx} ...")

    # Modify to start from start_idx, end at start_idx + max_samples
    end_idx = min(start_idx + max_samples, len(dataset))
    
    for batch_start in tqdm(range(start_idx, end_idx, batch_size), desc="Batch inference"):
        batch_end = min(batch_start + batch_size, end_idx)
        
        # Collect samples for this batch
        batch = [dataset[i] for i in range(batch_start, batch_end)]
        
        # Process batch inference for different datasets
        messages_list, processed_batch = process_batch(batch)

        # Batch process chat template and vision info
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        image_inputs_list = []
        video_inputs_list = []
        for m in messages_list:
            image_inputs, video_inputs = process_vision_info(m)
            image_inputs_list.append(image_inputs)
            video_inputs_list.append(video_inputs)

        # Check if all video_inputs are None
        if all(v is None for v in video_inputs_list):
            inputs = processor(
                text=texts,
                images=image_inputs_list,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=texts,
                images=image_inputs_list,
                videos=video_inputs_list,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Save results
        for idx, (sample, output_text) in enumerate(zip(processed_batch, output_texts)):
            global_idx = batch_start + idx
            
            # Display results
            print(f"\nSample {global_idx + 1}:")
            print(f"Question: {sample['problem']}")
            print(f"Reference answer: {sample['solution']}")
            print(f"Model answer: {output_text}")
            
            # Evaluate answer
            current_sample, question_type, is_correct = evaluate_response(
                output_text, sample, args.dataset, small_diff_threshold, global_idx
            )
            results.append(current_sample)
            
            # Update statistics
            total_samples += 1
            
            if args.dataset == "SAT" or args.dataset == "CLEVR":
                if question_type == "invalid":
                    invalid_answers += 1
                elif question_type == "counting":
                    counting_questions += 1
                    if current_sample.get("completely_correct", False):
                        correct_counting += 1
                        correct_within_threshold += 1
                    elif current_sample.get("correct_within_threshold", False):
                        correct_within_threshold += 1
                        wrong_counting += 1
                        diff = current_sample.get("diff", 0)
                        num_diff_LE_smt += 1
                    else:
                        wrong_counting += 1
                        diff = current_sample.get("diff", 0)
                        if diff <= small_diff_threshold:
                            num_diff_LE_smt += 1
                        else:
                            num_diff_grthan_smt += 1
                elif question_type == "other":
                    other_questions += 1
            
            # Print real-time statistics after current batch
            print("-" * 30)
            print(f"Samples processed: {total_samples}")
            print(f"Counting questions: {counting_questions}")
            print(f"Other type questions: {other_questions}")
            print(f"Invalid answers: {invalid_answers}")
            if counting_questions > 0:
                print(f"Counting questions exact accuracy: {correct_counting / counting_questions:.2%}")
                print(f"Counting questions within threshold accuracy: {correct_within_threshold / counting_questions:.2%}")
                print(f"Counting questions exact correct: {correct_counting}")
                print(f"Counting questions within threshold correct: {correct_within_threshold}")
                print(f"Counting questions not exact correct: {wrong_counting}")
                if wrong_counting > 0:
                    print(f"Counting questions with error <= {small_diff_threshold}: {num_diff_LE_smt}")
                    print(f"Counting questions with error > {small_diff_threshold}: {num_diff_grthan_smt}")
                    small_ratio = num_diff_LE_smt / wrong_counting if wrong_counting > 0 else 0
                    large_ratio = num_diff_grthan_smt / wrong_counting if wrong_counting > 0 else 0
                    print(f"Proportion of predictions with error <= {small_diff_threshold}: {small_ratio:.2%}")
                    print(f"Proportion of predictions with error > {small_diff_threshold}: {large_ratio:.2%}")
            
            print("=" * 50)
            
            # Periodically save interim results (every 10 samples)
            if total_samples % 10 == 0:
                # Create interim results directory
                interim_dir = os.path.join(results_dir, "interim")
                os.makedirs(interim_dir, exist_ok=True)
                
                # Statistics data
                if args.dataset == "SAT":
                    accuracy_stats = {
                        "total_samples": total_samples,
                        "counting_questions": counting_questions,
                        "other_questions": other_questions,
                        "invalid_answers": invalid_answers,
                        "exact_correct_counting": correct_counting,
                        "threshold_correct_counting": correct_within_threshold,
                        "wrong_counting": wrong_counting,
                        "small_diff_errors": num_diff_LE_smt,
                        "large_diff_errors": num_diff_grthan_smt,
                        "exact_counting_accuracy": correct_counting / counting_questions if counting_questions > 0 else 0,
                        "threshold_counting_accuracy": correct_within_threshold / counting_questions if counting_questions > 0 else 0,
                        "threshold": small_diff_threshold,
                    }
                    
                    # Build interim results filename
                    interim_filename = os.path.join(
                        interim_dir, 
                        f"sat_interim_results_{start_idx}-{global_idx}_smt{small_diff_threshold}.json"
                    )
                    
                    # Save interim results
                    interim_results = {
                        "statistics": accuracy_stats,
                        "results": results
                    }
                    
                    with open(interim_filename, "w", encoding="utf-8") as f:
                        json.dump(interim_results, f, ensure_ascii=False, indent=2)
                    
                    print(f"Interim results saved to: {interim_filename}")

    # 5. Save all results
    if args.dataset == "CLEVR":
        final_filename = os.path.join(results_dir, f"clevr_qwen_batch_results_{max_samples}_smt{small_diff_threshold}.json")
        accuracy_stats = {
            "total_samples": total_samples,
            "invalid_answers": invalid_answers,
            "counting_questions": counting_questions,
            "correct_counting": correct_counting,
            "wrong_counting": wrong_counting,
            "small_diff_errors": num_diff_LE_smt,
            "large_diff_errors": num_diff_grthan_smt,
            "counting_accuracy": correct_counting / counting_questions if counting_questions > 0 else 0,
            "threshold": small_diff_threshold,
        }
                
        print(f"\nStatistics:")
        print("-" * 50)
        print(f"Total samples: {total_samples}")
        print(f"Invalid answers: {invalid_answers}")
        print(f"Counting questions: {counting_questions}")
        if counting_questions > 0:
            print(f"Counting question accuracy: {accuracy_stats['counting_accuracy']:.2%}")
            print(f"Counting questions exact correct: {correct_counting}")
            print(f"Counting questions wrong: {wrong_counting}")
            if wrong_counting > 0:
                print(f"Counting questions with error <= {small_diff_threshold}: {num_diff_LE_smt}")
                print(f"Counting questions with error > {small_diff_threshold}: {num_diff_grthan_smt}")
            
    elif args.dataset == "SAT":
        # Build complete results filename, including sample range
        final_filename = os.path.join(
            results_dir, 
            f"sat_qwen_batch_results_{start_idx}-{start_idx+total_samples-1}_smt{small_diff_threshold}.json"
        )
        
        # Calculate counting question accuracy
        exact_accuracy = correct_counting / counting_questions if counting_questions > 0 else 0
        threshold_accuracy = correct_within_threshold / counting_questions if counting_questions > 0 else 0
        
        if wrong_counting > 0:
            ratio_small_diff = num_diff_LE_smt / wrong_counting
            ratio_large_diff = num_diff_grthan_smt / wrong_counting
        else:
            ratio_small_diff = 0
            ratio_large_diff = 0
        
        accuracy_stats = {
            "sample_range": f"{start_idx}-{start_idx+total_samples-1}",
            "total_samples": total_samples,
            "counting_questions": counting_questions,
            "other_questions": other_questions,
            "invalid_answers": invalid_answers,
            "exact_correct_counting": correct_counting,
            "threshold_correct_counting": correct_within_threshold,
            "wrong_counting": wrong_counting,
            "small_diff_errors": num_diff_LE_smt,
            "large_diff_errors": num_diff_grthan_smt,
            "exact_counting_accuracy": exact_accuracy,
            "threshold_counting_accuracy": threshold_accuracy,
            "ratio_small_diff": ratio_small_diff,
            "ratio_large_diff": ratio_large_diff,
            "threshold": small_diff_threshold,
        }
        
        print(f"\nStatistics:")
        print("-" * 50)
        print(f"Sample range: {start_idx}-{start_idx+total_samples-1}")
        print(f"Total samples: {total_samples}")
        print(f"Counting questions: {counting_questions}")
        print(f"Other type questions: {other_questions}")
        print(f"Invalid answers: {invalid_answers}")
        if counting_questions > 0:
            print(f"Counting questions exact accuracy: {exact_accuracy:.2%}")
            print(f"Counting questions within threshold accuracy: {threshold_accuracy:.2%}")
            print(f"Counting questions exact correct: {correct_counting}")
            print(f"Counting questions within threshold correct: {correct_within_threshold}")
            print(f"Counting questions not exact correct: {wrong_counting}")
            if wrong_counting > 0:
                print(f"Counting questions with error <= {small_diff_threshold}: {num_diff_LE_smt}")
                print(f"Counting questions with error > {small_diff_threshold}: {num_diff_grthan_smt}")
                print(f"Proportion of predictions with error <= {small_diff_threshold}: {ratio_small_diff:.2%}")
                print(f"Proportion of predictions with error > {small_diff_threshold}: {ratio_large_diff:.2%}")
            
    # Add statistics data to results
    results_with_stats = {
        "statistics": accuracy_stats,
        "results": results
    }

    with open(final_filename, "w", encoding="utf-8") as f:
        json.dump(results_with_stats, f, ensure_ascii=False, indent=2)

    print(f"\nAll results saved to: {final_filename}")
    print("-" * 50) 

if __name__ == "__main__":
    main() 