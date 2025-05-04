import sys
import os
sys.path.append(os.path.abspath("big_vision"))

import json
import io
import warnings
warnings.filterwarnings("ignore")
import jax
jax.config.update("jax_platform_name", "gpu")
import jax.numpy as jnp
import numpy as np
import ml_collections
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import sentencepiece
import functools
from PIL import Image
import re
from word2number import w2n

from datasets import load_dataset
from tqdm import tqdm

from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns

import logging
logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

print(f"JAX version:  {jax.__version__}")
print(f"JAX devices:  {jax.device_count()}")

# --- Setup ---
LLM_VARIANT = "gemma2_2b"
MODEL_PATH = "./checkpoints/paligemma/paligemma2-3b-mix-448.b16.npz"
TOKENIZER_PATH = "./checkpoints/paligemma/paligemma_tokenizer.model"
model_config = ml_collections.FrozenConfigDict({
    "llm": {"vocab_size": 257_152, "variant": LLM_VARIANT, "final_logits_softcap": 0.0},
    "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
})
model = paligemma.Model(**model_config)
tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)

params = paligemma.load(None, MODEL_PATH, model_config)

decode_fn = predict_fns.get_all(model)['decode']
decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id())

def preprocess_image(image, size=448):
    image = np.asarray(image)
    image = tf.constant(image)
    image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
    return image.numpy().astype(np.float16) / 127.5 - 1.0 # [0, 255]->[-1,1]

def preprocess_tokens(prefix, suffix=None, seqlen=None):
    separator = "\n"
    tokens = tokenizer.encode(prefix, add_bos=True) + tokenizer.encode(separator)
    mask_ar = [0] * len(tokens)    # 0 to use full attention for prefix.
    mask_loss = [0] * len(tokens)  # 0 to not use prefix tokens in the loss.

    if suffix:
        suffix = tokenizer.encode(suffix, add_eos=True)
        tokens += suffix
        mask_ar += [1] * len(suffix)    # 1 to use causal attention for suffix.
        mask_loss += [1] * len(suffix)  # 1 to use suffix tokens in the loss.

    mask_input = [1] * len(tokens)    # 1 if its a token, 0 if padding.
    if seqlen:
        padding = [0] * max(0, seqlen - len(tokens))
        tokens = tokens[:seqlen] + padding
        mask_ar = mask_ar[:seqlen] + padding
        mask_loss = mask_loss[:seqlen] + padding
        mask_input = mask_input[:seqlen] + padding

    return jax.tree.map(np.array, (tokens, mask_ar, mask_loss, mask_input))

def postprocess_tokens(tokens):
    tokens = tokens.tolist()  # np.array to list[int]
    try:  # Remove tokens at and after EOS if any.
        eos_pos = tokens.index(tokenizer.eos_id())
        tokens = tokens[:eos_pos]
    except ValueError:
        pass
    return tokenizer.decode(tokens)

def extract_gt_answer(gt_text):
    gt_text = gt_text.lower().strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", gt_text)
    if match:
        return match.group(1).strip().lower()
    return gt_text.strip().lower()

def extract_choice_answer(text):
    text = text.lower().strip()
    pattern1 = r'\(([a-e])\)'  #  (A), (B), (C) 
    pattern2 = r'\b([a-e])\b'  # A, B, C 
    
    match = re.search(pattern1, text)
    if match:
        return match.group(1).lower()  
    
    match = re.search(pattern2, text)
    if match:
        return match.group(1).lower()
    
    return text 

def normalize_answer(text, task_type=None):
    if not text:
        return ""
    
    text = text.lower().strip()
    
    if task_type in ['Count', 'Relation', 'Distance', 'Depth']:
        choice = extract_choice_answer(text)
        if choice and choice.lower() in "abcde":
            return choice.lower() 

    # numbers = re.findall(r"\d+", text)
    # if numbers:
    #     return numbers[0] 
    
    # if "yes" in text:
    #     return "yes"
    # elif "no" in text:
    #     return "no"
    
    # return text

def evaluate_batch(batch, batch_size, seqlen=256):
    images_raw = batch["image"]  # List of PIL images
    questions = batch["problem"] if "problem" in batch else batch["prompt"]
    gt_answers_raw = batch["solution"] if "solution" in batch else batch["answer"]
    gt_answers = [str(ans).lower() for ans in gt_answers_raw]

    images = [preprocess_image(image.convert("RGB") if hasattr(image, 'convert') else image) for image in images_raw]

    tokens_list, mask_ar_list, mask_loss_list, mask_input_list = [], [], [], []
    for question, answer in zip(questions, gt_answers):
        tokens, mask_ar, mask_loss, mask_input = preprocess_tokens(
            prefix=question, suffix=None, seqlen=seqlen
        )
        tokens_list.append(np.asarray(tokens))
        mask_ar_list.append(np.asarray(mask_ar))
        mask_loss_list.append(np.asarray(mask_loss))
        mask_input_list.append(np.asarray(mask_input))

    batch_dict = {
        "image": np.stack([np.asarray(img) for img in images]),  # (B, 448, 448, 3)
        "text": np.stack(tokens_list),                            # (B, SEQLEN)
        "mask_ar": np.stack(mask_ar_list),
        "mask_loss": np.stack(mask_loss_list),
        "mask_input": np.stack(mask_input_list),
        "_mask": np.array([True] * batch_size),                  # (B,)
    }

    tokens = decode({"params": params}, batch=batch_dict,
                    max_decode_len=seqlen, sampler="greedy")
    tokens, mask = jax.device_get((tokens, batch_dict["_mask"]))
    responses_batch = [postprocess_tokens(t) for t in tokens]
    
    return responses_batch, gt_answers, questions

def compute_accuracy(responses, ground_truths, task_type=None):
    assert len(responses) == len(ground_truths)
    correct = 0
    
    for pred, gt in zip(responses, ground_truths):
        pred_norm = normalize_answer(pred, task_type)
        gt_norm = normalize_answer(extract_gt_answer(gt), task_type)

        if pred_norm == gt_norm:
            correct += 1
        # else:
            # print(pred,gt)
            # print(pred_norm,gt_norm)
            # logger.debug(f"❌ Wrong — Pred: '{pred}' | GT: '{gt}' → ({pred_norm} ≠ {gt_norm})")
            
    return correct / len(responses)


def evaluate_cvbench():

    print("\n=== Evaluating CVBench dataset ===")

    try:
        ds = load_dataset("nyu-visionx/CV-Bench") 
        print(f"Loaded CVBench dataset with {len(ds['test'])} test samples")
    except Exception as e:
        print(f"Error loading CVBench dataset: {e}")
        return {}
    

    tasks = ['Count', 'Relation', 'Distance', 'Depth']
    # tasks = ['Count']
    
    task_data = {}
    for task in tasks:
        task_data[task] = [item for item in ds['test'] if item['task'] == task]
        print(f"Found {len(task_data[task])} samples for {task} task")
    

    results = {}
    batch_size = 5
    SEQLEN = 256
    
    for task in tasks:
        print(f"\n--- Evaluating {task} task ---")
        task_samples = task_data[task]
        
        num_samples = (len(task_samples) // batch_size) * batch_size
        if num_samples == 0:
            print(f"Skipping {task} task: not enough samples")
            continue
            
        task_samples = task_samples[:num_samples]
        
        responses = []
        ground_truths = []
        
        for i in tqdm(range(0, len(task_samples), batch_size)):
            batch = task_samples[i:i+batch_size]
            
            batch_dict = {
                "image": [item["image"] for item in batch],
                "prompt": [item["prompt"] for item in batch],
                "answer": [item["answer"] for item in batch]
            }
            
            responses_batch, gt_answers, questions = evaluate_batch(batch_dict, batch_size, SEQLEN)
            
            # for q, gt, pred in zip(questions, gt_answers, responses_batch):
            #     logger.info(f"Q: {q} | GT: {gt} | Pred: {pred}")
            
            responses.extend(responses_batch)
            ground_truths.extend(gt_answers)

            if i%100 ==0:
                accuracy = compute_accuracy(responses, ground_truths, task)
                print(f"{task} Accuracy: {accuracy:.2%}")
            
        accuracy = compute_accuracy(responses, ground_truths, task)
        print(f"{task} Accuracy: {accuracy:.2%}")
        results[task] = accuracy
    
    if results:
        overall_accuracy = sum(results.values()) / len(results)
        print(f"\nCVBench Overall Accuracy: {overall_accuracy:.2%}")
        results["overall"] = overall_accuracy
    
    return results


if __name__ == "__main__":
    
    results = {}
    results['cvbench'] = evaluate_cvbench()
    
    print("\n=== Summary of Results ===")
    for dataset, result in results.items():
        if isinstance(result, dict):
            print(f"{dataset} results:")
            for task, acc in result.items():
                print(f"  {task}: {acc:.2%}")
        else:
            print(f"{dataset}: {result:.2%}")