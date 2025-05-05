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
    level=logging.INFO,  # or DEBUG if you want more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")
backend = jax.extend.backend.get_backend()
print(f"JAX version:  {jax.__version__}")
print(f"JAX platform: {backend.platform}")
print(f"JAX devices:  {jax.device_count()}")

# --- Setup ---
LLM_VARIANT = "gemma2_2b"
MODEL_PATH = "/vast/yw4142/checkpoints/llvm/paligemma2-3b-mix-448.b16.npz"
TOKENIZER_PATH = "/vast/yw4142/checkpoints/llvm/paligemma_tokenizer.model"
model_config = ml_collections.FrozenConfigDict({
    "llm": {"vocab_size": 257_152, "variant": LLM_VARIANT, "final_logits_softcap": 0.0},
    "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
})
model = paligemma.Model(**model_config)
tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)

params = paligemma.load(None, MODEL_PATH, model_config)

decode_fn = predict_fns.get_all(model)['decode']
decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id())

# --- Eval ---
dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB", split="train")
# dataset = dataset.select(range(2000))  # For testing purposes
batch_size = 20
assert len(dataset) % batch_size == 0, "Dataset size must be divisible by batch size."

def preprocess_image(image, size=448):
    image = np.asarray(image)
    image = tf.constant(image)
    image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
    return image.numpy().astype(np.float16) / 127.5 - 1.0 # [0, 255]->[-1,1]

# def extract_answer(raw):
#     """Extracts the number inside <answer> tags."""
#     match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw)
#     return match.group(1).strip() if match else raw.strip()

def preprocess_tokens(prefix, suffix=None, seqlen=None):
    # Model has been trained to handle tokenized text composed of a prefix with
    # full attention and a suffix with causal attention.
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
    # Get value inside <answer> ... </answer>
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", gt_text)
    if match:
        return match.group(1).strip().lower()
    return gt_text.strip().lower()

def normalize_answer(text):
    # Lowercase and keep only alphanumerics and digits
    text = text.lower().strip()
    # Try to extract a number if present
    numbers = re.findall(r"\d+", text)
    if numbers:
        return numbers[0]  # assume single answer
    # Try yes/no logic
    if "yes" in text:
        return "yes"
    elif "no" in text:
        return "no"
    return text

def compute_accuracy(responses, ground_truths):
    assert len(responses) == len(ground_truths)
    correct = 0

    for pred, gt in zip(responses, ground_truths):
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(extract_gt_answer(gt))
        if pred_norm == gt_norm:
            correct += 1
        # else:
        #     print(f"❌ Wrong — Pred: '{pred}' | GT: '{gt}' → ({pred_norm} ≠ {gt_norm})")
    return correct / len(responses)

# SEQLEN = 128
SEQLEN = 256
responses = []
ground_truths = []
 
for i in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset[i:i+batch_size]  # This is a dict of columns

    # Unpack columns
    images_raw = batch["image"]  # List of PIL images
    questions = batch["problem"]
    gt_answers_raw = batch["solution"]
    gt_answers = [ans.lower() for ans in gt_answers_raw]

    # Preprocess images
    images = [preprocess_image(image.convert("RGB")) for image in images_raw]

    # Preprocess tokens per sample
    tokens_list, mask_ar_list, mask_loss_list, mask_input_list = [], [], [], []
    for question, answer in zip(questions, gt_answers):
        tokens, mask_ar, mask_loss, mask_input = preprocess_tokens(
            prefix=question, suffix=None, seqlen=SEQLEN
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
                    max_decode_len=SEQLEN, sampler="greedy")
    tokens, mask = jax.device_get((tokens, batch_dict["_mask"]))
    responses_batch = [postprocess_tokens(t) for t in tokens]
    
    for q, gt, pred in zip(questions, gt_answers, responses_batch):
        logger.info(f"Q: {q} | GT: {gt} | Pred: {pred}")
    responses.extend(responses_batch)
    ground_truths.extend(gt_answers)
correct = compute_accuracy(responses, ground_truths)
print(f"Accuracy: {correct:.2%}")
# Accuracy: 38.52%
