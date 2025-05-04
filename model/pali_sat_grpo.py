import os
import gc
import re
import json
import torch
import argparse
import transformers
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Union, Any, List, Dict
from collections import defaultdict
from packaging import version
from PIL import Image
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from datasets import Dataset, IterableDataset

from trl import GRPOConfig, ModelConfig, get_peft_config
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, callable]

dataset_prefix = "/content/drive/MyDrive/VLLM/SAT/"
dataset_path = "SAT_train_15000.json"

from huggingface_hub import login
login(token='')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def accuracy_reward(completions, solution, **kwargs):
    if isinstance(completions[0], str):
        contents = [completion for completion in completions]
    else:
        contents = [completion[0]["content"] for completion in completions]
    rewards = []
    correct_count = 0
    total_count = 0


    for content, sol in zip(contents, solution):
        total_count += 1
        # print(content, sol) 

        if content.strip() == "":
            reward = -0.2
        else:
            reward = 0.0
            content = re.sub(r"[\(\)\.\,\-]", "", content) 
            if content.lower() == sol.lower():
                reward = 1.0
                correct_count += 1 

        rewards.append(reward)

    return rewards


class PaliGemmaGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, List[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        pad_token_id: Optional[int] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        model_init_kwargs["torch_dtype"] = torch.bfloat16
        if isinstance(model, str):
            model_id = model
            # torch_dtype = model_init_kwargs.get("torch_dtype")
            # if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            #     pass  # torch_dtype is already a torch.dtype or "auto" or None
            # elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            #     torch_dtype = getattr(torch, torch_dtype)
            #     model_init_kwargs["torch_dtype"] = torch_dtype
            # else:
            #     raise ValueError(
            #         f"Invalid `torch_dtype` {torch_dtype}. Expected 'auto' or a torch.dtype string."
            #     )

            if "paligemma" in model_id.lower():
                # PaLI-Gemma do not accept use_cache
                if "use_cache" in model_init_kwargs:
                    model_init_kwargs.pop("use_cache")
                model = PaliGemmaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # PEFT
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "paligemma" in model_id.lower():
                self.ref_model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        if processing_class is None:
            if "Qwen2-VL" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
            elif "paligemma" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id

        self.reward_funcs = reward_funcs
        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        def data_collator(features):
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=0.9,
            # top_p=0.95,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True

        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        return inputs


    def normalize_answer(self, text):
        # text = text.lower().strip()
        if "\n" in text:
            text = text.split("\n")[-1].strip()
        return text

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPO Trainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]
        # print("prompts:")
        # print(prompts)

        prompt_texts = []
        for example in inputs:
            if isinstance(example["prompt"], list) and isinstance(example["prompt"][0], dict) and "content" in example["prompt"][0]:
                content_list = example["prompt"][0]["content"]
                text = ""
                for item in content_list:
                    if item["type"] == "text":
                        text = item["text"]
                        break
                prompt_texts.append(text)
            else:
                prompt_texts.append(str(example["prompt"]))

        images = []
        images = []
        for x in inputs:
            img_temp = Image.open(dataset_prefix + x["image_path"]).convert("RGB")
            images.append(img_temp)

        # for x in inputs:
        #     img_temp = x["image"].resize((384, 384), Image.Resampling.LANCZOS)
        #     images.append(img_temp)

        prompt_inputs = self.processing_class(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=True,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        # print(f"pixel_values shape: {pixel_values.shape}")
        image_grid_thw = prompt_inputs.get("image_grid_thw", None)


        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]


        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            # print("prompt_completion_ids:",prompt_completion_ids.size(1))
            # prompt_completion = self.processing_class.batch_decode(prompt_completion_ids, skip_special_tokens=True)
            # print(prompt_completion)
            prompt_length = prompt_ids.size(1)
            # print("prompt_length:",prompt_length)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            # completion_ = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            # print(len(completion_[0]))
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        pixel_values = torch.repeat_interleave(prompt_inputs["pixel_values"], repeats=self.num_generations, dim=0)

        if image_grid_thw is not None:
            image_grid_thw = torch.repeat_interleave(image_grid_thw, repeats=self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]


        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions_o = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # print("completions_o:")
        # print(completions_o)
        completions = [self.normalize_answer(text) for text in completions_o]

        # print("completions:")
        # print(completions)


        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]


        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]


        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)


        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):

                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:

                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, solution=reward_kwargs.get("solution", []))
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        current_rewards = rewards.clone().detach().cpu().numpy()
        correct_preds = (current_rewards == 1.0).sum()
        total_preds = len(current_rewards)
        batch_accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
        self._metrics["accuracy"].append(batch_accuracy)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        if std_grouped_rewards < 1e-5:
          if mean_grouped_rewards ==1:
            print("[GRPO] Skipping batch due to zero reward std: all correct!!!!!!!")
          elif mean_grouped_rewards ==0:
            print("[GRPO] Skipping batch due to zero reward std: all wrong.")
          else:
            print("[GRPO] Skipping batch due to zero reward std.")
          return torch.tensor(0.0, device=rewards.device, requires_grad=True)


        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # advantages = rewards - mean_grouped_rewards

        # print("advantage:",advantages)
        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # print("per_token_loss",per_token_loss)
        # print("per_token_kl",per_token_kl)
        # print("self.beta * per_token_kl",self.beta * per_token_kl)
        # print("per_token_loss",per_token_loss)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # print("loss_mean",loss)

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())




        # print("\n====== Debug Step ======")
        # print("per_token_logps:", per_token_logps.detach().cpu().numpy())  # 看前20个
        # print("advantages:", advantages.detach().cpu().numpy())  # 看前10个
        # print("per_token_kl:", per_token_kl.detach().cpu().numpy())
        # print("completion_mask:", completion_mask.detach().cpu().numpy())
        # print("rewards:", rewards.detach().cpu().numpy())
        # print("per_token_loss:", per_token_loss.detach().cpu().numpy())
        # print("final loss:", loss.item())
        # print("========================\n")


        print(f"Batch accuracy: {correct_preds}/{total_preds} = {batch_accuracy:.4f}")
        print(f"Actual loss value: {loss.item()}")


        torch.cuda.empty_cache()
        gc.collect()
        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

# def make_conversation_sat(example, base_model_prompt=False):
#     QUESTION_TEMPLATE = "{Question}"
#     image = Image.open(dataset_prefix + example["images"][0])
#     return {"image": image,
#         "image_path": example["images"][0],
#         "prompt": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image"},
#                     {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["messages"][0]["content"])},
#                 ],
#             },
#         ],
#         "solution":  example["messages"][1]["content"],}

def make_conversation_sat(example, base_model_prompt=False):
    QUESTION_TEMPLATE = "{Question}"
    return {
        "image_path": example["images"][0], 
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["messages"][0]["content"])},
                ],
            },
        ],
        "solution": example["messages"][1]["content"],
    }


def train_paligemma_sat(
    model_name="google/paligemma2-3b-mix-224",
    dataset_prefix = "/content/drive/MyDrive/VLLM/SAT/",
    dataset_path = "SAT_train_15000.json",
    output_dir="/content/drive/MyDrive/VLLM/paligemma-sat-grpo",
    num_train_epochs=1,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_steps=None,
    logging_steps=10,
    save_steps=1000,
    max_prompt_length=256,
    max_completion_length=128,
    num_generations=4,
    beta=0.01,
    train_samples=100, 
    use_wandb=True,
    attn_implementation="flash_attention_2",
    push_to_hub=False,
    seed=42,):



    os.environ["DEBUG_MODE"] = "true"
    os.environ["LOG_PATH"] = "grpo_paligemma_sat_training.log"
    torch.manual_seed(seed)


    model_args = ModelConfig(
        model_name_or_path=model_name,
        attn_implementation=attn_implementation,
        use_peft=True, 
        lora_r=8,      
        lora_alpha=16,
        lora_dropout=0.05,
        # lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_target_modules=["q_proj", "v_proj"],

    )


    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),  
        seed=seed,
        data_seed=seed,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        beta=beta,
        report_to="wandb" if use_wandb else "none",
        run_name=f"paligemma-sat-grpo-{datetime.now().strftime('%Y%m%d-%H%M')}",
        gradient_checkpointing=False,  
        push_to_hub=push_to_hub,
    )

    print("load data...")
    with open(dataset_prefix + dataset_path, 'r') as f:
      sat_dataset = json.load(f)

    if train_samples > 0 and train_samples < len(sat_dataset):
        sat_dataset = sat_dataset[:train_samples]


    processed_data = [make_conversation_sat(sample) for sample in sat_dataset]


    reward_funcs = [accuracy_reward]

    trainer = PaliGemmaGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=processed_data,
        peft_config=get_peft_config(model_args),
        attn_implementation=attn_implementation,
        max_pixels=401408,
        min_pixels=3136,
    )

    trainer.train()

    trainer.save_model(output_dir)

    if push_to_hub:

        trainer.push_to_hub()

    return trainer


# jupyter
def run_training():

    dataset_prefix = "/content/drive/MyDrive/VLLM/SAT/"
    dataset_path = "SAT_train_15000.json"

    trainer = train_paligemma_sat(
        model_name="google/paligemma2-3b-mix-224",
        dataset_prefix = dataset_prefix,
        dataset_path = dataset_path,
        output_dir="/content/drive/MyDrive/VLLM/paligemma-sat-grpo",
        num_train_epochs=1, 
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_steps=1000, 
        logging_steps=10,
        save_steps=50,
        max_prompt_length=2000,
        max_completion_length=128,
        num_generations=4,
        beta=0.04,
        train_samples=15000, 
        use_wandb=True,
        attn_implementation="flash_attention_2",
        push_to_hub=False,
        seed=42,
    )

    return trainer

run_training()
