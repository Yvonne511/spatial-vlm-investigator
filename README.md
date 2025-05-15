# 🧠 Enhancing Spatial Reasoning in Vision-Language Models via Chain-of-Thought Prompting and Reinforcement Learning

This repository contains the codebase and evaluation scripts for our paper:

**📄 [Paper (PDF)](./paper.pdf)**  

## 🏋️ GRPO and SFT Fine-tuning (on SAT Dataset)

We provide training scripts for **Group Relative Policy Optimization (GRPO)** and **Supervised Fine-Tuning (SFT)** under the `model/` directory.

| Script | Description |
|--------|-------------|
| `model/pali_sat_grpo.py` | Train PaLI-Gemma on SAT with GRPO (Reinforcement Learning) |
| `model/pali_sat_sft.py`  | Train PaLI-Gemma on SAT with standard Supervised Fine-Tuning |

Both scripts are standalone and can be executed directly.

### ▶️ Run GRPO Training

```bash
python model/pali_sat_grpo.py
### ▶️ Run SFT Training

```bash
python model/pali_sat_sft.py


# spatial-clip
Env only use v100 or T4
# Tests
Clevr_CoGenT_ValB spacial counting
https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_ValB

SuperClevr_Val super spacial counting on out of distribution objects
https://huggingface.co/datasets/MMInstruction/SuperClevr_Val

SAT 
https://huggingface.co/datasets/array/SAT static spacial reasoning

Clevr_CoGenT thinking
https://huggingface.co/datasets/ahmedheakl/clevr-cogent-r1

Clevr_CoGenT_ValA and Depth with Prompt
https://huggingface.co/datasets/erkam/clevr-with-depth
https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_ValA


