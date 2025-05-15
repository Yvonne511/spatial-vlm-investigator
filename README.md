# 🧠 Enhancing Spatial Reasoning in Vision-Language Models via Chain-of-Thought Prompting and Reinforcement Learning

This repository contains the codebase and evaluation scripts for our paper:

**📄 [Paper (PDF)](./paper.pdf)**  

# 🏋️ GRPO and SFT Fine-tuning (on SAT Dataset)

We provide training scripts for **Group Relative Policy Optimization (GRPO)** and **Supervised Fine-Tuning (SFT)** under the `model/` directory.

| Script | Description |
|--------|-------------|
| `model/pali_sat_grpo.py` | Train PaLI-Gemma on SAT with GRPO (Reinforcement Learning) |
| `model/pali_sat_sft.py`  | Train PaLI-Gemma on SAT with standard Supervised Fine-Tuning |

Both scripts are standalone and can be executed directly.
#### 🔧 Installation

```bash
pip install -r model/requirements.txt
```

#### ▶️ Run GRPO Training

```bash
python model/pali_sat_grpo.py
```
#### ▶️ Run SFT Training

```bash
python model/pali_sat_sft.py
```
#### 📊 CVBench Evaluation Results
| Model                                 | Counting<br>(Pass@4) | Relation<br>(Pass@4) | Depth<br>(Pass@4) | Distance<br>(Pass@4) | Total<br>(Pass@4) |
|:-------------------------------------:|:---------------------:|:---------------------:|:------------------:|:---------------------:|:------------------:|
| Pali-gemma2-3B-mix-224                | 64.00%                | 77.08%                | 51.83%             | 14.83%                | 52.16%             |
| Pali-gemma2-3B-mix-224-GRPO-v1        | 65.10%                | 78.41%                | 56.83%             | 18.00%                | 54.77%             |
| Pali-gemma2-3B-mix-224-SFT            | 65.07% (72.08%)   | 80.92% (82.92%)   | 61.67% (**91.17%**) | 59.50% (**90.00%**)   | 66.79% (**84.22%**) |
| Pali-gemma2-3B-mix-224-GRPO-v2        | **65.6%** (**73.07%**) | **84.92%** (**86.00%**) | **76.33%** (88.17%) | **62.67%** (78.00%) | **72.38%** (81.31%) |

#### 📊 OOD Generalization Results
| Model        | Depth(ID)| Depth(OOD)      | Distance(ID) | Distance(OOD)     | Distance ID-OOD Gap |
|--------------|------------|-------------------|----------------|---------------------|----------------------|
| Base Model   | 51.83%     | 59.00%             | 14.83%         | 5.83%               | -9.00%               |
| SFT Model    | 61.67%     | **56.33%** (-4.5%) 🔻 | 59.50%         | **47.47%** 🔻     | **12.03%** 🔻         |
| GRPO Model   | 76.33%     | **70.50%** (+19.5%) ✅ | 62.67%         | **59.50%** ✅     | **3.17%** ✅           |

# Evaluation
```bash
python eval_paligemma_clevr.py # eval on Clevr dataset
python eval_paligemma_cvbench.py # eval on CV-bench dataset
python eval_paligemma_vsr.py # eval on VSR dataset
python eval_paligemma_clevr_new.py # eval on generated Clevr dataset with prompts
```


# Clevr Description Prompt Generation
```bash
python clevr_desc/merge_clevr_w_depth_desc.py
```
## Testing with SpaceQwen

SpaceQwen is a model trained using a third-party generated dataset following Google's [SpatialVLM dataset scheme](https://github.com/remyxai/VQASynth/tree/main). The dataset was synthesized using the VQASynth pipeline, which enables the creation of spatial reasoning VQA datasets from arbitrary image collections. Our work focuses on the **counting aspect** of the trained model, evaluating its ability to perform spatial counting tasks.

For detailed instructions on how to evaluate SpaceQwen and reproduce our results, please refer to the [README `./Spaceqwen_evaluation/README.md`](https://github.com/Yvonne511/spatial-vlm-investigator/tree/main/Spaceqwen_evaluation).

# Env
Env setup: [Fine-Tuning PaliGemma](https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemma/docs/paligemma/fine-tuning-paligemma.ipynb).

Env only avaliable for v100 or T4

# Tests
Clevr_CoGenT_ValB spacial counting: https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_ValB

SuperClevr_Val (super spacial counting on out of distribution objects): https://huggingface.co/datasets/MMInstruction/SuperClevr_Val

SAT: https://huggingface.co/datasets/array/SAT static spacial reasoning

Clevr_CoGenT thinking: https://huggingface.co/datasets/ahmedheakl/clevr-cogent-r1

Clevr_CoGenT_ValA and Depth with Prompt:
- https://huggingface.co/datasets/erkam/clevr-with-depth
- https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_ValA


