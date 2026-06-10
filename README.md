# 🦙 LLM Fine-Tuning with QLoRA (PEFT)

> Fine-tuning **TinyLlama-1.1B** on Stanford Alpaca using **QLoRA** (4-bit quantization + LoRA)
> on a single T4 GPU — no expensive hardware required.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project demonstrates a complete, end-to-end pipeline for fine-tuning a large language model
using **Parameter-Efficient Fine-Tuning (PEFT)** — specifically the **QLoRA** method.

Instead of updating billions of parameters (which requires expensive A100 GPUs), QLoRA:
1. **Quantizes** the base model to 4-bit precision (reducing memory by ~75%)
2. **Injects low-rank adapter matrices** (LoRA) into attention layers
3. **Trains only the adapters** (~1% of total parameters) while the base model stays frozen

The result: fine-tuning a 1.1B parameter model on a **Google Colab GPU**.

---

## 📊 Results

> Results populated after training. See [`results/`](./results/) for plots and raw metrics.

| Metric | Base Model | Fine-Tuned (r=8) | Improvement |
|---|---|---|---|
| ROUGE-1 | — | — | — |
| ROUGE-2 | — | — | — |
| ROUGE-L | — | — | — |
| Instruction Format | ❌ | ✅ | — |

---

## 🔬 Ablation Study — LoRA Rank Comparison

One of the key experiments in this project: how does LoRA rank `r` affect quality vs. training time?

| LoRA Rank | Trainable Params | Training Time | ROUGE-L |
|---|---|---|---|
| r = 4 | — | — | — |
| r = 8 | — | — | — |
| r = 16 | — | — | — |

> See [`notebooks/05_ablation_study.ipynb`](./notebooks/05_ablation_study.ipynb) for the full experiment.

---

## 🗂️ Project Structure

```
llm-finetuning-peft/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Explore and understand the Alpaca dataset
│   ├── 02_baseline_inference.ipynb    # Run base model BEFORE fine-tuning
│   ├── 03_qlora_finetuning.ipynb      # Apply LoRA + train the model
│   ├── 04_evaluation.ipynb            # ROUGE scores, loss curves, qualitative analysis
│   ├── 05_ablation_study.ipynb        # Compare LoRA rank r=4 vs r=8 vs r=16
│   └── 06_gradio_demo.ipynb           # Interactive before/after demo
├── src/
│   ├── data_utils.py                  # Dataset loading and prompt formatting
│   ├── model_utils.py                 # Model + tokenizer loading, LoRA config
│   ├── train.py                       # Standalone training script
│   ├── evaluate.py                    # ROUGE evaluation pipeline
│   └── inference.py                   # Load fine-tuned model and run inference
├── configs/
│   └── qlora_config.yaml              # All hyperparameters in one place
├── results/                           # Training plots and metric outputs
├── assets/                            # Architecture diagrams
├── docs/
│   ├── concepts.md                    # Key ML concepts explained
│   └── model_card.md                  # HuggingFace-style model card
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/llm-finetuning-peft.git
cd llm-finetuning-peft
```

### 2. Open in Google Colab
All notebooks are designed to run on **Google Colab (free T4 GPU)**.

Click the badge at the top of any notebook, or go to:
[colab.research.google.com](https://colab.research.google.com) → File → Open from GitHub

### 3. Install dependencies (inside Colab)
```python
!pip install -r requirements.txt
```

### 4. Run notebooks in order
Start with `01_data_exploration.ipynb` and work through sequentially.

---

## 🧠 Key Concepts Covered

| Concept | Where Covered |
|---|---|
| Pre-trained LLMs & Tokenizers | `02_baseline_inference.ipynb` |
| Parameter-Efficient Fine-Tuning (PEFT) | `03_qlora_finetuning.ipynb` |
| LoRA — Low-Rank Adaptation | `03_qlora_finetuning.ipynb` + `docs/concepts.md` |
| 4-bit Quantization (QLoRA) | `03_qlora_finetuning.ipynb` |
| Instruction Tuning | `01_data_exploration.ipynb` |
| ROUGE Evaluation | `04_evaluation.ipynb` |
| PyTorch Training Loop | `03_qlora_finetuning.ipynb` |
| Ablation Studies | `05_ablation_study.ipynb` |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `torch` (PyTorch) | Core ML framework |
| `transformers` | Load TinyLlama + tokenizer |
| `datasets` | Load Stanford Alpaca dataset |
| `peft` | Apply LoRA adapters |
| `trl` | SFTTrainer for supervised fine-tuning |
| `bitsandbytes` | 4-bit quantization |
| `accelerate` | GPU device management |
| `evaluate` + `rouge-score` | Compute ROUGE metrics |
| `gradio` | Interactive demo UI |

---

## 📁 Model & Adapter

The fine-tuned LoRA adapter is published on HuggingFace Hub:
🤗 [YOUR_USERNAME/tinyllama-alpaca-qlora](https://huggingface.co/YOUR_USERNAME/tinyllama-alpaca-qlora)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/tinyllama-alpaca-qlora")
```

---

## 📚 Further Reading

- [LoRA Paper (Hu et al., 2022)](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
- [Stanford Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca)

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.
