# Core Concepts — LLM Fine-Tuning with QLoRA

> Your personal reference for every concept used in this project.
> Written to be understood first, memorized second.

---

## Table of Contents

1. [What is a Language Model?](#1-what-is-a-language-model)
2. [What is a Transformer?](#2-what-is-a-transformer)
3. [What is Tokenization?](#3-what-is-tokenization)
4. [Pre-training vs Fine-tuning](#4-pre-training-vs-fine-tuning)
5. [Why Not Full Fine-tuning?](#5-why-not-full-fine-tuning)
6. [What is PEFT?](#6-what-is-peft)
7. [What is LoRA?](#7-what-is-lora)
8. [What is QLoRA?](#8-what-is-qlora)
9. [What is Instruction Tuning?](#9-what-is-instruction-tuning)
10. [The Training Loop (PyTorch)](#10-the-training-loop-pytorch)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Glossary](#12-glossary)

---

## 1. What is a Language Model?

A language model is a system trained to **predict the next word** (or token) in a sequence.

Given: `"The cat sat on the"`
It predicts: `"mat"` (or `"floor"`, `"chair"`, etc. with probabilities)

That's it. Everything GPT-4, TinyLlama, and other LLMs do — answering questions,
writing code, summarizing text — comes from being extremely good at this one task,
trained on hundreds of billions of words.

**Key insight**: A model that can predict the next word perfectly must *understand*
language, facts, reasoning, and context. That understanding emerges from scale.

---

## 2. What is a Transformer?

The architecture behind every modern LLM. Introduced in the paper
*"Attention Is All You Need"* (Vaswani et al., 2017).

### The core idea — Self-Attention

When processing a sentence, every word looks at every other word and decides
how much to "pay attention" to it.

Example: `"The bank on the river bank was flooded"`

When processing the second "bank", the model attends strongly to "river" to understand
it means a riverbank, not a financial bank.

### Layers in a Transformer Block

```
Input Tokens
     ↓
[Self-Attention Layer]   ← where LoRA is applied
     ↓
[Feed-Forward Layer]
     ↓
Output (prediction for next token)
```

Each transformer block has these two components. TinyLlama-1.1B has **22 of these blocks**
stacked on top of each other.

### Attention Projections (q, k, v, o)

Inside self-attention, each token is transformed into three vectors:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I pass along?"
- **O (Output)**: Final output projection

These are implemented as linear layers (weight matrices). This is exactly where
**LoRA injects its adapters** — into `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

---

## 3. What is Tokenization?

Models don't read text. They read **tokens** — chunks of text mapped to integers.

```
Text:   "Hello, how are you?"
Tokens: [15043, 29892, 920, 526, 366, 29973]
```

A **tokenizer** converts text ↔ token IDs. Each model has its own tokenizer
trained alongside it, so you must always use the tokenizer that matches the model.

### Why tokens, not words?

- Handles unknown words ("ChatGPT" → ["Chat", "G", "PT"])
- More efficient than character-by-character
- Subword tokenization captures morphology ("running" → ["run", "ning"])

### Special tokens

```
<s>       ← Beginning of sequence
</s>      ← End of sequence
<pad>     ← Padding (makes all sequences same length in a batch)
```

**In code:**
```python
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokens = tokenizer("Hello world", return_tensors="pt")
# tokens["input_ids"] → tensor([[1, 15043, 3186]])
```

---

## 4. Pre-training vs Fine-tuning

### Pre-training

Training a model **from scratch** on a massive dataset (e.g., the entire internet).

- Dataset: trillions of tokens
- Time: weeks to months on thousands of GPUs
- Cost: millions of dollars
- Result: a general-purpose model that understands language

TinyLlama was pre-trained on **3 trillion tokens**. We do NOT do this.

### Fine-tuning

Taking a pre-trained model and **continuing to train it** on a smaller,
task-specific dataset.

- Dataset: thousands to millions of examples
- Time: minutes to hours on a single GPU
- Cost: near zero (with PEFT)
- Result: a model specialized for your task

**Analogy**: Pre-training is like getting a university education.
Fine-tuning is like on-the-job training at a specific company.
The general knowledge transfers; you learn the specifics quickly.

---

## 5. Why Not Full Fine-tuning?

TinyLlama has **1.1 billion parameters**. Each parameter is a floating-point number.

In full fine-tuning, **every parameter gets updated** on every training step.

### The memory problem

| What | Memory Required |
|---|---|
| Model weights (float32) | ~4.4 GB |
| Gradients (same size as weights) | ~4.4 GB |
| Optimizer states (Adam uses 2×) | ~8.8 GB |
| **Total** | **~17.6 GB** |

A free Colab T4 GPU has **15 GB VRAM**. Full fine-tuning won't even fit.

For larger models (7B, 13B, 70B), this becomes completely infeasible without
multiple A100 GPUs costing thousands of dollars.

**Solution**: PEFT — train only a tiny fraction of the parameters.

---

## 6. What is PEFT?

**Parameter-Efficient Fine-Tuning** — a family of techniques that fine-tune
a small number of parameters while keeping the base model frozen.

### Why it works

The pre-trained model already contains enormous amounts of knowledge.
Fine-tuning doesn't need to relearn everything — it just needs to **redirect**
that knowledge toward a specific task.

PEFT methods inject a small number of trainable parameters that act as
"steering wheels" on top of the frozen base model.

### PEFT methods comparison

| Method | How it works | Params trained |
|---|---|---|
| **LoRA** | Low-rank adapter matrices in attention layers | ~0.1–1% |
| Prefix Tuning | Prepend trainable tokens to input | ~0.1% |
| Prompt Tuning | Train soft prompt embeddings | ~0.01% |
| **QLoRA** | LoRA + 4-bit quantization | ~0.1–1% |

We use **QLoRA** — the most memory-efficient option.

---

## 7. What is LoRA?

**Low-Rank Adaptation** (Hu et al., 2022). The core idea behind our fine-tuning.

### The math (simplified)

Each attention layer has a weight matrix **W** (e.g., 2048 × 2048 = 4M parameters).

In full fine-tuning, we update W directly:
```
W_new = W + ΔW        ← ΔW is huge (same size as W)
```

LoRA's insight: **ΔW is low-rank**. It doesn't need to be a full 2048×2048 matrix.
It can be approximated by two smaller matrices:

```
ΔW ≈ A × B

Where:
  A is (2048 × r)    ← r is the "rank" (e.g., 8)
  B is (r × 2048)

Instead of 4,194,304 parameters → only 2 × (2048 × 8) = 32,768 parameters
That's 128× fewer parameters to train!
```

### During training

- **W (base model)** → frozen, never updated
- **A and B (LoRA adapters)** → trained

### During inference

The adapter is merged back:
```
output = (W + A × B) × input
```
No extra computation cost at inference time.

### The rank `r` — intuition

- `r = 4`: Minimal capacity, fast, less expressive
- `r = 8`: Balanced (what we use as default)
- `r = 16`: More capacity, slower, can capture more complex changes

Our ablation study tests all three.

### `lora_alpha` — scaling

```
effective_delta_W = (lora_alpha / r) × A × B
```

Setting `lora_alpha = 2 × r` (e.g., alpha=16 when r=8) is the standard recipe.
It ensures the adapter's contribution is properly scaled regardless of `r`.

---

## 8. What is QLoRA?

**Quantized LoRA** (Dettmers et al., 2023). LoRA + aggressive memory compression.

### The problem LoRA alone doesn't solve

LoRA reduces the *trainable* parameters. But the **frozen base model still takes memory**.
TinyLlama in float32 = ~4.4 GB. In float16 = ~2.2 GB. Still significant.

### What quantization does

Quantization reduces the **precision** of the model weights.

```
float32 → 32 bits per parameter  (full precision)
float16 → 16 bits per parameter  (half precision)
int8    →  8 bits per parameter  (8-bit quantization)
nf4     →  4 bits per parameter  (QLoRA uses this)
```

TinyLlama in nf4 = **~0.6 GB** — nearly 7× smaller than float32!

### NF4 — NormalFloat4

A special 4-bit format designed for neural network weights.
Unlike regular 4-bit integers, NF4 is **information-theoretically optimal**
for normally distributed weights (which most neural network weights are).

### Double quantization

QLoRA also quantizes the quantization constants themselves (meta-compression),
saving an additional ~0.4 GB.

### The full QLoRA picture

```
Base Model Weights  →  4-bit NF4 (frozen, ~0.6 GB)
                              +
LoRA Adapters A, B  →  float16 (trainable, ~30 MB)
                              ↓
           Total GPU memory: ~4-6 GB  ✅ fits on T4!
```

---

## 9. What is Instruction Tuning?

Teaching a model to follow instructions in a structured format.

### The problem with raw pre-training

A base LLM trained on internet text will complete text, not follow instructions:

```
Prompt:  "What is the capital of France?"
Base LLM output: "What is the capital of Germany? What is the capital of..."
                 (it continues the "quiz" pattern it saw in training data)
```

### The solution — instruction dataset

We show the model thousands of (instruction, response) pairs:

```
### Instruction:
What is the capital of France?

### Response:
The capital of France is Paris.
```

After seeing enough of these, the model learns the pattern:
"When I see `### Instruction:`, I should answer helpfully after `### Response:`"

### The Stanford Alpaca dataset

52,000 instruction-following examples generated by GPT-3.5 and cleaned.
Format:

```json
{
  "instruction": "Give three tips for staying healthy.",
  "input": "",
  "output": "1. Eat a balanced diet...\n2. Exercise regularly...\n3. Get enough sleep..."
}
```

When `input` is empty, the prompt template becomes:
```
Below is an instruction. Write a response that appropriately completes it.

### Instruction:
Give three tips for staying healthy.

### Response:
```

---

## 10. The Training Loop (PyTorch)

What actually happens when you call `trainer.train()`.

### One training step — step by step

```python
# 1. FORWARD PASS — run input through the model
outputs = model(input_ids=batch["input_ids"],
                labels=batch["labels"])
loss = outputs.loss   # How wrong was the model?

# 2. BACKWARD PASS — compute gradients
loss.backward()        # PyTorch traces back through every operation
                       # and computes: "how should each parameter change
                       # to reduce this loss?"

# 3. OPTIMIZER STEP — update parameters
optimizer.step()       # Apply the gradients (move parameters in the
                       # direction that reduces loss)

# 4. ZERO GRADIENTS — clear for next step
optimizer.zero_grad()  # Gradients accumulate by default — must reset
```

### Key concepts

**Loss**: A number measuring how wrong the model's predictions are.
Lower = better. Cross-entropy loss for language models:
"How surprised was the model by the correct next token?"

**Gradient**: The direction and magnitude to change each parameter
to reduce loss. Computed automatically by PyTorch's autograd engine.

**Learning rate**: How big of a step to take in the gradient direction.
Too high → training becomes unstable. Too low → training is very slow.

**Batch**: A group of training examples processed together.
Batch size of 4 = model sees 4 examples before updating weights.

**Epoch**: One complete pass through the entire training dataset.
We train for 3 epochs = the model sees each example 3 times.

**Gradient accumulation**: Process multiple batches before updating weights.
`gradient_accumulation_steps=4` with `batch_size=4` → effective batch size = 16.
Simulates a larger batch without requiring more GPU memory.

### Cosine learning rate schedule

We don't use a fixed learning rate. It follows a cosine curve:
```
High (warmup) → Peak → Gradually decreasing → Near zero (end)
```
This helps the model make bold updates early and fine adjustments later.

---

## 11. Evaluation Metrics

### Training Loss

The loss value during training. Should decrease over time.

```
Epoch 1: loss = 2.1
Epoch 2: loss = 1.6
Epoch 3: loss = 1.3  ← Good: learning is happening
```

**Overfitting**: Loss on training data keeps dropping but validation loss rises.
The model has memorized training examples instead of learning to generalize.

### ROUGE Score

**R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation.
Measures overlap between generated text and reference text.

**ROUGE-1**: Overlap of individual words (unigrams)
```
Reference: "The cat sat on the mat"
Generated: "The cat is on the mat"
ROUGE-1 = 5/6 = 0.83  (5 words match out of 6 reference words)
```

**ROUGE-2**: Overlap of word pairs (bigrams)
```
Reference: "The cat sat on the mat"
Generated: "The cat is on the mat"
Bigrams in reference: [The cat, cat sat, sat on, on the, the mat]
Bigrams in generated: [The cat, cat is, is on, on the, the mat]
Matching: [The cat, on the, the mat] = 3
ROUGE-2 = 3/5 = 0.60
```

**ROUGE-L**: Longest common subsequence — captures sentence structure.

### Why ROUGE is not enough

ROUGE only measures word overlap. It misses:
- Paraphrases ("automobile" vs "car" scores 0 overlap)
- Instruction following (did the model use the right format?)
- Factual correctness

That's why we also do **qualitative evaluation** — manually reading and comparing outputs.

---

## 12. Glossary

| Term | Definition |
|---|---|
| **Parameter** | A single learnable number in the model (weight or bias) |
| **Tensor** | A multi-dimensional array — the core data structure in PyTorch |
| **VRAM** | Video RAM — GPU memory. Where models live during training |
| **Checkpoint** | A saved snapshot of model weights at a point in training |
| **Adapter** | The small trainable matrices LoRA injects into the model |
| **Frozen** | Base model weights that are NOT updated during training |
| **Inference** | Running the model to generate output (not training) |
| **Perplexity** | How "surprised" the model is by text — lower = better language model |
| **Token** | The atomic unit of text a model processes |
| **Context window** | Maximum number of tokens a model can process at once |
| **Gradient** | Direction to adjust parameters to reduce loss |
| **Backpropagation** | Algorithm to compute gradients through all model layers |
| **Adam optimizer** | The algorithm that uses gradients to update parameters |
| **Epoch** | One full pass through the training dataset |
| **Batch** | A group of examples processed together in one forward pass |
| **Overfitting** | Model memorizes training data, fails to generalize |
| **Quantization** | Reducing numerical precision of weights to save memory |
| **NF4** | NormalFloat4 — 4-bit format optimized for LLM weights |
| **HuggingFace Hub** | Public repository for sharing models and datasets |
| **Model card** | Documentation describing a model's training, capabilities, and limitations |
