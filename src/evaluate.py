"""
evaluate.py — Evaluation pipeline for the fine-tuned QLoRA model.

What this module does:
  1. Generates model outputs on a held-out test set
  2. Computes ROUGE scores (base model vs fine-tuned)
  3. Plots and saves the training loss curve
  4. Runs qualitative comparison (side-by-side output comparison)
  5. Saves all results to the results/ directory

Three evaluation methods — and why all three matter:

  ROUGE scores:
    Automated, objective, reproducible.
    Measures n-gram overlap with reference answers.
    Limitation: misses paraphrases and instruction-following style.

  Loss curve:
    Shows whether training was healthy (smooth decline)
    or unhealthy (spikes, plateau, divergence).
    Reveals overfitting if validation loss rises while training loss falls.

  Qualitative comparison:
    Human judgment — the ground truth.
    Did the model actually get better at following instructions?
    Does its output format match what was expected?
    Numbers alone can't answer this.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import evaluate as hf_evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Load Fine-tuned Model for Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
):
    """
    Loads the fine-tuned model by combining base model + LoRA adapter.

    This is different from training — during evaluation we load the
    adapter on top of the base model and run in inference mode.

    Args:
        base_model_name: HuggingFace ID of the base model.
        adapter_path:    Path to the saved LoRA adapter directory.

    Returns:
        model:     Fine-tuned model in eval mode.
        tokenizer: Matching tokenizer.

    How PeftModel works:
        The base model weights are loaded frozen.
        Then PeftModel.from_pretrained() loads the saved A and B matrices
        and injects them back into the attention layers.
        Result: identical to the model state at the end of training.
    """
    print(f"📥 Loading base model: {base_model_name}")
    print(f"🔌 Loading LoRA adapter from: {adapter_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model in 4-bit (same config as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Layer the LoRA adapter on top of the base model
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Set to evaluation mode:
    # - Disables dropout (we want deterministic outputs)
    # - Disables gradient computation (saves memory during inference)
    model.eval()

    print("✅ Fine-tuned model loaded and ready for evaluation.")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Text Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """
    Generates a response from the model for a given prompt.

    Used both for qualitative evaluation (single prompts) and
    batch ROUGE evaluation (many prompts).

    Args:
        model:          The model to generate from.
        tokenizer:      The tokenizer for encoding/decoding.
        prompt:         The input text (instruction portion only, no response).
        max_new_tokens: Maximum tokens to generate in the response.
        temperature:    Sampling temperature.
                        0.1 = nearly deterministic (good for evaluation)
                        1.0 = more random/creative
                        We use 0.1 for reproducible evaluation results.

    Returns:
        str: The generated response text (only the new tokens, not the prompt).

    Key concept — torch.no_grad():
        During inference we don't need gradients.
        torch.no_grad() disables gradient tracking, which:
          - Saves ~50% GPU memory
          - Speeds up inference
        Always use this when not training.
    """
    # Tokenize the input prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",    # Return PyTorch tensors
        truncation=True,
        max_length=512,
    ).to(model.device)          # Move to same device as model (GPU)

    input_length = inputs["input_ids"].shape[1]   # Number of input tokens

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,          # Use sampling (vs greedy decoding)
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode ONLY the newly generated tokens (not the input prompt)
    # outputs[0] = all tokens (input + generated)
    # [input_length:] = slice off the input portion
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge_scores(
    model,
    tokenizer,
    test_dataset,
    num_samples: int = 100,
) -> dict:
    """
    Computes ROUGE scores by generating responses for test examples
    and comparing them to reference answers.

    Args:
        model:       Model to evaluate (base or fine-tuned).
        tokenizer:   Matching tokenizer.
        test_dataset: HuggingFace dataset with "text" column.
        num_samples: How many test examples to evaluate.
                     100 gives reliable estimates without taking too long.

    Returns:
        dict: ROUGE-1, ROUGE-2, ROUGE-L scores (0 to 1, higher is better).

    How ROUGE works:
        ROUGE-1: Overlap of single words (unigrams)
        ROUGE-2: Overlap of word pairs (bigrams)
        ROUGE-L: Longest common subsequence

        Example:
          Reference: "Eat a balanced diet and exercise regularly"
          Generated: "Eat healthy food and exercise daily"
          ROUGE-1: 3/7 = 0.43  (eat, and, exercise match)
    """
    rouge = hf_evaluate.load("rouge")

    predictions = []
    references = []

    print(f"\n📊 Computing ROUGE on {num_samples} test samples...")
    print("   (This takes a few minutes — model generates text for each sample)")

    # Use a subset for speed
    samples = test_dataset.select(range(min(num_samples, len(test_dataset))))

    for example in tqdm(samples, desc="Generating"):
        full_text = example["text"]

        # Split the text at "### Response:" to get prompt (input) and reference (output)
        # The model should only see the prompt, not the reference answer
        if "### Response:" in full_text:
            parts = full_text.split("### Response:")
            prompt = parts[0] + "### Response:"
            reference = parts[1].strip()
        else:
            continue

        # Generate model's response to the prompt
        prediction = generate_response(model, tokenizer, prompt)

        predictions.append(prediction)
        references.append(reference)

    # Compute ROUGE scores
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,   # "running" and "run" treated as same word
    )

    print(f"\n📈 ROUGE Results:")
    print(f"   ROUGE-1: {results['rouge1']:.4f}")
    print(f"   ROUGE-2: {results['rouge2']:.4f}")
    print(f"   ROUGE-L: {results['rougeL']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Loss Curve Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(trainer, save_path: str = "results/training_loss_curve.png") -> None:
    """
    Plots the training loss curve from the trainer's log history
    and saves it to results/.

    Args:
        trainer:   The SFTTrainer object after training.
        save_path: Where to save the plot.

    Why plot the loss curve?
        It tells you if training went well:
          Healthy:     Loss steadily decreases, then plateaus
          Overfitting: Training loss drops but validation loss rises
          Unstable:    Loss spikes or oscillates wildly
          Too slow:    Loss barely moves (learning rate too low)
    """
    os.makedirs("results", exist_ok=True)

    # Extract loss values from trainer's log history
    logs = trainer.state.log_history

    steps, losses = [], []
    for log in logs:
        if "loss" in log and "step" in log:
            steps.append(log["step"])
            losses.append(log["loss"])

    if not steps:
        print("⚠️  No training logs found. Was training completed?")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(steps, losses, color="#4F46E5", linewidth=2, label="Training Loss")

    # Add a smoothed trendline
    if len(losses) > 10:
        window = max(5, len(losses) // 20)
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        smooth_steps = steps[window - 1:]
        ax.plot(smooth_steps, smoothed, color="#EC4899", linewidth=2,
                linestyle="--", label=f"Smoothed (window={window})", alpha=0.8)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curve — TinyLlama QLoRA Fine-Tuning", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate start and end loss
    ax.annotate(f"Start: {losses[0]:.3f}", xy=(steps[0], losses[0]),
                xytext=(steps[0] + max(steps) * 0.05, losses[0] + 0.1),
                fontsize=10, color="#4F46E5")
    ax.annotate(f"End: {losses[-1]:.3f}", xy=(steps[-1], losses[-1]),
                xytext=(steps[-1] - max(steps) * 0.15, losses[-1] + 0.1),
                fontsize=10, color="#4F46E5")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n📉 Loss curve saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_rouge_comparison(
    base_scores: dict,
    finetuned_scores: dict,
    save_path: str = "results/rouge_scores_comparison.png",
) -> None:
    """
    Creates a side-by-side bar chart comparing ROUGE scores
    between the base model and fine-tuned model.

    Args:
        base_scores:      ROUGE dict from base model evaluation.
        finetuned_scores: ROUGE dict from fine-tuned model evaluation.
        save_path:        Where to save the chart.
    """
    os.makedirs("results", exist_ok=True)

    metrics = ["rouge1", "rouge2", "rougeL"]
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

    base_vals = [base_scores[m] for m in metrics]
    ft_vals = [finetuned_scores[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width / 2, base_vals, width,
                   label="Base Model", color="#94A3B8", edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width / 2, ft_vals, width,
                   label="Fine-tuned (QLoRA)", color="#4F46E5", edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10,
                color="#4F46E5", fontweight="bold")

    # Add improvement % labels
    for i, (b, f) in enumerate(zip(base_vals, ft_vals)):
        if b > 0:
            pct = (f - b) / b * 100
            ax.text(x[i], max(b, f) + 0.04, f"+{pct:.1f}%",
                    ha="center", fontsize=10, color="#10B981", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("ROUGE Score", fontsize=12)
    ax.set_title("ROUGE Scores: Base Model vs QLoRA Fine-tuned\n(TinyLlama-1.1B on Alpaca)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(base_vals), max(ft_vals)) + 0.15)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n📊 ROUGE comparison chart saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Qualitative Comparison
# ─────────────────────────────────────────────────────────────────────────────

# Five diverse test prompts covering different instruction types
QUALITATIVE_PROMPTS = [
    {
        "label": "List task",
        "prompt": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nGive three tips for staying healthy.\n\n### Response:"
        ),
    },
    {
        "label": "Explanation task",
        "prompt": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nExplain what photosynthesis is in simple terms.\n\n### Response:"
        ),
    },
    {
        "label": "Creative task",
        "prompt": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nWrite a short poem about the night sky.\n\n### Response:"
        ),
    },
    {
        "label": "Reasoning task",
        "prompt": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nWhat are the advantages and disadvantages of remote work?\n\n### Response:"
        ),
    },
    {
        "label": "Factual task",
        "prompt": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nName five programming languages and what they are commonly used for.\n\n### Response:"
        ),
    },
]


def run_qualitative_evaluation(
    base_model,
    base_tokenizer,
    finetuned_model,
    finetuned_tokenizer,
    save_path: str = "results/qualitative_comparison.json",
) -> list:
    """
    Runs 5 fixed prompts through both models and compares outputs side by side.

    This is the most insightful evaluation — numbers tell you IF the model improved,
    but qualitative comparison tells you HOW it improved.

    Look for:
      - Does fine-tuned model better follow the instruction format?
      - Is the response more structured (numbered lists when asked)?
      - Is the length more appropriate?
      - Does it stop generating at the right place?

    Args:
        base_model:           The original TinyLlama (before fine-tuning).
        base_tokenizer:       Tokenizer for base model.
        finetuned_model:      The QLoRA fine-tuned model.
        finetuned_tokenizer:  Tokenizer for fine-tuned model.
        save_path:            Path to save comparison JSON.

    Returns:
        list of dicts with prompt, base_response, finetuned_response.
    """
    os.makedirs("results", exist_ok=True)
    results = []

    print("\n🔍 Running qualitative evaluation on 5 prompts...")
    print("=" * 70)

    for item in QUALITATIVE_PROMPTS:
        label = item["label"]
        prompt = item["prompt"]

        print(f"\n📌 [{label}]")
        print(f"Prompt: {prompt.split('### Instruction:')[1].split('### Response:')[0].strip()}")
        print("-" * 70)

        # Generate from base model
        base_response = generate_response(base_model, base_tokenizer, prompt)
        print(f"🔵 BASE MODEL:\n{base_response}\n")

        # Generate from fine-tuned model
        ft_response = generate_response(finetuned_model, finetuned_tokenizer, prompt)
        print(f"🟣 FINE-TUNED:\n{ft_response}\n")
        print("=" * 70)

        results.append({
            "label": label,
            "prompt": prompt,
            "base_response": base_response,
            "finetuned_response": ft_response,
        })

    # Save results
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Qualitative results saved to: {save_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save Final Evaluation Summary
# ─────────────────────────────────────────────────────────────────────────────

def save_evaluation_summary(
    base_rouge: dict,
    finetuned_rouge: dict,
    lora_rank: int = 8,
    save_path: str = "results/evaluation_results.json",
) -> None:
    """
    Saves a structured JSON summary of all evaluation results.
    This is what gets filled into the README results table.

    Args:
        base_rouge:      ROUGE scores for base model.
        finetuned_rouge: ROUGE scores for fine-tuned model.
        lora_rank:       The LoRA rank used during training.
        save_path:       Output path for the JSON file.
    """
    os.makedirs("results", exist_ok=True)

    summary = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dataset": "yahma/alpaca-cleaned",
        "lora_rank": lora_rank,
        "base_model": {
            "rouge1": round(base_rouge["rouge1"], 4),
            "rouge2": round(base_rouge["rouge2"], 4),
            "rougeL": round(base_rouge["rougeL"], 4),
        },
        "finetuned_model": {
            "rouge1": round(finetuned_rouge["rouge1"], 4),
            "rouge2": round(finetuned_rouge["rouge2"], 4),
            "rougeL": round(finetuned_rouge["rougeL"], 4),
        },
        "improvement": {
            "rouge1_pct": round(
                (finetuned_rouge["rouge1"] - base_rouge["rouge1"]) / max(base_rouge["rouge1"], 1e-8) * 100, 2
            ),
            "rouge2_pct": round(
                (finetuned_rouge["rouge2"] - base_rouge["rouge2"]) / max(base_rouge["rouge2"], 1e-8) * 100, 2
            ),
            "rougeL_pct": round(
                (finetuned_rouge["rougeL"] - base_rouge["rougeL"]) / max(base_rouge["rougeL"], 1e-8) * 100, 2
            ),
        },
    }

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Evaluation summary saved to: {save_path}")
    print(f"   → Update your README.md results table with these numbers!")
