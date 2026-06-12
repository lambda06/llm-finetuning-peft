"""
train.py — Fine-tuning pipeline for TinyLlama with QLoRA.

What this module does:
  1. Reads hyperparameters from configs/qlora_config.yaml
  2. Loads dataset (via data_utils) and model (via model_utils)
  3. Configures the SFTTrainer (Supervised Fine-Tuning Trainer)
  4. Runs the training loop
  5. Saves the LoRA adapter weights

Why SFTTrainer (from TRL) instead of a raw PyTorch training loop?
  SFTTrainer wraps the standard PyTorch loop and handles:
    - Gradient accumulation
    - Mixed precision (fp16)
    - Logging
    - Checkpointing
    - Data collation
  This lets us focus on understanding concepts, not boilerplate.
  In production, most teams use trainers exactly like this.

Usage (in a Colab notebook cell):
    from src.train import run_training
    run_training()

Or from command line (on a GPU machine):
    python src/train.py
"""

import os
import yaml
import torch
from trl import SFTTrainer, SFTConfig


from data_utils import load_alpaca_dataset
from model_utils import load_model_and_tokenizer, print_trainable_parameters


# ─────────────────────────────────────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/qlora_config.yaml") -> dict:
    """
    Loads hyperparameters from the YAML config file.

    Why read from a file instead of hardcoding values?
      - Change a hyperparameter without touching code
      - Version control your experiments (different YAML = different run)
      - Standard practice in production ML systems

    Args:
        config_path: Path to the YAML config file.

    Returns:
        dict: Nested dictionary of all configuration values.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"✅ Config loaded from: {config_path}")
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Training Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_training_args(config: dict) -> SFTConfig:
    """
    Creates HuggingFace TrainingArguments from our config.

    TrainingArguments is a dataclass that controls everything about
    how training runs — batch sizes, learning rate, logging, saving.

    Args:
        config: Full config dict loaded from qlora_config.yaml

    Returns:
        TrainingArguments: Passed directly to SFTTrainer.

    Key parameter explanations:

        output_dir:
            Where checkpoints are saved during training.
            Note: We DON'T track this in git (.gitignore excludes outputs/).
            At the end we save ONLY the LoRA adapter (tiny, ~50MB).

        num_train_epochs:
            How many times the model sees the full training set.
            3 epochs = each example is seen 3 times.
            More epochs → better fitting but risk of overfitting.

        per_device_train_batch_size:
            How many examples in one forward pass.
            4 means the model processes 4 examples simultaneously.
            Limited by GPU memory.

        gradient_accumulation_steps:
            Instead of updating weights after every batch, accumulate
            gradients across N batches then update once.
            Effective batch size = batch_size × gradient_accumulation_steps
                                 = 4 × 4 = 16
            This simulates a larger batch without extra memory cost.

        learning_rate:
            How big each parameter update step is.
            Too high → training unstable (loss spikes, oscillates)
            Too low  → training very slow
            2e-4 (0.0002) is the standard QLoRA recipe.

        lr_scheduler_type="cosine":
            Learning rate follows a cosine curve:
              Start: warmup to peak
              Middle: gradually decrease following cosine curve
              End: near zero
            This helps the model make bold updates early and
            fine adjustments later.

        warmup_ratio=0.03:
            For the first 3% of training steps, linearly ramp up
            the learning rate from 0 to the peak value.
            Prevents large, destabilizing updates at the start.

        fp16=True:
            Use 16-bit floating point for computations.
            This halves memory usage vs 32-bit and speeds up training
            on GPUs with Tensor Cores (T4, A100, etc.)

        logging_steps=10:
            Print the training loss every 10 steps.
            You'll see loss values decreasing — this is your signal
            that the model is actually learning.

        save_steps=100:
            Save a checkpoint every 100 steps.
            Protects against Colab disconnecting mid-training.

        save_total_limit=2:
            Keep only the 2 most recent checkpoints.
            Prevents filling up disk with old checkpoints.

        report_to="none":
            Don't report to WandB or other experiment trackers.
            Keeps setup simple. (You can add WandB later.)
    """
    t = config["training"]

    return SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        fp16=t["fp16"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        optim="paged_adamw_32bit",   # Memory-efficient Adam optimizer (QLoRA default)
        report_to="none",            # Disable WandB / other loggers
        push_to_hub=False,           # We'll push manually after evaluating
        dataset_text_field="text",   # New requirement in latest trl
        max_seq_length=config["model"]["max_seq_length"], # New requirement in latest trl
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    config_path: str = "configs/qlora_config.yaml",
    lora_r: int = None,
    output_adapter_path: str = "outputs/qlora-adapter",
):
    """
    Full training pipeline: load data → load model → train → save adapter.

    Args:
        config_path:          Path to the YAML config file.
        lora_r:               LoRA rank override (used by ablation study).
                              If None, uses the value from config.
        output_adapter_path:  Where to save the final LoRA adapter weights.

    Returns:
        trainer: The trained SFTTrainer object (useful for extracting metrics).
    """
    # ── Step 1: Load config ───────────────────────────────────────────────────
    config = load_config(config_path)

    model_name = config["model"]["name"]
    max_seq_length = config["model"]["max_seq_length"]

    # Allow overriding LoRA rank (for ablation study)
    rank = lora_r if lora_r is not None else config["lora"]["r"]

    print("\n" + "="*60)
    print(f"  QLoRA Fine-Tuning: {model_name}")
    print(f"  LoRA rank: r={rank}")
    print("="*60 + "\n")

    # ── Step 2: Load dataset ──────────────────────────────────────────────────
    train_dataset, test_dataset = load_alpaca_dataset(
        train_size=config["dataset"]["train_size"],
        test_size=config["dataset"]["test_size"],
        seed=config["dataset"]["seed"],
    )

    # ── Step 3: Load model + tokenizer + apply LoRA ───────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_name, lora_r=rank)

    # ── Step 4: Configure training arguments ──────────────────────────────────
    training_args = get_training_args(config)

    # Override output dir to include rank (useful for ablation study runs)
    training_args.output_dir = f"outputs/qlora-r{rank}"

    # ── Step 5: Create the SFTTrainer ─────────────────────────────────────────
    #
    # SFTTrainer (Supervised Fine-Tuning Trainer) from TRL handles:
    #   - The full training loop (forward pass, loss, backward, optimizer step)
    #   - Tokenizing the "text" column automatically
    #   - Gradient accumulation
    #   - Mixed precision training
    #   - Logging and checkpointing
    #
    # dataset_text_field="text":
    #   Tells the trainer which column of the dataset contains the prompt strings.
    #   This is the "text" key we created in data_utils.format_prompt().
    #
    # max_seq_length=512:
    #   Truncate any prompt longer than 512 tokens.
    #   Prevents memory issues from very long examples.
    #   512 is sufficient for Alpaca examples (most are < 300 tokens).

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    #
    # This is where the actual learning happens.
    # Under the hood, for each batch:
    #   1. Forward pass: model predicts next tokens
    #   2. Loss: cross-entropy between predictions and actual tokens
    #   3. Backward pass: compute gradients (only for LoRA params)
    #   4. Optimizer step: update LoRA A and B matrices
    #   5. Zero gradients: clear for next batch
    #
    # You'll see output like:
    #   {'loss': 2.1834, 'learning_rate': 0.0002, 'epoch': 0.16}
    #   {'loss': 1.8921, 'learning_rate': 0.00019, 'epoch': 0.32}
    #   ...
    # Loss should trend downward. That's the model learning.

    print("\n🚀 Starting training...")
    print(f"   Effective batch size: "
          f"{config['training']['per_device_train_batch_size']} × "
          f"{config['training']['gradient_accumulation_steps']} = "
          f"{config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"   Total training steps: ~{len(train_dataset) * config['training']['num_train_epochs'] // (config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'])}")

    # HOTFIX: Force trainable params to float32 to prevent BFloat16 mixed-precision crashes
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    train_result = trainer.train()

    print("\n✅ Training complete!")
    print(f"   Final training loss: {train_result.training_loss:.4f}")
    print(f"   Total steps: {train_result.global_step}")
    print(f"   Training time: {train_result.metrics['train_runtime']:.1f} seconds")

    # ── Step 7: Save the LoRA adapter ─────────────────────────────────────────
    #
    # IMPORTANT: We save ONLY the LoRA adapter — NOT the full model.
    # The adapter is ~50 MB. The full model would be ~600 MB.
    # To load later: base_model + adapter = full fine-tuned model.
    #
    # This is a key PEFT advantage: tiny, shareable, version-controllable.

    save_path = f"{output_adapter_path}-r{rank}"
    print(f"\n💾 Saving LoRA adapter to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"   Adapter saved! Size: check {save_path}/ directory")

    # Save training metrics alongside the adapter
    metrics_path = f"{save_path}/training_metrics.yaml"
    import yaml
    with open(metrics_path, "w") as f:
        yaml.dump({
            "lora_rank": rank,
            "final_loss": train_result.training_loss,
            "total_steps": train_result.global_step,
            "training_time_seconds": train_result.metrics["train_runtime"],
            "train_samples": len(train_dataset),
        }, f)

    print(f"\n📊 Training metrics saved to: {metrics_path}")

    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — run from command line
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Check GPU availability before starting
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected!")
        print("   This script requires a CUDA GPU.")
        print("   Please run on Google Colab (Runtime → Change runtime type → T4 GPU)")
        print("   or Kaggle (Settings → Accelerator → GPU T4 x2)")
        exit(1)

    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    run_training()
