"""
model_utils.py — Model loading, quantization, and LoRA configuration.

What this module does:
  1. Loads TinyLlama-1.1B in 4-bit (NF4) quantization  → the "Q" in QLoRA
  2. Loads and configures the tokenizer
  3. Configures LoRA adapter settings                   → the "LoRA" in QLoRA
  4. Applies LoRA adapters to the quantized model
  5. Reports trainable parameter counts (so you can see PEFT's efficiency)

Key libraries:
  - transformers: AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
  - peft: LoraConfig, get_peft_model
  - bitsandbytes: powers the 4-bit quantization under the hood
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


# ─────────────────────────────────────────────────────────────────────────────
# Quantization Configuration (the "Q" in QLoRA)
# ─────────────────────────────────────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    """
    Creates the BitsAndBytes configuration for 4-bit quantization.

    This compresses TinyLlama from ~2.2 GB (float16) down to ~0.6 GB (4-bit),
    making it possible to run on a free T4 GPU.

    Returns:
        BitsAndBytesConfig: Configuration object passed to model loading.

    Parameter explanation:
        load_in_4bit:
            Load and store model weights as 4-bit integers.
            Normally weights are 32-bit floats (float32) or 16-bit (float16).
            4-bit = 75% memory reduction vs float16.

        bnb_4bit_compute_dtype:
            Even though weights are STORED in 4-bit, actual COMPUTATION
            (matrix multiplications) happens in float16 for numerical stability.
            Think of it as: compressed storage, uncompressed when used.

        bnb_4bit_quant_type ("nf4"):
            NormalFloat4 — a special 4-bit format designed for LLM weights.
            Neural network weights follow a normal (bell curve) distribution.
            NF4 is optimized for this distribution → more accurate than int4.

        bnb_4bit_use_double_quant:
            Quantizes the quantization constants themselves (meta-compression).
            Saves an additional ~0.4 GB. Barely affects quality.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Loads and configures the tokenizer for TinyLlama.

    A tokenizer converts text ↔ token IDs (integers).
    The model only understands integers, not raw text.

    Args:
        model_name: HuggingFace model ID (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    Returns:
        AutoTokenizer: Configured tokenizer ready for use.

    Key settings:
        padding_side="right":
            When batching sequences of different lengths, we pad shorter ones
            to match the longest. Padding goes on the RIGHT for causal LMs.
            (Padding on the left would confuse the model's position awareness.)

        pad_token = eos_token:
            TinyLlama has no dedicated padding token.
            We reuse the end-of-sequence token for padding.
            This is the standard workaround for LLaMA-family models.
    """
    print(f"🔤 Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # LLaMA-family models don't have a pad token by default
    # We set it to eos_token (end-of-sequence) — standard practice
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # Required for causal LM training

    print(f"   Vocabulary size: {tokenizer.vocab_size:,} tokens")
    print(f"   EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    print(f"   PAD token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")

    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Base Model Loading (with 4-bit Quantization)
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(model_name: str) -> AutoModelForCausalLM:
    """
    Loads TinyLlama-1.1B from HuggingFace Hub with 4-bit quantization.

    This downloads ~600 MB (4-bit compressed) instead of ~2.2 GB (float16).
    The model is loaded directly onto the GPU in quantized form.

    Args:
        model_name: HuggingFace model ID

    Returns:
        AutoModelForCausalLM: The base model, quantized, on GPU.

    Key settings:
        quantization_config:
            Applies the BitsAndBytes 4-bit config we defined above.

        device_map="auto":
            Automatically places model layers on available devices.
            With one GPU: puts everything on GPU.
            With CPU only: puts on CPU (very slow but possible).

        trust_remote_code=True:
            Some models include custom code. We trust TinyLlama's code.
    """
    print(f"\n🤖 Loading base model: {model_name}")
    print("   (First run downloads ~600 MB — cached for future runs)")

    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",           # Automatically use GPU if available
        torch_dtype=torch.float16,   # Prevent .to() errors and force master dtype
        trust_remote_code=True,
    )

    # Disable model's cache during training (saves memory, not needed for training)
    model.config.use_cache = False

    # Needed for gradient checkpointing compatibility
    model.config.pretraining_tp = 1

    print(f"   Model loaded on: {next(model.parameters()).device}")
    print(f"   Model dtype: {next(model.parameters()).dtype}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Configuration (the "LoRA" in QLoRA)
# ─────────────────────────────────────────────────────────────────────────────

def get_lora_config(r: int = 8) -> LoraConfig:
    """
    Creates the LoRA adapter configuration.

    LoRA works by injecting small trainable matrices (A and B) into
    specific layers of the frozen base model. Instead of updating the
    full weight matrix W, we learn ΔW ≈ A × B where A and B are small.

    Args:
        r: LoRA rank. Controls the size of adapter matrices.
           - r=4:  Fewest params, fastest, least expressive
           - r=8:  Default — good balance (used in ablation as middle)
           - r=16: More expressive, slightly more params, slower

    Returns:
        LoraConfig: Configuration for PEFT adapter injection.

    Parameter explanation:
        r (rank):
            The inner dimension of the low-rank matrices.
            For a 2048×2048 weight matrix:
              Full matrix: 4,194,304 parameters
              LoRA (r=8):  2×(2048×8) = 32,768 parameters  → 128× fewer

        lora_alpha:
            Scaling factor for the LoRA update: (alpha/r) × A×B
            Convention: alpha = 2 × r (so alpha=16 for r=8)
            This keeps the effective learning rate stable across ranks.

        target_modules:
            Which layers to inject LoRA into.
            We target the 4 attention projection matrices:
              q_proj: Query  — "What am I looking for?"
              k_proj: Key    — "What do I contain?"
              v_proj: Value  — "What do I pass along?"
              o_proj: Output — Final output projection
            These are where most of the model's "reasoning" happens.

        lora_dropout:
            Randomly zeros out adapter activations during training.
            Prevents the small adapter from overfitting.
            0.05 = 5% dropout — light regularization.

        bias="none":
            Don't add trainable bias terms. Keeps adapter size minimal.

        task_type=CAUSAL_LM:
            Tells PEFT we're fine-tuning a causal language model
            (predicts next token left-to-right).
    """
    return LoraConfig(
        r=r,
        lora_alpha=r * 2,          # Convention: alpha = 2 × r
        target_modules=[
            "q_proj",              # Query projection
            "k_proj",              # Key projection
            "v_proj",              # Value projection
            "o_proj",              # Output projection
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Apply LoRA to Model
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model: AutoModelForCausalLM, lora_config: LoraConfig):
    """
    Prepares the quantized model for training and injects LoRA adapters.

    Two steps happen here:
      1. prepare_model_for_kbit_training():
           Sets up the quantized model for gradient computation.
           Quantized models need special handling because 4-bit weights
           can't directly receive gradients — this bridges that gap.

      2. get_peft_model():
           Injects the LoRA adapter matrices (A and B) into the target
           layers and freezes all base model parameters.
           After this, ONLY the LoRA adapters are trainable.

    Args:
        model:       The quantized base model
        lora_config: The LoRA configuration

    Returns:
        model: The model with LoRA adapters injected, ready to train.
    """
    print("\n🔧 Preparing model for quantized training...")
    model = prepare_model_for_kbit_training(model)

    print("💉 Injecting LoRA adapters...")
    model = get_peft_model(model, lora_config)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Trainable Parameter Counter
# ─────────────────────────────────────────────────────────────────────────────

def print_trainable_parameters(model) -> None:
    """
    Prints the number of trainable vs total parameters.

    This is the most satisfying output when learning PEFT —
    seeing that you're training only ~1% of all parameters.

    Example output:
        trainable params: 4,194,304
        all params:       1,100,048,384
        trainable %:      0.38%
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / all_params

    print("\n📊 Parameter Summary:")
    print(f"   Trainable params : {trainable_params:>15,}")
    print(f"   Frozen params    : {all_params - trainable_params:>15,}")
    print(f"   Total params     : {all_params:>15,}")
    print(f"   Trainable %      : {trainable_pct:.4f}%")
    print(f"\n   💡 We're training only {trainable_pct:.2f}% of parameters — that's PEFT!")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: Load everything at once
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, lora_r: int = 8):
    """
    Convenience function: loads tokenizer + model + applies LoRA in one call.

    This is what the training notebook will call.

    Args:
        model_name: HuggingFace model ID
        lora_r:     LoRA rank (default 8, change for ablation study)

    Returns:
        model:     PEFT model with LoRA adapters, ready to train
        tokenizer: Configured tokenizer
    """
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name)
    lora_config = get_lora_config(r=lora_r)
    model = apply_lora(model, lora_config)
    print_trainable_parameters(model)

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run directly to verify imports work
# Usage: python src/model_utils.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Only test the config objects (no GPU needed for this)
    print("Testing BitsAndBytes config:")
    bnb = get_bnb_config()
    print(f"  load_in_4bit: {bnb.load_in_4bit}")
    print(f"  quant_type: {bnb.bnb_4bit_quant_type}")

    print("\nTesting LoRA config (r=8):")
    lora = get_lora_config(r=8)
    print(f"  r: {lora.r}")
    print(f"  alpha: {lora.lora_alpha}")
    print(f"  target_modules: {lora.target_modules}")

    print("\nTesting LoRA config (r=4):")
    lora_small = get_lora_config(r=4)
    print(f"  r: {lora_small.r}")
    print(f"  alpha: {lora_small.lora_alpha}")

    print("\n✅ Config objects created successfully (GPU not tested here)")
