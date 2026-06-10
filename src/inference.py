"""
inference.py — Load the fine-tuned model and generate responses.

What this module does:
  1. Loads the base model + LoRA adapter (the fine-tuned model)
  2. Provides a clean interface to generate responses to any instruction
  3. Supports both single prompts and batch inference
  4. Can be used interactively or imported into notebooks

This is the file you use AFTER training to actually USE your model.

Usage in a Colab notebook:
    from src.inference import FineTunedModel
    model = FineTunedModel("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "outputs/qlora-adapter-r8")
    response = model.ask("What are 3 tips for better sleep?")
    print(response)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Template (must match what was used during training)
# ─────────────────────────────────────────────────────────────────────────────

# IMPORTANT: The inference prompt format must EXACTLY match the training format.
# If training used "### Instruction:" headers, inference must use them too.
# The model learned to respond to this exact pattern — changing it breaks things.

INFERENCE_PROMPT = """\
Below is an instruction that describes a task. \
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

INFERENCE_PROMPT_WITH_INPUT = """\
Below is an instruction that describes a task, paired with an input that \
provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""


# ─────────────────────────────────────────────────────────────────────────────
# FineTunedModel — Main Inference Class
# ─────────────────────────────────────────────────────────────────────────────

class FineTunedModel:
    """
    A clean, reusable wrapper around the fine-tuned TinyLlama model.

    Instead of repeating model loading code in every notebook,
    we wrap everything in a class with a simple .ask() interface.

    Example usage:
        model = FineTunedModel(
            base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            adapter_path="outputs/qlora-adapter-r8"
        )
        print(model.ask("Explain gravity to a 5 year old."))
    """

    def __init__(self, base_model_name: str, adapter_path: str):
        """
        Loads base model + LoRA adapter and prepares for inference.

        Args:
            base_model_name: HuggingFace model ID for TinyLlama.
            adapter_path:    Path to the saved LoRA adapter directory.
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path

        print("🚀 Loading fine-tuned model for inference...")
        self.model, self.tokenizer = self._load()
        print("✅ Model ready! Use .ask(instruction) to generate responses.\n")

    def _load(self):
        """Internal: loads tokenizer, base model, and LoRA adapter."""

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 4-bit quantization config (same as training)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Base model
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Apply LoRA adapter on top of base model
        model = PeftModel.from_pretrained(base, self.adapter_path)
        model.eval()   # Disable dropout, no gradient tracking needed

        return model, tokenizer

    def ask(
        self,
        instruction: str,
        context: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        The main interface — ask the model to follow an instruction.

        Args:
            instruction:    The task or question for the model.
            context:        Optional extra context (maps to "### Input:" field).
                            Leave empty for simple instruction-only prompts.
            max_new_tokens: Maximum tokens to generate.
                            256 is good for most Alpaca-style responses.
                            Increase for longer outputs.
            temperature:    Randomness of generation.
                            0.1 = deterministic/conservative (good for facts)
                            0.7 = balanced (good for general use)
                            1.0 = creative/varied
            do_sample:      If True, uses sampling (respects temperature).
                            If False, uses greedy decoding (always picks most likely token).

        Returns:
            str: The model's response text.

        Example:
            >>> model.ask("List 3 benefits of exercise.")
            "1. Improves cardiovascular health\n2. Boosts mood and mental health\n3. Increases strength..."
        """
        # Build the formatted prompt
        if context.strip():
            prompt = INFERENCE_PROMPT_WITH_INPUT.format(
                instruction=instruction,
                input=context,
            )
        else:
            prompt = INFERENCE_PROMPT.format(instruction=instruction)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,   # Slightly penalize repeating the same phrases
            )

        # Decode only the new tokens (strip off the input prompt)
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def compare_with_base(
        self,
        instruction: str,
        base_model_name: str = None,
        temperature: float = 0.1,
    ) -> dict:
        """
        Loads the BASE model (no adapter) and runs the same prompt through both.
        Returns a dict with both responses for direct comparison.

        This is the most powerful demo — shows exactly what fine-tuning changed.

        Args:
            instruction:     The instruction to test.
            base_model_name: Override base model name (defaults to self.base_model_name).
            temperature:     Low temperature for consistent comparison.

        Returns:
            dict with keys "instruction", "base_response", "finetuned_response"

        Note: Loading two models at once requires more VRAM.
              On a T4 (15GB), load base model, generate, then unload before loading fine-tuned.
              This method handles that automatically.
        """
        base_name = base_model_name or self.base_model_name
        prompt = INFERENCE_PROMPT.format(instruction=instruction)

        print(f"📌 Instruction: {instruction}\n")

        # Get fine-tuned response (already loaded)
        print("🟣 Generating fine-tuned response...")
        ft_response = self.ask(instruction, temperature=temperature)
        print(f"Fine-tuned: {ft_response}\n")

        # Load base model temporarily
        print("🔵 Loading base model for comparison...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        base_model.eval()

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(base_model.device)

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        base_response = self.tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()

        print(f"Base model: {base_response}\n")

        # Free base model from GPU memory
        del base_model
        torch.cuda.empty_cache()

        return {
            "instruction": instruction,
            "base_response": base_response,
            "finetuned_response": ft_response,
        }

    def __repr__(self):
        return (
            f"FineTunedModel(\n"
            f"  base='{self.base_model_name}',\n"
            f"  adapter='{self.adapter_path}',\n"
            f"  device='{next(self.model.parameters()).device}'\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helper — useful in notebooks without instantiating the class
# ─────────────────────────────────────────────────────────────────────────────

def format_instruction_prompt(instruction: str, context: str = "") -> str:
    """
    Formats an instruction into the Alpaca prompt template.

    Useful when you want to build the prompt manually.

    Args:
        instruction: The task description.
        context:     Optional additional context.

    Returns:
        str: Formatted prompt string ready for tokenization.
    """
    if context.strip():
        return INFERENCE_PROMPT_WITH_INPUT.format(
            instruction=instruction, input=context
        )
    return INFERENCE_PROMPT.format(instruction=instruction)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive mode — run from command line for quick testing
# Usage: python src/inference.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ADAPTER_PATH = "outputs/qlora-adapter-r8"

    if not torch.cuda.is_available():
        print("⚠️  No GPU detected. Inference will be very slow on CPU.")

    print(f"Loading model from adapter: {ADAPTER_PATH}")
    model = FineTunedModel(BASE_MODEL, ADAPTER_PATH)

    print("\n" + "="*60)
    print("  Interactive Inference — Type 'quit' to exit")
    print("="*60)

    while True:
        instruction = input("\n📝 Enter instruction: ").strip()
        if instruction.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not instruction:
            continue

        response = model.ask(instruction)
        print(f"\n🤖 Response:\n{response}")
