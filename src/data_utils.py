"""
data_utils.py — Dataset loading and prompt formatting for Alpaca fine-tuning.

What this module does:
  1. Loads the Stanford Alpaca dataset from HuggingFace Hub
  2. Formats each example into a single prompt string the model can learn from
  3. Creates reproducible train/test splits

Why a separate module?
  Instead of writing this logic in every notebook, we write it once here
  and import it. This is standard software engineering practice.
"""

from datasets import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

# The Alpaca dataset has two types of examples:
#   1. Instruction only (no extra input context)
#   2. Instruction + input (e.g., "Summarize this: <article>")
# We need a template for each case.

PROMPT_WITH_INPUT = """\
Below is an instruction that describes a task, paired with an input that \
provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

PROMPT_WITHOUT_INPUT = """\
Below is an instruction that describes a task. \
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def format_prompt(example: dict) -> dict:
    """
    Converts a raw Alpaca dataset example into a formatted prompt string.

    The model learns to complete text, so we pack instruction + response
    into one string. During training, loss is computed over the FULL string
    (including the response), so the model learns to generate the response.

    Args:
        example: A dict with keys "instruction", "input", "output"

    Returns:
        A dict with a new key "text" containing the formatted prompt string.

    Example:
        Input:  {"instruction": "List 3 colors", "input": "", "output": "Red, Green, Blue"}
        Output: {"text": "Below is an instruction...\n### Instruction:\nList 3 colors\n\n### Response:\nRed, Green, Blue"}
    """
    # Use the appropriate template based on whether "input" is empty
    if example["input"].strip():
        # Example has extra context (e.g., a passage to summarize)
        text = PROMPT_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    else:
        # Example is instruction-only
        text = PROMPT_WITHOUT_INPUT.format(
            instruction=example["instruction"],
            output=example["output"],
        )

    return {"text": text}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_alpaca_dataset(
    train_size: int = 2000,
    test_size: int = 200,
    seed: int = 42,
) -> tuple:
    """
    Loads the cleaned Stanford Alpaca dataset from HuggingFace Hub,
    applies prompt formatting, and returns train/test splits.

    Args:
        train_size: Number of examples to use for training.
                    (Full dataset has 52,002 examples — we use a subset
                    to keep training fast on a free T4 GPU.)
        test_size:  Number of examples held out for evaluation.
                    These are NEVER seen during training.
        seed:       Random seed for reproducibility.
                    Same seed = same train/test split every time.

    Returns:
        train_dataset: HuggingFace Dataset object for training
        test_dataset:  HuggingFace Dataset object for evaluation

    Note on 'yahma/alpaca-cleaned':
        The original Stanford Alpaca dataset had some noisy/incorrect examples.
        This cleaned version (by yahma) removed ~1,700 bad examples.
        Always prefer cleaned datasets.
    """
    print("📥 Loading Alpaca dataset from HuggingFace Hub...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    print(f"   Full dataset size: {len(dataset):,} examples")

    # Shuffle before splitting so we get a random, representative subset.
    # Setting seed ensures the same shuffle every time → reproducible experiments.
    dataset = dataset.shuffle(seed=seed)

    # Take our subset
    total_needed = train_size + test_size
    dataset = dataset.select(range(total_needed))

    # Apply the prompt template to every example.
    # map() applies format_prompt() to every row efficiently (in parallel).
    # remove_columns drops the original columns — we only keep "text".
    print("📝 Formatting prompts...")
    dataset = dataset.map(
        format_prompt,
        remove_columns=["instruction", "input", "output"],
    )

    # Split into train and test
    split = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
    )

    train_dataset = split["train"]
    test_dataset = split["test"]

    print(f"✅ Train: {len(train_dataset):,} examples")
    print(f"✅ Test:  {len(test_dataset):,} examples")
    print(f"\n📌 Sample prompt (first training example):\n")
    print("-" * 60)
    print(train_dataset[0]["text"])
    print("-" * 60)

    return train_dataset, test_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run this file directly to verify everything works
# Usage: python src/data_utils.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, test_ds = load_alpaca_dataset(train_size=10, test_size=3)
    print(f"\nDataset features: {train_ds.features}")
    print(f"First example text length: {len(train_ds[0]['text'])} characters")
