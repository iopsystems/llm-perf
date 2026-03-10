# Dataset Preparation Scripts

## prepare_dataset.py

Downloads and processes the OpenOrca dataset from HuggingFace to create benchmark-ready JSONL files.

### Requirements

```bash
pip install pandas pyarrow requests
```

### Usage

**Generate default dataset (10,000 samples):**
```bash
python scripts/prepare_dataset.py
```

**Customize sample count:**
```bash
python scripts/prepare_dataset.py --samples 1000
```

**Custom output directory:**
```bash
python scripts/prepare_dataset.py --output-dir my_datasets
```

### What it does

1. Downloads the OpenOrca 1M-GPT4-Augmented parquet file from HuggingFace (~500MB)
2. Samples N random prompts from the dataset
3. Estimates appropriate `max_tokens` based on reference response length
4. Generates `openorca-{N}.jsonl` file (e.g., `openorca-10000.jsonl` for 10,000 samples)

Use llm-perf's `sample_size` config parameter for quick testing with fewer prompts.

### Output format

Each line in the JSONL file contains:
```json
{
  "prompt": "What is the capital of France?",
  "max_tokens": 150
}
```

The benchmark tool automatically adds unique IDs (`[req-N]`) to each request to prevent response caching, so dataset prompts are kept clean and unmodified.

### Fallback behavior

If the download fails (no internet, HuggingFace issues), the script automatically falls back to generating synthetic examples based on common instruction-following patterns.
