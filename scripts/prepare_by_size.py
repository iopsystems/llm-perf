#!/usr/bin/env python3
"""
Prepare prompt files by t-shirt size from OpenOrca dataset.

This script organizes prompts into size classes matching the internal metrics:
- small: 0-200 tokens
- medium: 201-500 tokens
- large: 501-2000 tokens
- xlarge: 2001-8000 tokens
- xxlarge: 8000+ tokens
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import tiktoken


# Size class definitions (matching metrics.rs)
SIZE_CLASSES = {
    'small': (0, 200),
    'medium': (201, 500),
    'large': (501, 2000),
    'xlarge': (2001, 8000),
    'xxlarge': (8001, float('inf'))
}


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def get_size_class(token_count: int) -> str:
    """Determine size class for a given token count."""
    for size_name, (min_tokens, max_tokens) in SIZE_CLASSES.items():
        if min_tokens <= token_count <= max_tokens:
            return size_name
    return None


def load_openorca(max_samples: int = None) -> list:
    """Load OpenOrca dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not found. Install with: pip3 install datasets")
        exit(1)

    print("Loading OpenOrca dataset from HuggingFace...")
    dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)

    samples = []
    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # OpenOrca format: system_prompt + question + response
        system_prompt = example.get('system_prompt', '')
        question = example.get('question', '')
        response = example.get('response', '')

        # Combine system prompt and question as the input
        full_prompt = f"{system_prompt}\n\n{question}".strip()

        if full_prompt:
            samples.append({
                'prompt': full_prompt,
                'response': response
            })

        if (i + 1) % 10000 == 0:
            print(f"Loaded {i + 1} samples...")

    print(f"Loaded {len(samples)} total samples")
    return samples


def organize_by_size(samples: list, samples_per_size: int = None) -> dict:
    """Organize samples into size classes."""
    print("\nCounting tokens and organizing by size...")

    size_buckets = defaultdict(list)
    stats = defaultdict(int)
    all_buckets_full = False

    for i, sample in enumerate(samples):
        prompt = sample['prompt']
        response = sample['response']

        # Count input tokens to determine size class
        input_token_count = count_tokens(prompt)
        size_class = get_size_class(input_token_count)

        if size_class:
            # Check if we've hit the limit for this size class
            if samples_per_size is None or len(size_buckets[size_class]) < samples_per_size:
                # Count response tokens to set max_tokens (matching prepare_dataset.py logic)
                response_tokens = count_tokens(response)
                max_tokens = min(int(response_tokens * 1.5), 2500)
                max_tokens = max(max_tokens, 50)

                size_buckets[size_class].append({
                    'prompt': prompt,
                    'input_token_count': input_token_count,
                    'max_tokens': max_tokens
                })
            stats[size_class] += 1

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} samples...")

        # Check if all buckets are full (early termination)
        if samples_per_size is not None:
            all_full = all(
                len(size_buckets[size_name]) >= samples_per_size
                for size_name in SIZE_CLASSES.keys()
            )
            if all_full and not all_buckets_full:
                all_buckets_full = True
                print(f"All size classes filled after {i + 1} samples, stopping early...")
                break

    # Print statistics
    print("\nToken distribution across size classes:")
    for size_name in ['small', 'medium', 'large', 'xlarge', 'xxlarge']:
        min_tok, max_tok = SIZE_CLASSES[size_name]
        max_str = f"{int(max_tok)}" if max_tok != float('inf') else "inf"
        available = stats[size_name]
        selected = len(size_buckets[size_name])
        print(f"  {size_name:8} ({min_tok:5}-{max_str:>5} tokens): {selected:6} selected / {available:6} available")

    return dict(size_buckets)


def write_prompt_files(size_buckets: dict, output_dir: Path, shuffle: bool = True):
    """Write separate JSONL files for each size class."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting prompt files to {output_dir}/...")

    for size_name, samples in size_buckets.items():
        if not samples:
            print(f"  Skipping {size_name} (no samples)")
            continue

        # Optionally shuffle
        if shuffle:
            import random
            random.shuffle(samples)

        output_file = output_dir / f"openorca-{size_name}.jsonl"

        with open(output_file, 'w') as f:
            for sample in samples:
                # Write in llm-perf format (simple prompt + max_tokens)
                # max_tokens already calculated based on response length (1.5x response tokens)
                entry = {
                    "prompt": sample['prompt'],
                    "max_tokens": sample['max_tokens']
                }
                f.write(json.dumps(entry) + '\n')

        print(f"  {size_name:8}: {len(samples):6} prompts -> {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare size-specific prompt files from OpenOrca dataset"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('examples/prompts'),
        help='Output directory for prompt files (default: examples/prompts)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to load from dataset'
    )
    parser.add_argument(
        '--samples-per-size',
        type=int,
        help='Maximum number of samples per size class'
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle samples before writing'
    )

    args = parser.parse_args()

    # Load dataset
    samples = load_openorca(max_samples=args.max_samples)

    # Organize by size
    size_buckets = organize_by_size(samples, samples_per_size=args.samples_per_size)

    # Write files
    write_prompt_files(size_buckets, args.output_dir, shuffle=not args.no_shuffle)

    print("\nDone!")


if __name__ == '__main__':
    main()
