"""Token-length statistics, to choose a defensible --max-length.

Replaces the old main2.py, which ran its work at import time with a hardcoded
path, no main guard, and no way to point it at a different file. Picking
max_length matters: too low truncates real content, too high wastes compute
quadratically in the attention.
"""

import argparse

import numpy as np

from data.preprocess import build_local_tokenizer, clean_dataset, get_tokenizer
from data.schema import read_dataset
from training.config import TEXT_COLUMN


def token_length_stats(texts, model_name="bert-base-uncased", tokenizer=None,
                       max_length=512, percentiles=(50, 75, 90, 95, 99)):
    """Distribution of tokenized lengths, plus the truncation cost per cutoff.

    Returns:
        dict: count, mean/min/max, the requested percentiles, and the share of
        posts that would be truncated at a range of candidate max_lengths.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model_name)

    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return {"count": 0}

    lengths = np.array([
        len(tokenizer.encode(text, truncation=True, max_length=max_length,
                             add_special_tokens=True))
        for text in texts
    ])

    stats = {
        "count": int(len(lengths)),
        "mean": float(lengths.mean()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
    }
    for p in percentiles:
        stats[f"p{p}"] = float(np.percentile(lengths, p))

    stats["truncation_rate"] = {
        int(cutoff): round(float((lengths > cutoff).mean()), 4)
        for cutoff in (64, 128, 256, 512)
    }
    return stats


def run(dataset_path, model_name="bert-base-uncased", offline=False, column=None,
        column_map=None):
    """Print token-length statistics for a dataset."""
    print(f"Loading {dataset_path}...")
    data = read_dataset(dataset_path, column_map)
    data = clean_dataset(data)
    column = column or "clean_text"
    texts = data[column if column in data.columns else TEXT_COLUMN].tolist()

    tokenizer = None
    if offline:
        print("Training a WordPiece tokenizer on this corpus (offline mode).")
        tokenizer = build_local_tokenizer(texts)

    stats = token_length_stats(texts, model_name=model_name, tokenizer=tokenizer)
    print(f"\nToken lengths across {stats['count']} posts")
    for key in ("mean", "min", "max", "p50", "p75", "p90", "p95", "p99"):
        if key in stats:
            print(f"  {key:<6} {stats[key]:.1f}" if isinstance(stats[key], float)
                  else f"  {key:<6} {stats[key]}")

    print("\nShare of posts truncated at each candidate --max-length:")
    for cutoff, rate in stats["truncation_rate"].items():
        print(f"  {cutoff:>4}: {rate:6.1%}")

    # Note the offline caveat rather than letting someone tune on the wrong
    # vocabulary: a corpus-trained WordPiece splits words differently from
    # BERT's 30k vocabulary, so these lengths do not transfer.
    if offline:
        print(
            "\nNOTE: measured with a corpus-trained tokenizer, not BERT's. The "
            "vocabularies differ, so re-measure with the real tokenizer before "
            "settling on --max-length."
        )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Token-length statistics for a dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--offline", action="store_true",
                        help="Use a corpus-trained tokenizer instead of downloading one.")
    parser.add_argument("--column", default=None, help="Text column to measure.")
    parser.add_argument("--column-map", default="",
                        help="canonical=source pairs, e.g. 'selftext=body'.")
    args = parser.parse_args()
    run(args.dataset, args.model_name, args.offline, args.column,
        column_map=args.column_map)


if __name__ == "__main__":
    main()
