"""Run every model x feature-set combination and print one comparison table.

The grid is TF-IDF vs BERT, each with and without the temporal features. That
2x2 answers the two questions the project actually poses:

  * Is the transformer earning its cost over a linear baseline?
  * Are the timestamps contributing anything, in either architecture?

Every cell trains on the same rows, the same labels, and the same split, and is
scored by the same compute_metrics, so the numbers are directly comparable.

    python benchmark.py --synthetic --tiny-model --epochs 6 --learning-rate 1e-3
    python benchmark.py --dataset datasets/posts.csv --epochs 3
"""

import argparse
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import main as pipeline  # noqa: E402
from models.embedding_baseline import DEFAULT_ENCODER  # noqa: E402
from models.llm_baseline import DEFAULT_MODEL as DEFAULT_LLM_MODEL  # noqa: E402
from models.model_utils import save_metrics, set_seed  # noqa: E402
from training.config import Config  # noqa: E402


def _build_config(args, use_temporal):
    return Config(
        dataset_path=args.dataset,
        label_strategy=args.label_strategy,
        split_strategy=args.split_strategy,
        max_rows=args.max_rows,
        column_map=args.column_map,
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        llm_model=args.llm_model,
        max_length=args.max_length,
        use_temporal_features=use_temporal,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )


def run_grid(args):
    """Run each configuration and collect its validation metrics."""
    rows = []

    for use_temporal in (False, True):
        config = _build_config(args, use_temporal)
        config.ensure_dirs()
        set_seed(config.seed)

        feature_set = "text + temporal" if use_temporal else "text only"
        print(f"\n{'=' * 70}\nTF-IDF | {feature_set}\n{'=' * 70}")
        results, _ = pipeline.run_tfidf_baseline(
            config, run_analysis=False, show_features=False
        )
        rows.append({"model": "TF-IDF", "features": feature_set,
                     "metrics": results["final_metrics"]})

    # Frozen embeddings sit between the two: pretrained semantics, no
    # task-specific training. Opt-in, because unlike the TF-IDF half it needs a
    # model download and so cannot run on the offline path.
    if args.embeddings:
        for use_temporal in (False, True):
            config = _build_config(args, use_temporal)
            config.ensure_dirs()
            set_seed(config.seed)

            feature_set = "text + temporal" if use_temporal else "text only"
            label = f"Frozen {args.encoder_name.split('/')[-1]}"
            print(f"\n{'=' * 70}\n{label} | {feature_set}\n{'=' * 70}")
            results, _ = pipeline.run_embedding_baseline(config, run_analysis=False)
            rows.append({"model": label, "features": feature_set,
                         "metrics": results["final_metrics"]})

    # Zero-shot LLM: no task training at all, so it splits a different axis
    # than the frozen-embedding row. Opt-in and billed per validation row --
    # never reached unless --llm is passed explicitly.
    if args.llm:
        for use_temporal in (False, True):
            config = _build_config(args, use_temporal)
            config.ensure_dirs()
            set_seed(config.seed)

            feature_set = "text + temporal" if use_temporal else "text only"
            label = f"Zero-shot {args.llm_model}"
            print(f"\n{'=' * 70}\n{label} | {feature_set}\n{'=' * 70}")
            results, _ = pipeline.run_llm_baseline(config, run_analysis=False,
                                                   dry_run=False)
            rows.append({"model": label, "features": feature_set,
                         "metrics": results["final_metrics"]})

    if args.skip_bert:
        return rows

    for use_temporal in (False, True):
        config = _build_config(args, use_temporal)
        config.ensure_dirs()
        set_seed(config.seed)

        feature_set = "text + temporal" if use_temporal else "text only"
        label = "BERT (tiny, random)" if args.tiny_model else "BERT"
        print(f"\n{'=' * 70}\n{label} | {feature_set}\n{'=' * 70}")

        offline = args.offline_tokenizer or args.tiny_model
        train_loader, val_loader, info = pipeline.load_and_prepare_data(
            config, run_analysis=False, offline_tokenizer=offline
        )
        # Take the vocabulary size from the tokenizer that was actually built.
        # Deriving it from the training split's highest token id would undersize
        # the embedding table whenever the validation split contains a rarer
        # token, and that surfaces as an index error mid-evaluation.
        model = pipeline.build_model(config, tiny=args.tiny_model,
                                     vocab_size=info.get("vocab_size"))
        from models.model_utils import resolve_device
        from training.evaluate import evaluate_model
        from training.train import build_optimizer, train_model

        device = resolve_device(config.device)
        train_model(
            train_loader, model,
            optimizer=build_optimizer(model, config.learning_rate, config.weight_decay),
            epochs=config.epochs, val_loader=val_loader, device=device,
            log_every=0, config=config,
        )
        metrics = evaluate_model(val_loader, model, device=device, verbose=False)
        rows.append({"model": label, "features": feature_set, "metrics": metrics})

    return rows


def print_table(rows):
    """Print the comparison as a markdown table, ready to paste into the README."""
    if not rows:
        return

    baseline = rows[0]["metrics"]["majority_baseline"]
    print(f"\n{'=' * 70}\nRESULTS\n{'=' * 70}")
    print(f"Majority-class baseline: {baseline:.4f}\n")

    header = f"| {'Model':<22} | {'Features':<16} | {'Accuracy':>8} | {'Macro F1':>8} | {'Lift':>7} |"
    print(header)
    print("|" + "-" * 24 + "|" + "-" * 18 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 9 + "|")
    for row in rows:
        m = row["metrics"]
        print(f"| {row['model']:<22} | {row['features']:<16} | {m['accuracy']:>8.4f} | "
              f"{m['macro_f1']:>8.4f} | {m['lift_over_baseline']:>+7.4f} |")

    best = max(rows, key=lambda r: r["metrics"]["accuracy"])
    print(f"\nBest: {best['model']} ({best['features']}) at "
          f"{best['metrics']['accuracy']:.4f}")

    # The comparison only means something if both sides were given a fair run.
    tfidf = [r for r in rows if r["model"] == "TF-IDF"]
    bert = [r for r in rows if r["model"].startswith("BERT")]
    if tfidf and bert:
        best_tfidf = max(tfidf, key=lambda r: r["metrics"]["accuracy"])["metrics"]["accuracy"]
        best_bert = max(bert, key=lambda r: r["metrics"]["accuracy"])["metrics"]["accuracy"]
        if best_tfidf > best_bert:
            print(
                f"\nThe linear baseline beats the transformer here "
                f"({best_tfidf:.4f} vs {best_bert:.4f})."
            )
            if any("tiny, random" in r["model"] for r in bert):
                print(
                    "  Note: that BERT is randomly initialised with no pretrained\n"
                    "  weights, so this is not evidence about real BERT. Re-run\n"
                    "  without --tiny-model before drawing any conclusion."
                )
            else:
                print(
                    "  Worth taking seriously rather than tuning away: on short\n"
                    "  text with a few thousand rows this is a common and\n"
                    "  legitimate outcome."
                )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark TF-IDF against BERT, with and without temporal features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--column-map", default="",
                        help="canonical=source pairs, e.g. 'selftext=body'.")
    parser.add_argument("--label-strategy", default="median",
                        choices=["median", "positive", "threshold"])
    parser.add_argument("--split-strategy", default="random",
                        choices=["temporal", "random"])
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--offline-tokenizer", action="store_true")
    parser.add_argument("--skip-bert", action="store_true",
                        help="Only run the TF-IDF half. Fast, and needs no network.")
    parser.add_argument("--embeddings", action="store_true",
                        help="Add a frozen sentence-embedding row between TF-IDF "
                             "and BERT. Needs a model download.")
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER,
                        help="Encoder for the --embeddings row.")
    parser.add_argument("--llm", action="store_true",
                        help="Add a zero-shot LLM row. COSTS MONEY: one API "
                             "call per validation row, per ablation arm.")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,
                        help="Model for the --llm row.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default=None, help="Write the table as JSON here.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.synthetic or args.dataset is None:
        from training.config import DATASET_DIR
        sample = DATASET_DIR / "sample_posts.csv"
        if not sample.exists():
            from data.make_sample_data import write_dataset
            print("Generating synthetic sample dataset...")
            write_dataset(sample)
        args.dataset = sample

    rows = run_grid(args)
    print_table(rows)

    out = args.out or (Config().checkpoint_dir.parent / "benchmark.json")
    save_metrics({"rows": rows}, out)
    print(f"\nWritten to {out}")
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
