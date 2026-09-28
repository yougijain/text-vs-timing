"""Run the whole experiment and write RESULTS.md. One command, four steps.

The results document is the deliverable, and the surest way to get a wrong one
is to run the pieces separately over a couple of days and assemble the table by
hand -- a grid from Tuesday next to an error analysis from Thursday, on a
different seed, looks exactly like a consistent result.

So: one entry point, one seed, one split strategy, one dataset, straight
through to the document.

  1. benchmark.py       -> outputs/benchmark.json          (the grid)
  2. error_analysis     -> outputs/error_analysis.json     (slices, ECE)
  3. --compare-feature-sets -> outputs/feature_set_comparison.json
                                                           (fixed vs broken)
  4. analysis.report    -> RESULTS.md
  5. analysis.site      -> docs/index.html              (with --site)

The real run, on a T4 or better:

    python run_experiment.py --dataset datasets/posts.csv --epochs 3 \\
        --split-strategy temporal --label-strategy median \\
        --max-length 256 --batch-size 16 --embeddings --site

The offline smoke run, in about a minute on CPU:

    python run_experiment.py --synthetic --skip-bert
"""

import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path  # noqa: E402

from training.config import DATASET_DIR, OUTPUT_DIR, PROJECT_ROOT  # noqa: E402


def _banner(step, total, title):
    print(f"\n{'=' * 70}\n[{step}/{total}] {title}\n{'=' * 70}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the grid, the error analysis, and write RESULTS.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", default=None, help="Path to the CSV corpus.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use the generated sample data. Smoke runs only.")
    parser.add_argument("--column-map", default="",
                        help="canonical=source pairs, e.g. 'selftext=body'.")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--label-strategy", default="median",
                        choices=["median", "positive", "threshold"])
    parser.add_argument("--split-strategy", default="temporal",
                        choices=["temporal", "random"],
                        help="Chronological by default: the timestamp is a "
                             "feature, so a random split lets the model see "
                             "the future.")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--encoder-name", default=None,
                        help="Encoder for the frozen-embedding row.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tiny-model", action="store_true",
                        help="Random miniature BERT. Plumbing check, not a result.")
    parser.add_argument("--offline-tokenizer", action="store_true")
    parser.add_argument("--skip-bert", action="store_true",
                        help="Linear half only. Fast, and needs no network.")
    parser.add_argument("--embeddings", action="store_true",
                        help="Include the frozen sentence-embedding row.")
    parser.add_argument("--llm", action="store_true",
                        help="Include the zero-shot LLM row. COSTS MONEY.")
    parser.add_argument("--llm-model", default=None,
                        help="Model for the --llm row.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--results", default=str(PROJECT_ROOT / "RESULTS.md"))
    parser.add_argument("--site", action="store_true",
                        help="Also write the published page under docs/.")
    parser.add_argument("--site-dir", default=str(PROJECT_ROOT / "docs"))
    return parser.parse_args(argv)


def resolve_dataset(args):
    """Pick the dataset, generating the sample one only when asked."""
    if args.dataset:
        path = Path(args.dataset)
        if not path.exists():
            raise FileNotFoundError(f"No dataset at {path}.")
        return path

    sample = DATASET_DIR / "sample_posts.csv"
    if not args.synthetic:
        raise SystemExit(
            "Pass --dataset path/to/corpus.csv, or --synthetic for a smoke run.\n"
            "Fetch a real corpus with:\n"
            "  python -m data.fetch_dataset --site stackoverflow --rows 20000 "
            "--out datasets/posts.csv"
        )
    if not sample.exists():
        from data.make_sample_data import write_dataset
        print("Generating synthetic sample dataset...")
        write_dataset(sample)
    return sample


def main(argv=None):
    args = parse_args(argv)
    dataset = resolve_dataset(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic or args.tiny_model:
        print(
            "\nNOTE: this run uses "
            + ("generated sample data" if args.synthetic else "")
            + (" and " if args.synthetic and args.tiny_model else "")
            + ("a randomly initialised miniature BERT" if args.tiny_model else "")
            + ". The document it writes is a plumbing check, not a result."
        )

    started = time.time()
    total_steps = 5 if args.site else 4

    # --- 1. the grid ---------------------------------------------------
    _banner(1, total_steps, "Benchmark grid")
    import benchmark

    grid_argv = [
        "--dataset", str(dataset),
        "--column-map", args.column_map,
        "--max-rows", str(args.max_rows),
        "--label-strategy", args.label_strategy,
        "--split-strategy", args.split_strategy,
        "--model-name", args.model_name,
        "--max-length", str(args.max_length),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed),
        "--device", args.device,
        "--out", str(output_dir / "benchmark.json"),
    ]
    if args.encoder_name:
        grid_argv += ["--encoder-name", args.encoder_name]
    if args.llm_model:
        grid_argv += ["--llm-model", args.llm_model]
    for flag, enabled in (("--tiny-model", args.tiny_model),
                          ("--offline-tokenizer", args.offline_tokenizer),
                          ("--skip-bert", args.skip_bert),
                          ("--embeddings", args.embeddings),
                          ("--llm", args.llm)):
        if enabled:
            grid_argv.append(flag)
    benchmark.main(grid_argv)

    # --- 2 & 3. error analysis ------------------------------------------
    # Both run on the TF-IDF baseline: it fits in seconds and needs no network,
    # which is what makes error analysis something that actually gets run.
    from analysis.error_analysis import analyse_dataset, compare_feature_sets

    _banner(2, total_steps, "Error analysis: slices and calibration")
    analyse_dataset(
        dataset, label_strategy=args.label_strategy,
        split_strategy=args.split_strategy, seed=args.seed,
        output_dir=output_dir / "figures", max_rows=args.max_rows,
        json_out=output_dir / "error_analysis.json",
        column_map=args.column_map,
    )

    _banner(3, total_steps, "Error analysis: does the timestamp fix or shuffle?")
    compare_feature_sets(
        dataset, label_strategy=args.label_strategy,
        split_strategy=args.split_strategy, seed=args.seed,
        max_rows=args.max_rows,
        json_out=output_dir / "feature_set_comparison.json",
        column_map=args.column_map,
    )

    # --- 4. the document -------------------------------------------------
    _banner(4, total_steps, "Writing RESULTS.md")
    from analysis.report import load_artifacts, write_report

    artifacts = load_artifacts(output_dir, dataset)
    # A document built from generated text, or from a randomly initialised
    # model, is stamped as such at the top -- that is the line a reader skims.
    path = write_report(args.results, artifacts,
                        synthetic=args.synthetic or args.tiny_model)

    # --- 5. the page ------------------------------------------------------
    # Opt-in, because docs/ is tracked and published: a smoke run must not
    # quietly rewrite the page the live link serves.
    if args.site:
        _banner(5, total_steps, "Writing the published page")
        from analysis.site import build_site

        page = build_site(
            artifacts, out_dir=args.site_dir,
            synthetic=args.synthetic or args.tiny_model,
            figure_dir=output_dir / "figures",
        )
        print(f"Wrote {page}")

    elapsed = time.time() - started
    print(f"\nWrote {path} in {elapsed / 60:.1f} min.")
    if not artifacts.get("provenance"):
        print("  No provenance sidecar for this dataset -- that section of the "
              "report is a placeholder.\n  Fetch with `python -m data.fetch_dataset` "
              "to get one.")
    return path


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
