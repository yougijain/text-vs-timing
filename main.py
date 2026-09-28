"""End-to-end pipeline: load, clean, analyse, train, evaluate.

Examples:
    # Train on the bundled synthetic data (generates it if missing).
    python main.py --synthetic --epochs 2

    # Train on a real corpus.
    python main.py --dataset datasets/posts.csv --epochs 3

    # Text-only ablation, to measure what the temporal branch is worth.
    python main.py --synthetic --no-temporal

    # Quick smoke run on a tiny random model -- no 440MB download.
    python main.py --synthetic --tiny-model --epochs 1 --max-rows 200
"""

import argparse
import os
import sys


# Silence the HF symlink warning before transformers is imported, otherwise the
# setting arrives too late to have any effect.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from analysis.pattern_detection import (  # noqa: E402
    add_marker_columns,
    marker_label_correlations,
)
from analysis.timestamp_analysis import (  # noqa: E402
    plot_hourly_activity,
    plot_loss_curve,
    plot_weekly_heatmap,
    summarize_temporal,
)
from data.data_loader import build_dataloaders, split_indices  # noqa: E402
from data.schema import describe_schema, read_dataset  # noqa: E402
from data.preprocess import (  # noqa: E402
    batch_tokenize,
    build_labels,
    build_local_tokenizer,
    clean_dataset,
)
from models.bert_temporal_model import BertTemporalClassifier  # noqa: E402
from models.embedding_baseline import (  # noqa: E402
    DEFAULT_ENCODER,
    EmbeddingBaseline,
)
from models.llm_baseline import (  # noqa: E402
    DEFAULT_MODEL as DEFAULT_LLM_MODEL,
    LlmBaseline,
)
from models.tfidf_baseline import TfidfBaseline, print_top_features  # noqa: E402
from models.model_utils import (  # noqa: E402
    describe_model,
    resolve_device,
    save_metrics,
    set_seed,
)
from training.config import Config, TIMESTAMP_COLUMN  # noqa: E402
from training.evaluate import evaluate_model  # noqa: E402
from training.train import build_optimizer, train_model  # noqa: E402
from utils.time_utils import add_temporal_features, temporal_feature_matrix  # noqa: E402


def prepare_frame(config, run_analysis=True):
    """Load, clean, featurise, and label the dataset.

    Everything both models share, so the TF-IDF baseline and the neural model
    see identical rows, identical labels, and (via config.seed and
    config.split_strategy) identical splits. A baseline trained on a different
    split is not a comparison.

    Args:
        config (Config): Run configuration.
        run_analysis (bool): Produce the descriptive analysis and figures.
    Returns:
        tuple: (data, labels, temporal_matrix, info_dict). temporal_matrix is
        None when config.use_temporal_features is False.
    """
    print(f"Loading dataset from {config.dataset_path}...")
    if not config.dataset_path.exists():
        raise FileNotFoundError(
            f"No dataset at {config.dataset_path}. Either point --dataset at a "
            "real CSV, or pass --synthetic to generate a sample one."
        )
    # Map whatever the corpus calls its columns onto the names every module
    # below refers to. Explicit --column-map wins; inference only fills columns
    # that are genuinely absent.
    data = read_dataset(config.dataset_path, config.column_map)
    print(f"  {len(data)} rows")
    print(describe_schema(data))

    data = clean_dataset(data, min_tokens=config.min_tokens)

    print("Deriving temporal features...")
    data = add_temporal_features(data, TIMESTAMP_COLUMN)

    # Sort chronologically so a temporal split really is train-on-past,
    # validate-on-future. The old code split on whatever order the CSV happened
    # to be in and called it a split.
    if config.split_strategy == "temporal":
        data = data.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    if config.max_rows and config.max_rows < len(data):
        print(f"  Subsampling to {config.max_rows} rows (--max-rows).")
        data = data.head(config.max_rows).reset_index(drop=True)

    labels, label_summary = build_labels(
        data, strategy=config.label_strategy, threshold=config.label_threshold
    )
    print(f"  Labels: {label_summary}")

    info = {"label_summary": label_summary, "num_rows": int(len(data))}

    if run_analysis:
        print("\nRunning descriptive analysis...")
        config.ensure_dirs()
        temporal_summary = summarize_temporal(data, labels)
        for key, value in temporal_summary.items():
            print(f"  {key}: {value}")
        info["temporal_summary"] = temporal_summary

        with_markers = add_marker_columns(data)
        correlations = marker_label_correlations(with_markers, labels)
        print("\n  Top linguistic markers by label correlation:")
        for name, value in correlations.head(5).items():
            print(f"    {name:<26} {value:+.4f}")
        info["marker_correlations"] = correlations.round(6).to_dict()

        figures = [
            plot_hourly_activity(data, config.figure_dir, labels),
            plot_weekly_heatmap(data, config.figure_dir),
        ]
        info["figures"] = [str(p) for p in figures]
        print("  Figures: " + ", ".join(str(p.name) for p in figures))

    temporal_matrix = None
    if config.use_temporal_features:
        temporal_matrix = temporal_feature_matrix(data, config.temporal_features)
        print(f"  Temporal feature matrix: {temporal_matrix.shape}")

    return data, labels, temporal_matrix, info


def load_and_prepare_data(config, run_analysis=True, offline_tokenizer=False):
    """Tokenize and wrap the prepared frame in DataLoaders, for the neural path.

    Args:
        config (Config): Run configuration.
        run_analysis (bool): Produce the descriptive analysis and figures.
        offline_tokenizer (bool): Train a WordPiece tokenizer on this corpus
            instead of downloading one. Lets the pipeline run with no network.
    Returns:
        tuple: (train_loader, val_loader, info_dict)
    """
    data, labels, temporal_matrix, info = prepare_frame(config, run_analysis)

    print("\nTokenizing...")
    tokenizer = None
    if offline_tokenizer:
        print("  Training a WordPiece tokenizer on this corpus (offline mode).")
        tokenizer = build_local_tokenizer(data["clean_text"].tolist())
        info["tokenizer"] = f"local-wordpiece (vocab {len(tokenizer)})"
        print(f"  Vocabulary size: {len(tokenizer)}")
    encodings = batch_tokenize(
        data["clean_text"].tolist(),
        model_name=config.model_name,
        batch_size=config.tokenize_batch_size,
        max_length=config.max_length,
        tokenizer=tokenizer,
    )
    info["vocab_size"] = int(len(tokenizer)) if tokenizer is not None else None

    print("\nBuilding dataloaders...")
    train_loader, val_loader = build_dataloaders(
        encodings,
        labels,
        temporal_features=temporal_matrix,
        batch_size=config.batch_size,
        val_split=config.val_split,
        strategy=config.split_strategy,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, info


def run_sklearn_baseline(config, model, label, run_analysis=True):
    """Fit and score any sklearn-style baseline on the shared rows and split.

    The TF-IDF and frozen-embedding baselines differ only in how they turn text
    into a matrix, so everything around that -- the rows, the labels, the split
    indices, the metrics -- lives here. Two models that were prepared
    differently are not a comparison, and the cheapest way to guarantee they
    were not is to have one code path.

    Args:
        config (Config): Run configuration.
        model: Anything exposing fit/evaluate/num_features, i.e. TfidfBaseline
            or EmbeddingBaseline.
        label (str): Name for the log line.
        run_analysis (bool): Produce the descriptive analysis and figures.
    Returns:
        tuple: (results_dict, fitted_model)
    """
    data, labels, temporal_matrix, info = prepare_frame(config, run_analysis)
    texts = data["clean_text"].tolist()

    train_idx, val_idx = split_indices(
        len(data), config.val_split, config.split_strategy, config.seed
    )
    print(f"\n  train: {len(train_idx)} samples | val: {len(val_idx)} samples")

    def take(values, idx):
        return None if values is None else values[idx]

    print(f"\nFitting {label}...")
    model.fit(
        [texts[i] for i in train_idx],
        labels[train_idx],
        take(temporal_matrix, train_idx),
    )
    print(f"  {model.num_features:,} features")

    metrics = model.evaluate(
        [texts[i] for i in val_idx],
        labels[val_idx],
        take(temporal_matrix, val_idx),
        verbose=True,
    )

    results = {"config": config.to_dict(), "data": info, "final_metrics": metrics,
               "num_features": int(model.num_features)}
    return results, model


def run_tfidf_baseline(config, run_analysis=True, show_features=True):
    """Train and evaluate the TF-IDF baseline on the same data and split.

    Returns:
        dict: config, data info, metrics, and the top signed features.
    """
    model = TfidfBaseline(
        use_temporal_features=config.use_temporal_features,
        seed=config.seed,
    )
    results, model = run_sklearn_baseline(config, model, "TF-IDF baseline",
                                          run_analysis)

    if show_features:
        top = model.top_features(n=15, temporal_feature_names=config.temporal_features)
        print_top_features(top, n=10)
        results["top_features"] = top

    return results, model


def run_embedding_baseline(config, run_analysis=True, embed_fn=None):
    """Train and evaluate the frozen-embedding baseline on the same split.

    The middle of the grid: pretrained semantics without task-specific
    training, so a gap between this row and fine-tuned BERT is attributable to
    the fine-tuning rather than to the encoder.

    Args:
        embed_fn (callable | None): `texts -> (n, d)` array, bypassing the real
            encoder. Used by the tests; also the hook for cached embeddings.
    Returns:
        tuple: (results_dict, fitted_model)
    """
    model = EmbeddingBaseline(
        model_name=config.encoder_name,
        use_temporal_features=config.use_temporal_features,
        max_length=config.max_length,
        device=config.device,
        seed=config.seed,
        embed_fn=embed_fn,
    )
    results, model = run_sklearn_baseline(
        config, model, f"frozen embeddings ({config.encoder_name})", run_analysis
    )
    results["encoder_name"] = model.describe_encoder()
    results["embedding_dim"] = model.embedding_dim
    return results, model


def run_llm_baseline(config, run_analysis=True, classify_fn=None,
                     dry_run=True, cache_path=None):
    """Train-free LLM row. Costs money, so it is opt-in at every layer.

    The zero-shot row splits an axis the other three cannot: every other model
    learns this task from this corpus, and this one has never seen a label.

    Args:
        classify_fn (callable | None): `prompt -> 0|1`, bypassing the API.
        dry_run (bool): When true and no classify_fn is given, the row refuses
            to call the model and reports what a real pass would cost.
        cache_path: Disk cache, so a re-run of the grid is free.
    """
    model = LlmBaseline(
        model=config.llm_model,
        use_temporal_features=config.use_temporal_features,
        classify_fn=classify_fn,
        cache_path=cache_path or (config.checkpoint_dir.parent / "llm_cache.json"),
        dry_run=dry_run,
    )
    results, model = run_sklearn_baseline(
        config, model, f"zero-shot LLM ({config.llm_model})", run_analysis
    )
    results["llm_model"] = model.describe_model()
    return results, model


def build_model(config, tiny=False, vocab_size=None):
    """Construct the classifier, optionally as the tiny test model.

    `vocab_size` must match the tokenizer actually in use -- the offline path
    trains its own vocabulary, which is far smaller than BERT's 30522.
    """
    n_features = config.num_temporal_features
    if tiny:
        print("Building a randomly initialised tiny BERT (smoke-test mode).")
        return BertTemporalClassifier.tiny_for_testing(
            num_temporal_features=n_features,
            num_labels=config.num_labels,
            use_attention_pooling=config.use_attention_pooling,
            vocab_size=vocab_size or 2048,
            max_position_embeddings=max(config.max_length, 64),
        )
    print(f"Loading encoder '{config.model_name}'...")
    return BertTemporalClassifier.from_config(config)


def _report_lift(metrics):
    """The number that actually matters: beating the majority-class baseline."""
    lift = metrics.get("lift_over_baseline", 0.0)
    if lift <= 0:
        print(
            f"\nNOTE: accuracy {metrics['accuracy']:.4f} is at or below the "
            f"majority-class baseline {metrics['majority_baseline']:.4f}. The "
            "model has not learned anything useful -- train longer, unfreeze the "
            "encoder, or check the label strategy."
        )
    else:
        print(f"\nBeat the majority-class baseline by {lift:+.4f} accuracy.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Text-vs-timing engagement classification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--dataset", default=None, help="Path to the CSV dataset.")
    data_group.add_argument("--synthetic", action="store_true",
                            help="Use (and generate if missing) the synthetic sample dataset.")
    data_group.add_argument("--max-rows", type=int, default=0,
                            help="Cap the number of rows used. 0 uses everything.")
    data_group.add_argument("--column-map", default="",
                            help="Map your CSV's columns onto the ones the pipeline "
                                 "expects, as canonical=source pairs, e.g. "
                                 "'selftext=body,created_utc=creation_date'. "
                                 "Unmapped columns are inferred from known aliases.")
    data_group.add_argument("--label-strategy", default="median",
                            choices=["median", "positive", "threshold"],
                            help="How to turn score into a binary label.")
    data_group.add_argument("--split-strategy", default="temporal",
                            choices=["temporal", "random"],
                            help="Chronological or shuffled train/val split.")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", default="bert",
                             choices=["bert", "tfidf", "embeddings", "llm"],
                             help="Which classifier to run. 'tfidf' is the linear "
                                  "baseline and 'embeddings' is frozen sentence "
                                  "vectors plus logistic regression; both ignore "
                                  "the BERT flags.")
    model_group.add_argument("--encoder-name", default=DEFAULT_ENCODER,
                             help="Encoder for --model embeddings. Frozen, never "
                                  "fine-tuned.")
    model_group.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,
                             help="Model for --model llm. Costs money to run.")
    model_group.add_argument("--model-name", default="bert-base-uncased")
    model_group.add_argument("--max-length", type=int, default=256)
    model_group.add_argument("--no-temporal", action="store_true",
                             help="Text-only ablation: drop the temporal branch.")
    model_group.add_argument("--no-attention-pooling", action="store_true",
                             help="Use mean pooling instead of learned attention.")
    model_group.add_argument("--freeze-bert", action="store_true",
                             help="Train only the head. Much faster on CPU.")
    model_group.add_argument("--tiny-model", action="store_true",
                             help="Random miniature BERT for smoke tests. No download.")
    model_group.add_argument("--offline-tokenizer", action="store_true",
                             help="Train a WordPiece tokenizer on this corpus instead "
                                  "of downloading one. Implied by --tiny-model.")

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--epochs", type=int, default=3)
    train_group.add_argument("--batch-size", type=int, default=16)
    train_group.add_argument("--learning-rate", type=float, default=2e-5)
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    train_group.add_argument("--no-amp", action="store_true",
                             help="Disable mixed precision even on CUDA.")
    train_group.add_argument("--skip-analysis", action="store_true",
                             help="Skip the descriptive analysis and figures.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    dataset_path = args.dataset
    if args.synthetic or dataset_path is None:
        from training.config import DATASET_DIR
        sample_path = DATASET_DIR / "sample_posts.csv"
        if dataset_path is None and not args.synthetic and not sample_path.exists():
            print(
                "No --dataset given. Falling back to synthetic sample data.\n"
                "Pass --dataset path/to/posts.csv for results that mean something.\n"
            )
        if not sample_path.exists():
            from data.make_sample_data import write_dataset
            print("Generating synthetic sample dataset...")
            write_dataset(sample_path)
        dataset_path = sample_path

    config = Config(
        dataset_path=dataset_path,
        label_strategy=args.label_strategy,
        split_strategy=args.split_strategy,
        max_rows=args.max_rows,
        column_map=args.column_map,
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        llm_model=args.llm_model,
        max_length=args.max_length,
        use_temporal_features=not args.no_temporal,
        use_attention_pooling=not args.no_attention_pooling,
        freeze_bert=args.freeze_bert,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        use_amp=not args.no_amp,
    )
    config.ensure_dirs()
    set_seed(config.seed)

    if config.use_temporal_features:
        print(f"Temporal branch: ON ({config.num_temporal_features} features)")
    else:
        print("Temporal branch: OFF (text-only ablation)")

    if args.model in ("tfidf", "embeddings", "llm"):
        if args.model == "tfidf":
            results, _ = run_tfidf_baseline(config, run_analysis=not args.skip_analysis)
        elif args.model == "embeddings":
            results, _ = run_embedding_baseline(config,
                                                run_analysis=not args.skip_analysis)
        else:
            # Reaching this line is the opt-in; --model llm is not a default
            # anywhere and the row bills per row of the validation split.
            results, _ = run_llm_baseline(config,
                                          run_analysis=not args.skip_analysis,
                                          dry_run=False)
        metrics_path = save_metrics(
            results, config.checkpoint_dir.parent / f"results_{args.model}.json"
        )
        print(f"\nResults written to {metrics_path}")
        _report_lift(results["final_metrics"])
        return results

    offline_tokenizer = args.offline_tokenizer or args.tiny_model
    train_loader, val_loader, info = load_and_prepare_data(
        config,
        run_analysis=not args.skip_analysis,
        offline_tokenizer=offline_tokenizer,
    )

    model = build_model(config, tiny=args.tiny_model, vocab_size=info.get("vocab_size"))
    print(describe_model(model))

    device = resolve_device(config.device)
    optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)

    checkpoint_path = config.checkpoint_dir / "bert_temporal_model.pth"
    history = train_model(
        train_loader,
        model,
        optimizer=optimizer,
        epochs=config.epochs,
        save_path=checkpoint_path,
        val_loader=val_loader,
        device=device,
        use_amp=config.use_amp,
        max_grad_norm=config.max_grad_norm,
        log_every=config.log_every,
        config=config,
        save_every_epoch=config.save_every_epoch,
    )

    if history["batch_losses"]:
        loss_figure = plot_loss_curve(history["batch_losses"], config.figure_dir)
        print(f"Loss curve: {loss_figure}")

    print("\nFinal evaluation")
    metrics = evaluate_model(val_loader, model, device=device, verbose=True)

    results = {
        "config": config.to_dict(),
        "data": info,
        "history": history,
        "final_metrics": metrics,
    }
    metrics_path = save_metrics(results, config.checkpoint_dir.parent / "results.json")
    print(f"\nResults written to {metrics_path}")

    _report_lift(metrics)
    return results


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
