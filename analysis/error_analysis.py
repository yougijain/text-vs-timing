"""Error analysis: what the model gets wrong, and under what conditions.

An accuracy number tells you how often a model is right. It does not tell you
whether the mistakes are spread evenly or concentrated somewhere specific, and
that difference decides whether a result is usable. A classifier at 70% overall
that collapses to 45% on late-night posts has a problem the headline hides --
and on this project that particular slice is the whole thesis, so it is worth
checking rather than assuming.

What this module answers:

  * Which slices does it fail on? Error rate by posting hour, weekday, text
    length, and the linguistic markers.
  * Is it failing symmetrically? A model that only ever errs in one direction
    is miscalibrated, not merely inaccurate.
  * Are the predicted probabilities meaningful? A confident wrong answer is
    worse than an unsure one, and expected calibration error quantifies that.
  * Which specific posts does it get most confidently wrong? Those are the
    instructive ones -- read them.
  * Where do the two models disagree, and who is right when they do?
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.pattern_detection import add_marker_columns  # noqa: E402
from data.data_loader import split_indices  # noqa: E402
from data.preprocess import build_labels, clean_dataset  # noqa: E402
from data.schema import read_dataset  # noqa: E402
from training.config import (  # noqa: E402
    FIGURE_DIR,
    OUTPUT_DIR,
    TEMPORAL_FEATURES,
    TIMESTAMP_COLUMN,
)
from utils.time_utils import add_temporal_features, temporal_feature_matrix  # noqa: E402

# A slice with a handful of rows produces a meaningless error rate -- 2 of 3
# wrong reads as 67% and means nothing. Slices below this are reported but
# flagged, never ranked.
MIN_SLICE_COUNT = 15

ERROR_TYPES = {
    (0, 0): "true_negative",
    (1, 1): "true_positive",
    (0, 1): "false_positive",  # predicted 1, actually 0
    (1, 0): "false_negative",  # predicted 0, actually 1
}


def build_error_frame(data, y_true, y_pred, y_prob=None):
    """Join predictions back onto the source rows, one record per example.

    Args:
        data (pd.DataFrame): The rows that were predicted on, in prediction
            order. Index is reset so it aligns positionally.
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like, optional): Class probabilities, shape (n, n_classes)
            or (n,) for the positive class. Enables confidence and calibration.
    Returns:
        pd.DataFrame: The original columns plus y_true, y_pred, correct,
        error_type, and (when probabilities were given) prob_positive and
        confidence.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(data) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: {len(data)} rows, {len(y_true)} labels, "
            f"{len(y_pred)} predictions. All three must align positionally."
        )

    frame = data.reset_index(drop=True).copy()
    frame["y_true"] = y_true
    frame["y_pred"] = y_pred
    frame["correct"] = (y_true == y_pred)
    frame["error_type"] = [
        ERROR_TYPES.get((int(t), int(p)), "other") for t, p in zip(y_true, y_pred)
    ]

    if y_prob is not None:
        probabilities = np.asarray(y_prob)
        if probabilities.ndim == 1:
            positive = probabilities
        elif probabilities.shape[1] == 2:
            positive = probabilities[:, 1]
        else:
            # Multiclass: keep the predicted class's probability as confidence
            # and leave prob_positive undefined.
            frame["confidence"] = probabilities.max(axis=1)
            return frame
        frame["prob_positive"] = positive
        # For a binary classifier the confidence in whichever class was
        # predicted, so it sits in [0.5, 1.0] regardless of direction.
        frame["confidence"] = np.maximum(positive, 1.0 - positive)

    return frame


def error_summary(frame):
    """Overall counts and the asymmetry between the two error directions."""
    counts = frame["error_type"].value_counts().to_dict()
    n = len(frame)
    false_pos = counts.get("false_positive", 0)
    false_neg = counts.get("false_negative", 0)
    total_errors = false_pos + false_neg

    summary = {
        "num_samples": int(n),
        "num_errors": int(total_errors),
        "error_rate": float(total_errors / n) if n else 0.0,
        "false_positive": int(false_pos),
        "false_negative": int(false_neg),
        "true_positive": int(counts.get("true_positive", 0)),
        "true_negative": int(counts.get("true_negative", 0)),
    }

    # A model whose mistakes run almost entirely one way is systematically
    # biased toward a class rather than uniformly imprecise, and the fix for
    # that is a threshold change, not more training.
    if total_errors:
        summary["false_positive_share"] = float(false_pos / total_errors)
        skew = abs(false_pos - false_neg) / total_errors
        summary["error_skew"] = float(skew)
        summary["one_sided"] = bool(skew > 0.6)
    else:
        summary["false_positive_share"] = 0.0
        summary["error_skew"] = 0.0
        summary["one_sided"] = False

    return summary


def error_rate_by_slice(frame, column, bins=None, labels=None,
                        min_count=MIN_SLICE_COUNT):
    """Error rate within each bucket of `column`.

    Args:
        frame (pd.DataFrame): Output of build_error_frame.
        column (str): Column to slice on.
        bins (int | sequence, optional): Bin edges (or a count) for a
            continuous column. Omit for categorical/low-cardinality columns.
        labels (sequence, optional): Bin labels, passed to pd.cut.
        min_count (int): Buckets smaller than this are marked unreliable.
    Returns:
        pd.DataFrame: indexed by bucket, with count, errors, error_rate, and
        reliable.
    """
    if column not in frame.columns:
        raise KeyError(f"Column {column!r} is not in the error frame.")

    values = frame[column]
    if bins is not None:
        values = pd.cut(values.astype("float64"), bins=bins, labels=labels,
                        include_lowest=True)

    grouped = frame.assign(_bucket=values).groupby("_bucket", observed=False)
    result = pd.DataFrame({
        "count": grouped.size(),
        "errors": grouped["correct"].apply(lambda s: int((~s).sum())),
    })
    result["error_rate"] = (result["errors"] / result["count"]).where(result["count"] > 0)
    result["reliable"] = result["count"] >= min_count
    return result


def worst_slices(frame, columns=None, min_count=MIN_SLICE_COUNT, top_n=5):
    """The buckets with the highest error rate across several columns.

    Only buckets meeting min_count are ranked, so a three-row bucket at 100%
    error cannot masquerade as the model's biggest weakness.

    Returns:
        pd.DataFrame: column, bucket, count, error_rate, sorted worst-first.
    """
    columns = columns or _default_slice_columns(frame)
    rows = []

    for column in columns:
        spec = SLICE_SPECS.get(column, {})
        try:
            table = error_rate_by_slice(frame, column, bins=spec.get("bins"),
                                        labels=spec.get("labels"),
                                        min_count=min_count)
        except (KeyError, ValueError, TypeError):
            continue
        for bucket, row in table.iterrows():
            if not row["reliable"]:
                continue
            rows.append({
                "column": column,
                "bucket": str(bucket),
                "count": int(row["count"]),
                "error_rate": float(row["error_rate"]),
            })

    if not rows:
        return pd.DataFrame(columns=["column", "bucket", "count", "error_rate"])

    result = pd.DataFrame(rows).sort_values("error_rate", ascending=False)
    return result.head(top_n).reset_index(drop=True)


# Binning for the continuous columns worth slicing on. Hour gets 4-hour bands
# so each bucket keeps enough rows to mean something.
SLICE_SPECS = {
    "hour": {"bins": [-0.5, 3.5, 7.5, 11.5, 15.5, 19.5, 23.5],
             "labels": ["00-03", "04-07", "08-11", "12-15", "16-19", "20-23"]},
    "word_count": {"bins": [0, 15, 30, 60, 120, np.inf],
                   "labels": ["<15", "15-30", "30-60", "60-120", "120+"]},
    "lexical_diversity": {"bins": 4},
    "question_rate": {"bins": 4},
}


def _default_slice_columns(frame):
    """Columns worth slicing on that are actually present."""
    candidates = ["hour", "day_of_week", "is_late_night", "is_weekend",
                  "word_count", "lexical_diversity", "question_rate", "y_true"]
    return [c for c in candidates if c in frame.columns]


def worst_errors(frame, n=10, text_column="clean_text"):
    """The most confidently wrong predictions.

    These are the instructive mistakes: a model that is 95% sure and wrong has
    learned something false, which is more diagnostic than a coin-flip miss.
    Read them -- they usually explain the error rate faster than any statistic.

    Returns:
        pd.DataFrame: The wrong rows sorted by confidence, highest first.
    """
    errors = frame[~frame["correct"]]
    if errors.empty:
        return errors

    if "confidence" in errors.columns:
        errors = errors.sort_values("confidence", ascending=False)

    columns = [c for c in (text_column, "y_true", "y_pred", "confidence",
                           "hour", "is_late_night", "error_type")
               if c in errors.columns]
    return errors.head(n)[columns]


def calibration_table(frame, bins=10):
    """Predicted confidence against observed accuracy, plus the ECE.

    A well-calibrated model that says 0.9 is right about 90% of the time. When
    it is right 60% of the time instead, the probabilities cannot be used as
    anything but a ranking, which rules out thresholding for precision.

    Returns:
        tuple[pd.DataFrame, float]: per-bin table and expected calibration
        error (0 is perfect; above ~0.1 is badly calibrated).
    """
    if "confidence" not in frame.columns:
        raise KeyError(
            "No 'confidence' column. Pass y_prob to build_error_frame to enable "
            "calibration analysis."
        )

    edges = np.linspace(0.5, 1.0, bins + 1)
    bucket = pd.cut(frame["confidence"], bins=edges, include_lowest=True)
    grouped = frame.groupby(bucket, observed=False)

    table = pd.DataFrame({
        "count": grouped.size(),
        "mean_confidence": grouped["confidence"].mean(),
        "accuracy": grouped["correct"].mean(),
    })
    table["gap"] = table["accuracy"] - table["mean_confidence"]

    total = table["count"].sum()
    if total == 0:
        return table, 0.0

    weights = table["count"] / total
    ece = float((weights * table["gap"].abs()).fillna(0.0).sum())
    return table, ece


def compare_predictions(frame_a, frame_b, name_a="model_a", name_b="model_b"):
    """Agreement between two models on the same rows, and who wins when they differ.

    Two models at the same accuracy can be right about entirely different
    examples. When they are, an ensemble is worth trying; when they agree on
    nearly everything, the more expensive one is not earning its cost.

    Returns:
        dict: agreement rate, the four agree/disagree cells, and -- where the
        two disagree -- how often each is the correct one.
    """
    if len(frame_a) != len(frame_b):
        raise ValueError(
            f"Frames cover different numbers of rows ({len(frame_a)} vs "
            f"{len(frame_b)}); they must be predictions over the same split."
        )
    if not np.array_equal(frame_a["y_true"].to_numpy(), frame_b["y_true"].to_numpy()):
        raise ValueError(
            "The two frames have different ground-truth labels, so they are not "
            "predictions over the same rows in the same order."
        )

    a_correct = frame_a["correct"].to_numpy()
    b_correct = frame_b["correct"].to_numpy()
    same_prediction = frame_a["y_pred"].to_numpy() == frame_b["y_pred"].to_numpy()

    n = len(frame_a)
    disagreements = int((~same_prediction).sum())

    result = {
        "num_samples": n,
        "agreement_rate": float(same_prediction.mean()) if n else 0.0,
        "both_correct": int((a_correct & b_correct).sum()),
        "both_wrong": int((~a_correct & ~b_correct).sum()),
        f"only_{name_a}_correct": int((a_correct & ~b_correct).sum()),
        f"only_{name_b}_correct": int((~a_correct & b_correct).sum()),
        "num_disagreements": disagreements,
    }

    if disagreements:
        mask = ~same_prediction
        result[f"{name_a}_wins_when_disagreeing"] = float(a_correct[mask].mean())
        result[f"{name_b}_wins_when_disagreeing"] = float(b_correct[mask].mean())

    # If one model is right on a meaningful set the other misses, they are
    # making different mistakes and combining them could actually help.
    exclusive = result[f"only_{name_a}_correct"] + result[f"only_{name_b}_correct"]
    result["complementary"] = bool(n and exclusive / n > 0.05)
    return result


def plot_error_rate_by_hour(frame, output_dir=FIGURE_DIR,
                            filename="error_rate_by_hour.png"):
    """Error rate across the UTC clock, with per-bucket sample counts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = error_rate_by_slice(frame, "hour")
    overall = 1.0 - frame["correct"].mean()

    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = ["#C44E52" if reliable else "#CCCCCC" for reliable in table["reliable"]]
    ax.bar(table.index.astype(float), table["error_rate"].fillna(0), color=colors)
    ax.axhline(overall, color="#4C72B0", linestyle="--", linewidth=1.5,
               label=f"Overall error rate ({overall:.2f})")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Error rate")
    ax.set_title("Error rate by posting hour (grey = too few samples to trust)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_calibration(frame, output_dir=FIGURE_DIR, bins=10,
                     filename="calibration.png"):
    """Reliability diagram: confidence against observed accuracy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table, ece = calibration_table(frame, bins=bins)
    valid = table.dropna(subset=["mean_confidence", "accuracy"])

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot([0.5, 1.0], [0.5, 1.0], color="#888", linestyle="--", linewidth=1,
            label="Perfect calibration")
    if not valid.empty:
        ax.plot(valid["mean_confidence"], valid["accuracy"], marker="o",
                color="#C44E52", linewidth=2, label="Observed")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Calibration (ECE = {ece:.3f})")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def print_report(frame, top_n=5):
    """Print the full analysis. Returns the pieces as a dict."""
    summary = error_summary(frame)

    print(f"\nError summary over {summary['num_samples']} predictions")
    print(f"  errors            {summary['num_errors']} "
          f"({summary['error_rate']:.1%})")
    print(f"  false positives   {summary['false_positive']}")
    print(f"  false negatives   {summary['false_negative']}")
    if summary["one_sided"]:
        direction = ("false positives" if summary["false_positive_share"] > 0.5
                     else "false negatives")
        print(f"  NOTE: {summary['error_skew']:.0%} of the errors run one way "
              f"({direction}). That is a threshold/bias problem rather than a\n"
              f"        general accuracy problem -- moving the decision threshold "
              f"will trade one for the other.")

    worst = worst_slices(frame, top_n=top_n)
    if not worst.empty:
        print(f"\nHardest slices (>= {MIN_SLICE_COUNT} samples):")
        for _, row in worst.iterrows():
            print(f"  {row['column']:<20} {row['bucket']:<12} "
                  f"n={row['count']:<5} error rate {row['error_rate']:.1%}")

        overall = summary["error_rate"]
        worst_row = worst.iloc[0]
        if overall > 0 and worst_row["error_rate"] > overall * 1.5:
            print(
                f"\n  {worst_row['column']}={worst_row['bucket']} fails at "
                f"{worst_row['error_rate']:.1%} against {overall:.1%} overall. "
                "Errors are\n  concentrated, not uniform -- the headline accuracy "
                "is hiding this slice."
            )

    calibration = None
    if "confidence" in frame.columns:
        table, ece = calibration_table(frame)
        calibration = {"ece": ece, "table": table}
        verdict = ("well calibrated" if ece < 0.05
                   else "usable" if ece < 0.1 else "poorly calibrated")
        print(f"\nCalibration: ECE {ece:.3f} ({verdict})")
        if ece >= 0.1:
            print("  The probabilities do not mean what they say. Use them to "
                  "rank, not\n  as thresholds for a precision target.")

    errors = worst_errors(frame, n=min(top_n, 5))
    if not errors.empty and "clean_text" in errors.columns:
        print("\nMost confident mistakes:")
        for _, row in errors.iterrows():
            confidence = f"{row['confidence']:.2f}" if "confidence" in row else "n/a"
            text = str(row["clean_text"])[:88]
            print(f"  [true {row['y_true']} pred {row['y_pred']} conf {confidence}] {text}")

    return {"summary": summary, "worst_slices": worst, "calibration": calibration}


def report_to_json(report):
    """Flatten print_report's output into something json.dump can take.

    The report carries DataFrames, which exist to be printed. The report
    generator needs numbers, and a results document assembled from scraped
    stdout is a results document that breaks the next time a log line moves.
    """
    payload = {"summary": report.get("summary", {})}

    calibration = report.get("calibration")
    if calibration:
        payload["ece"] = float(calibration["ece"])
        table = calibration.get("table")
        if table is not None:
            payload["calibration_bins"] = [
                {"bin": str(index), "count": int(row["count"]),
                 "mean_confidence": _finite(row["mean_confidence"]),
                 "accuracy": _finite(row["accuracy"]),
                 "gap": _finite(row["gap"])}
                for index, row in table.iterrows()
            ]

    worst = report.get("worst_slices")
    if worst is not None and not worst.empty:
        payload["worst_slices"] = [
            {"column": str(row["column"]), "bucket": str(row["bucket"]),
             "count": int(row["count"]), "error_rate": float(row["error_rate"])}
            for _, row in worst.iterrows()
        ]

    if report.get("figures"):
        payload["figures"] = list(report["figures"])
    return payload


def _finite(value):
    """NaN is not valid JSON, and an empty calibration bin produces one."""
    value = float(value)
    return None if np.isnan(value) or np.isinf(value) else value


def analyse_dataset(dataset_path, label_strategy="median", split_strategy="random",
                    seed=42, val_split=0.2, output_dir=FIGURE_DIR, max_rows=0,
                    json_out=None, column_map=None):
    """Fit the TF-IDF baseline and run the full error analysis on its validation split.

    Uses the baseline rather than the neural model because it trains in seconds
    and needs no network, which makes error analysis something you actually run
    rather than something you mean to get around to.

    Returns:
        tuple[pd.DataFrame, dict]: the error frame and the report pieces.
    """
    from models.tfidf_baseline import TfidfBaseline

    print(f"Loading {dataset_path}...")
    data = clean_dataset(read_dataset(dataset_path, column_map))
    data = add_temporal_features(data, TIMESTAMP_COLUMN)
    data = add_marker_columns(data)

    if split_strategy == "temporal":
        data = data.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    if max_rows and max_rows < len(data):
        data = data.head(max_rows).reset_index(drop=True)

    labels, label_summary = build_labels(data, strategy=label_strategy)
    print(f"  Labels: {label_summary}")

    features = temporal_feature_matrix(data, TEMPORAL_FEATURES)
    texts = data["clean_text"].tolist()
    train_idx, val_idx = split_indices(len(data), val_split, split_strategy, seed)

    print(f"  Fitting baseline on {len(train_idx)} rows...")
    model = TfidfBaseline(seed=seed)
    model.fit([texts[i] for i in train_idx], labels[train_idx], features[train_idx])

    val_texts = [texts[i] for i in val_idx]
    predictions = model.predict(val_texts, features[val_idx])
    probabilities = model.predict_proba(val_texts, features[val_idx])

    frame = build_error_frame(data.iloc[val_idx], labels[val_idx], predictions,
                              probabilities)
    report = print_report(frame)

    figures = [
        plot_error_rate_by_hour(frame, output_dir),
        plot_calibration(frame, output_dir),
    ]
    print("\nFigures written:")
    for path in figures:
        print(f"  {path}")

    report["figures"] = [str(p) for p in figures]

    if json_out:
        json_out = Path(json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report_to_json(report), indent=2) + "\n")
        print(f"  {json_out}")

    return frame, report


def compare_feature_sets(dataset_path, label_strategy="median",
                         split_strategy="random", seed=42, val_split=0.2,
                         max_rows=0, json_out=None, column_map=None):
    """Error analysis of the temporal ablation: same model, with and without.

    The benchmark says the temporal features raise accuracy. It does not say
    whether they fix errors or merely move them around -- a model can gain two
    points overall while getting worse on the slice you care about. This
    compares the two runs example by example.

    Returns:
        dict: the comparison, plus each variant's error summary.
    """
    from models.tfidf_baseline import TfidfBaseline

    print(f"Loading {dataset_path}...")
    data = clean_dataset(read_dataset(dataset_path, column_map))
    data = add_temporal_features(data, TIMESTAMP_COLUMN)
    data = add_marker_columns(data)

    if split_strategy == "temporal":
        data = data.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    if max_rows and max_rows < len(data):
        data = data.head(max_rows).reset_index(drop=True)

    labels, _ = build_labels(data, strategy=label_strategy)
    features = temporal_feature_matrix(data, TEMPORAL_FEATURES)
    texts = data["clean_text"].tolist()
    train_idx, val_idx = split_indices(len(data), val_split, split_strategy, seed)
    val_texts = [texts[i] for i in val_idx]

    frames = {}
    for name, use_temporal in (("text_only", False), ("text_temporal", True)):
        model = TfidfBaseline(use_temporal_features=use_temporal, seed=seed)
        train_features = features[train_idx] if use_temporal else None
        model.fit([texts[i] for i in train_idx], labels[train_idx], train_features)

        eval_features = features[val_idx] if use_temporal else None
        frames[name] = build_error_frame(
            data.iloc[val_idx], labels[val_idx],
            model.predict(val_texts, eval_features),
            model.predict_proba(val_texts, eval_features),
        )

    comparison = compare_predictions(frames["text_only"], frames["text_temporal"],
                                     "text_only", "text_temporal")

    print(f"\nTemporal ablation, example by example ({comparison['num_samples']} rows)")
    print(f"  agreement rate           {comparison['agreement_rate']:.1%}")
    print(f"  both correct             {comparison['both_correct']}")
    print(f"  both wrong               {comparison['both_wrong']}")
    print(f"  only text-only correct   {comparison['only_text_only_correct']}")
    print(f"  only text+temporal right {comparison['only_text_temporal_correct']}")

    gained = comparison["only_text_temporal_correct"]
    lost = comparison["only_text_only_correct"]
    if gained > lost:
        print(f"\n  Adding the timestamps fixes {gained} predictions and breaks "
              f"{lost}, a net\n  gain of {gained - lost}. They are adding "
              "information rather than reshuffling it.")
    elif lost and gained <= lost:
        print(f"\n  Adding the timestamps fixes {gained} predictions and breaks "
              f"{lost}. The\n  headline gain is not coming from the temporal "
              "features on this split.")

    summaries = {name: error_summary(frame) for name, frame in frames.items()}
    for name, summary in summaries.items():
        print(f"\n  {name}: error rate {summary['error_rate']:.1%} "
              f"(FP {summary['false_positive']}, FN {summary['false_negative']})")

    result = {"comparison": comparison, "summaries": summaries, "frames": frames}

    if json_out:
        json_out = Path(json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        # `frames` holds DataFrames for the caller; only the counts persist.
        json_out.write_text(json.dumps(
            {"comparison": comparison, "summaries": summaries}, indent=2) + "\n")
        print(f"\n  {json_out}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Error analysis: which slices the model fails on, and why."
    )
    parser.add_argument("--dataset", required=True, help="Path to the CSV.")
    parser.add_argument("--output-dir", default=str(FIGURE_DIR))
    parser.add_argument("--label-strategy", default="median",
                        choices=["median", "positive", "threshold"])
    parser.add_argument("--split-strategy", default="random",
                        choices=["temporal", "random"])
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-feature-sets", action="store_true",
                        help="Compare the text-only and text+temporal runs "
                             "example by example instead of analysing one.")
    parser.add_argument("--column-map", default="",
                        help="canonical=source pairs, e.g. 'selftext=body'.")
    parser.add_argument("--json-out", default=None,
                        help="Also write the numbers as JSON, for analysis.report. "
                             "Defaults to outputs/error_analysis.json (or "
                             "feature_set_comparison.json with --compare-feature-sets).")
    args = parser.parse_args()

    if args.compare_feature_sets:
        json_out = args.json_out or OUTPUT_DIR / "feature_set_comparison.json"
        compare_feature_sets(args.dataset, label_strategy=args.label_strategy,
                             split_strategy=args.split_strategy, seed=args.seed,
                             max_rows=args.max_rows, json_out=json_out,
                             column_map=args.column_map)
    else:
        json_out = args.json_out or OUTPUT_DIR / "error_analysis.json"
        analyse_dataset(args.dataset, label_strategy=args.label_strategy,
                        split_strategy=args.split_strategy, seed=args.seed,
                        output_dir=args.output_dir, max_rows=args.max_rows,
                        json_out=json_out, column_map=args.column_map)


if __name__ == "__main__":
    main()
