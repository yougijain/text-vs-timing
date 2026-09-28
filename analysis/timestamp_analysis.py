"""Temporal exploratory analysis: when posts are written, and whether it matters.

This is the descriptive half of the "temporal" side of the project. It answers,
before any model is trained: does posting hour actually carry signal, or are the
temporal features noise? If the engagement rate is flat across the clock, the
temporal branch has nothing to learn and the ablation will show no gap.
"""

import argparse
from pathlib import Path

import numpy as np

# Agg backend so plots render on a headless box. The old main.py called
# plt.show(), which blocks forever over SSH or in CI and saves nothing.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.preprocess import build_labels, clean_dataset  # noqa: E402
from data.schema import read_dataset  # noqa: E402
from training.config import FIGURE_DIR, TIMESTAMP_COLUMN  # noqa: E402
from utils.time_utils import (  # noqa: E402
    LATE_NIGHT_END,
    LATE_NIGHT_START,
    add_temporal_features,
    hourly_distribution,
    late_night_share,
)

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def summarize_temporal(data, labels=None):
    """Descriptive statistics over the timestamp column.

    Args:
        data (pd.DataFrame): Dataset that has been through add_temporal_features.
        labels (array-like, optional): Binary labels, for engagement-by-hour.
    Returns:
        dict: Summary numbers, JSON-serialisable.
    """
    hours = data["hour"].dropna().astype(int)
    summary = {
        "num_posts": int(len(data)),
        "date_range": [
            str(data[TIMESTAMP_COLUMN].min()),
            str(data[TIMESTAMP_COLUMN].max()),
        ],
        "peak_hour": int(hours.value_counts().idxmax()) if len(hours) else None,
        "quietest_hour": int(hours.value_counts().idxmin()) if len(hours) else None,
        "late_night_share": round(late_night_share(data), 4),
        "weekend_share": round(float(data["is_weekend"].mean()), 4),
    }

    if labels is not None:
        frame = data.assign(_label=np.asarray(labels))
        by_hour = frame.groupby("hour", observed=True)["_label"].mean()
        summary["positive_rate_overall"] = round(float(frame["_label"].mean()), 4)
        summary["positive_rate_late_night"] = round(
            float(frame.loc[frame["is_late_night"] == 1, "_label"].mean()), 4
        ) if (frame["is_late_night"] == 1).any() else None
        summary["positive_rate_daytime"] = round(
            float(frame.loc[frame["is_late_night"] == 0, "_label"].mean()), 4
        ) if (frame["is_late_night"] == 0).any() else None
        summary["hour_spread"] = round(float(by_hour.max() - by_hour.min()), 4)

    return summary


def plot_hourly_activity(data, output_dir=FIGURE_DIR, labels=None):
    """Save a posts-per-hour chart, with engagement overlaid when labels exist."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = hourly_distribution(data, TIMESTAMP_COLUMN, normalize=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(counts.index, counts.to_numpy(), color="#4C72B0", label="Posts")
    ax.axvspan(LATE_NIGHT_START - 0.5, LATE_NIGHT_END - 0.5, color="#2b2b40",
               alpha=0.15, label=f"Late night ({LATE_NIGHT_START:02d}-{LATE_NIGHT_END:02d})")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Number of posts")
    ax.set_title("Posting activity by hour")
    ax.set_xticks(range(0, 24, 2))

    if labels is not None:
        frame = data.assign(_label=np.asarray(labels))
        rate = frame.groupby("hour", observed=True)["_label"].mean().reindex(range(24))
        twin = ax.twinx()
        twin.plot(rate.index, rate.to_numpy(), color="#C44E52", marker="o",
                  linewidth=2, label="High-engagement rate")
        twin.set_ylabel("Share of high-engagement posts")
        twin.set_ylim(0, 1)
        lines, labs = ax.get_legend_handles_labels()
        l2, lab2 = twin.get_legend_handles_labels()
        ax.legend(lines + l2, labs + lab2, loc="upper left", fontsize=8)
    else:
        ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    path = output_dir / "hourly_activity.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_weekly_heatmap(data, output_dir=FIGURE_DIR):
    """Save a day-of-week x hour-of-day heatmap of posting volume."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pivot = (
        data.dropna(subset=["hour", "day_of_week"])
        .astype({"hour": int, "day_of_week": int})
        .pivot_table(index="day_of_week", columns="hour", values=TIMESTAMP_COLUMN,
                     aggfunc="count")
        .reindex(index=range(7), columns=range(24))
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(11, 3.8))
    mesh = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma", origin="lower")
    ax.set_yticks(range(7), DAY_NAMES)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_title("Posting volume by weekday and hour")
    fig.colorbar(mesh, ax=ax, label="Posts")
    fig.tight_layout()

    path = output_dir / "weekly_heatmap.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_loss_curve(batch_losses, output_dir=FIGURE_DIR, window=None):
    """Save the training loss trend with a moving average overlaid.

    `window=None` sizes the smoothing to the series, so a short run does not ask
    for a window wider than the data it has.
    """
    from utils.loss_utils import moving_average, smooth_for_plot

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(batch_losses, color="#B0B0C0", linewidth=0.8, label="Batch loss")

    if window is None:
        smoothed, window = smooth_for_plot(batch_losses)
    else:
        smoothed = moving_average(batch_losses, window_size=window)
    if len(smoothed):
        offset = len(batch_losses) - len(smoothed)
        ax.plot(range(offset, len(batch_losses)), smoothed, color="#C44E52",
                linewidth=2, label=f"Moving average ({window})")

    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "loss_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run(dataset_path, output_dir=FIGURE_DIR, label_strategy="median",
        column_map=None):
    """Load a dataset, print the temporal summary, and write the figures."""
    print(f"Loading {dataset_path}...")
    data = read_dataset(dataset_path, column_map)
    data = clean_dataset(data)
    data = add_temporal_features(data, TIMESTAMP_COLUMN)
    labels, label_summary = build_labels(data, strategy=label_strategy)

    summary = summarize_temporal(data, labels)
    print("\nTemporal summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nLabel summary: {label_summary}")

    figures = [
        plot_hourly_activity(data, output_dir, labels),
        plot_weekly_heatmap(data, output_dir),
    ]
    print("\nFigures written:")
    for path in figures:
        print(f"  {path}")

    if summary.get("hour_spread") is not None and summary["hour_spread"] < 0.05:
        print(
            "\nNOTE: engagement barely varies across the clock "
            f"(spread {summary['hour_spread']:.3f}). The temporal features have "
            "little to contribute on this dataset -- expect the ablation to show "
            "no meaningful gap."
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Temporal analysis of the post dataset.")
    parser.add_argument("--dataset", required=True, help="Path to the CSV.")
    parser.add_argument("--output-dir", default=str(FIGURE_DIR), help="Figure output dir.")
    parser.add_argument("--label-strategy", default="median",
                        choices=["median", "positive", "threshold"])
    parser.add_argument("--column-map", default="",
                        help="canonical=source pairs, e.g. 'selftext=body'.")
    args = parser.parse_args()
    run(args.dataset, args.output_dir, args.label_strategy,
        column_map=args.column_map)


if __name__ == "__main__":
    main()
