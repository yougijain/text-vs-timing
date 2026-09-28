"""Turn run artefacts into RESULTS.md. Nothing in the report is hand-typed.

A results table copied by hand into a README goes stale on the next run and
nobody notices, because a number that is slightly wrong looks exactly like a
number that is right. This reads `benchmark.json`, the error-analysis output,
and the dataset's provenance sidecar, and writes the whole document.

It also decides what happened rather than leaving that to the writer. "BERT
won" is the least interesting outcome and the easiest to claim by eyeballing a
table, so the four readings the project cares about are selected by a function
with a stated rule:

  * BERT beats TF-IDF clearly    -> the text carries structure a bag of words
                                    cannot reach.
  * They tie                     -> the transformer is not earning its compute
                                    on short text. Worth knowing before anyone
                                    deploys one.
  * Temporal helps both          -> the timing signal is real and
                                    architecture-independent.
  * Temporal helps only one      -> the fusion head is doing the work, not the
                                    clock.

Where the per-example predictions exist, the temporal verdict uses McNemar's
test on the fixed-versus-broken counts, which is the correct test for two
classifiers on the same rows. Where only aggregate accuracies exist -- the
model-vs-model comparison, since `benchmark.json` stores metrics and not
predictions -- the rule falls back to overlapping confidence intervals, and the
report says which rule it used.

    python -m analysis.report --output-dir outputs --out RESULTS.md
"""

import argparse
import json
import math
from pathlib import Path

from training.config import OUTPUT_DIR, PROJECT_ROOT

# Above this, the probabilities cannot be read as probabilities -- they still
# rank, but they cannot be thresholded for a precision target.
ECE_CEILING = 0.1


def accuracy_interval(accuracy, n_samples, z=1.96):
    """Normal-approximation confidence interval for an accuracy.

    Deliberately crude, and the report says so. It exists to stop a 0.3-point
    difference on 200 validation rows from being written up as a finding.
    """
    if not n_samples:
        return (0.0, 0.0)
    half = z * math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / n_samples)
    return (max(0.0, accuracy - half), min(1.0, accuracy + half))


def mcnemar(only_a_correct, only_b_correct):
    """McNemar's test on two classifiers scored over the same rows.

    The right test here, and not the obvious one. A two-sample test on the two
    accuracies throws away the pairing: the cases both models get right carry
    no information about which is better, and including them inflates the
    sample size. Only the disagreements count, and they are exactly the "fixed"
    and "broken" cells the feature-set comparison already produces.

    Uses the continuity-corrected chi-square form. Below ~25 discordant pairs
    that approximation is unreliable, so an exact binomial p-value is used
    instead.

    Args:
        only_a_correct (int): rows model A got right and B did not.
        only_b_correct (int): rows model B got right and A did not.
    Returns:
        dict: statistic, p_value, n_discordant, method, and significant (at
        0.05). p_value is None when there is nothing to test.
    """
    b, c = int(only_a_correct), int(only_b_correct)
    n = b + c

    if n == 0:
        return {"statistic": 0.0, "p_value": None, "n_discordant": 0,
                "method": "none (models never disagree)", "significant": False}

    if n < 25:
        # Exact two-sided binomial against p=0.5. Guarded by an import so the
        # module stays importable without scipy for the rendering paths.
        try:
            from scipy.stats import binomtest
            p_value = float(binomtest(min(b, c), n, 0.5).pvalue)
            method = "exact binomial (few discordant pairs)"
        except ImportError:  # pragma: no cover - scipy ships with sklearn
            p_value = None
            method = "exact binomial unavailable (scipy missing)"
        return {"statistic": float(min(b, c)), "p_value": p_value,
                "n_discordant": n, "method": method,
                "significant": bool(p_value is not None and p_value < 0.05)}

    statistic = (abs(b - c) - 1) ** 2 / n
    try:
        from scipy.stats import chi2
        p_value = float(chi2.sf(statistic, df=1))
    except ImportError:  # pragma: no cover
        p_value = None
    return {"statistic": float(statistic), "p_value": p_value,
            "n_discordant": n, "method": "chi-square with continuity correction",
            "significant": bool(p_value is not None and p_value < 0.05)}


def _row_key(row):
    return (row["model"], row["features"])


def _accuracy(rows, model_prefix, features):
    for row in rows:
        if row["model"].startswith(model_prefix) and row["features"] == features:
            return row["metrics"]
    return None


def temporal_lift(rows, model_prefix):
    """Accuracy gained by adding the timestamp, for one architecture.

    Returns None when the grid does not contain both cells for that model --
    a half-run grid must not be written up as an ablation.
    """
    text_only = _accuracy(rows, model_prefix, "text only")
    both = _accuracy(rows, model_prefix, "text + temporal")
    if text_only is None or both is None:
        return None
    return {
        "model": model_prefix,
        "text_only": text_only["accuracy"],
        "text_temporal": both["accuracy"],
        "delta": both["accuracy"] - text_only["accuracy"],
        "n_samples": both.get("num_samples", 0),
    }


def classify_outcome(rows, comparison=None):
    """Decide which of the four readings the numbers support.

    Args:
        rows (list[dict]): benchmark.json's "rows".
        comparison (dict | None): compare_predictions output for the temporal
            ablation, when per-example predictions were available. Its presence
            upgrades the temporal verdict from interval overlap to McNemar.
    Returns:
        dict: per-question verdicts plus a `headline` sentence.
    """
    if isinstance(rows, dict):
        # Handed the whole benchmark.json instead of its "rows" list. Iterating
        # a dict yields its keys, so this would otherwise fail several frames
        # later with "string indices must be integers".
        raise TypeError(
            "classify_outcome expects the list of benchmark rows, not the "
            "whole benchmark document. Pass benchmark['rows']."
        )

    models = []
    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])

    lifts = {}
    for model in models:
        lift = temporal_lift(rows, model)
        if lift is not None:
            lifts[model] = lift

    verdicts = {"models": models, "temporal_lifts": lifts, "notes": []}

    # --- Does the timestamp help? ---
    if comparison:
        gained = comparison.get("only_text+temporal_correct")
        lost = comparison.get("only_text-only_correct")
        if gained is None or lost is None:
            gained, lost = _infer_comparison_cells(comparison)
        if gained is not None and lost is not None:
            test = mcnemar(lost, gained)
            verdicts["temporal_test"] = {
                "fixed": int(gained), "broken": int(lost), **test,
            }
            verdicts["temporal_verdict"] = (
                "helps" if test["significant"] and gained > lost
                else "hurts" if test["significant"] and lost > gained
                else "inconclusive"
            )
            verdicts["notes"].append(
                "Temporal verdict from McNemar's test on the paired "
                "predictions, not on the accuracy difference."
            )

    if "temporal_verdict" not in verdicts and lifts:
        helped = []
        for model, lift in lifts.items():
            lo_text, hi_text = accuracy_interval(lift["text_only"], lift["n_samples"])
            lo_both, _ = accuracy_interval(lift["text_temporal"], lift["n_samples"])
            helped.append(lo_both > hi_text)
        if all(helped) and len(helped) > 1:
            verdicts["temporal_verdict"] = "helps every architecture"
        elif any(helped):
            verdicts["temporal_verdict"] = "helps some architectures"
        else:
            verdicts["temporal_verdict"] = "inconclusive"
        verdicts["notes"].append(
            "Temporal verdict from non-overlapping confidence intervals. "
            "A McNemar test on the paired predictions would be sharper -- run "
            "`error_analysis --compare-feature-sets` to get it."
        )

    # Helps one architecture but not the other: the fusion head, not the clock.
    if len(lifts) > 1:
        deltas = {m: lift["delta"] for m, lift in lifts.items()}
        positive = [m for m, d in deltas.items() if d > 0.01]
        if positive and len(positive) < len(deltas):
            verdicts["architecture_dependence"] = (
                f"The timestamp helps {', '.join(positive)} but not "
                f"{', '.join(m for m in deltas if m not in positive)}. That "
                "points at the fusion head rather than a signal in the clock."
            )
        elif len(positive) == len(deltas) and deltas:
            verdicts["architecture_dependence"] = (
                "The timestamp helps every architecture tried, which is what "
                "you would expect from a real signal rather than one "
                "architecture's inductive bias."
            )

    # --- Is the transformer earning its compute? ---
    best = {}
    for row in rows:
        name = row["model"]
        if name not in best or row["metrics"]["accuracy"] > best[name]["accuracy"]:
            best[name] = row["metrics"]

    bert = next((n for n in best if n.startswith("BERT")), None)
    tfidf = next((n for n in best if n.startswith("TF-IDF")), None)

    if bert and tfidf:
        bert_acc = best[bert]["accuracy"]
        tfidf_acc = best[tfidf]["accuracy"]
        n = best[bert].get("num_samples", 0)
        bert_lo, bert_hi = accuracy_interval(bert_acc, n)
        tfidf_lo, tfidf_hi = accuracy_interval(tfidf_acc, n)

        if bert_lo > tfidf_hi:
            verdict = "bert_wins"
            sentence = (
                f"{bert} beats {tfidf} ({bert_acc:.4f} vs {tfidf_acc:.4f}), "
                "by more than the sampling error on this validation set. The "
                "text carries structure a bag of words cannot reach."
            )
        elif tfidf_lo > bert_hi:
            verdict = "tfidf_wins"
            sentence = (
                f"{tfidf} beats {bert} ({tfidf_acc:.4f} vs {bert_acc:.4f}). On "
                "short text with a few thousand rows this is a common and "
                "legitimate outcome, not a bug to tune away."
            )
        else:
            verdict = "tie"
            sentence = (
                f"{bert} and {tfidf} are within sampling error of each other "
                f"({bert_acc:.4f} vs {tfidf_acc:.4f}). The transformer is not "
                "earning its compute on this data, which is worth knowing "
                "before anyone deploys one."
            )
        verdicts["model_verdict"] = verdict
        verdicts["headline"] = sentence
        verdicts["notes"].append(
            "Model-vs-model verdict from overlapping normal-approximation "
            "intervals. benchmark.json stores metrics rather than predictions, "
            "so a paired test is not available for this comparison."
        )

    if "tiny" in " ".join(models).lower():
        verdicts["notes"].insert(0, (
            "A BERT row here is a randomly initialised miniature model with no "
            "pretrained weights. It is a plumbing check, not a competitor, and "
            "no model-vs-model conclusion below is evidence about real BERT."
        ))

    return verdicts


def _infer_comparison_cells(comparison):
    """Pull the two 'only X correct' cells out of a compare_predictions dict.

    compare_predictions names those keys after the models it was given, so the
    exact key depends on the caller. Rather than hardcode one spelling, find
    them by shape.
    """
    only_keys = sorted(k for k in comparison
                       if k.startswith("only_") and k.endswith("_correct"))
    if len(only_keys) != 2:
        return None, None
    # "text+temporal" sorts after "text-only", and the caller in
    # compare_feature_sets passes them in that order.
    first, second = only_keys
    if "temporal" in first and "temporal" not in second:
        return comparison[first], comparison[second]
    return comparison[second], comparison[first]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _fmt(value, spec=".4f", default="n/a"):
    return default if value is None else format(value, spec)


def render_grid(rows):
    """The 2x2 (or 3x2) grid as a markdown table, with lift on every cell."""
    if not rows:
        return "_No benchmark rows found._\n"

    baseline = rows[0]["metrics"].get("majority_baseline", 0.0)
    best_accuracy = max(r["metrics"]["accuracy"] for r in rows)

    lines = [
        f"Majority-class baseline: **{baseline:.4f}** "
        f"({rows[0]['metrics'].get('num_samples', 0)} validation rows)",
        "",
        "| Model | Features | Accuracy | Macro F1 | Lift over baseline |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        m = row["metrics"]
        accuracy = f"{m['accuracy']:.4f}"
        if m["accuracy"] == best_accuracy:
            accuracy = f"**{accuracy}**"
        lines.append(
            f"| {row['model']} | {row['features']} | {accuracy} | "
            f"{m['macro_f1']:.4f} | {m['lift_over_baseline']:+.4f} |"
        )
    lines.append("")
    lines.append(
        "Lift is accuracy minus the majority-class baseline. **At or below "
        "zero means the model learned nothing**, whatever the accuracy says."
    )
    return "\n".join(lines) + "\n"


def render_ablation(verdicts):
    """Per-architecture temporal deltas, and the verdict with its test."""
    lifts = verdicts.get("temporal_lifts") or {}
    if not lifts:
        return "_No complete text-only / text+temporal pair in the grid._\n"

    lines = [
        "| Architecture | Text only | Text + temporal | Delta |",
        "|---|---|---|---|",
    ]
    for model, lift in lifts.items():
        lines.append(
            f"| {model} | {lift['text_only']:.4f} | "
            f"{lift['text_temporal']:.4f} | {lift['delta']:+.4f} |"
        )
    lines.append("")

    test = verdicts.get("temporal_test")
    if test:
        p = _fmt(test.get("p_value"), ".4g")
        lines += [
            f"Adding the timestamp **fixes {test['fixed']}** predictions and "
            f"**breaks {test['broken']}**, a net gain of "
            f"{test['fixed'] - test['broken']}.",
            "",
            f"McNemar's test on those {test['n_discordant']} discordant pairs: "
            f"statistic {test['statistic']:.4g}, p = {p} "
            f"({test['method']}).",
            "",
            "That is the right test for two classifiers over the same rows: "
            "the cases both get right carry no information about which is "
            "better, so only the disagreements count. A two-sample test on "
            "the two accuracies would ignore the pairing and overstate the "
            "sample size.",
        ]

    if verdicts.get("architecture_dependence"):
        lines += ["", verdicts["architecture_dependence"]]
    return "\n".join(lines) + "\n"


def render_calibration(error_analysis):
    """ECE and what it licenses you to do with the probabilities."""
    if not error_analysis:
        return "_Error analysis not run._\n"

    ece = error_analysis.get("ece")
    if ece is None:
        ece = (error_analysis.get("calibration") or {}).get("ece")
    if ece is None:
        return "_No calibration data in the error-analysis output._\n"

    summary = error_analysis.get("summary", {})
    lines = [f"Expected calibration error: **{ece:.4f}**", ""]

    if ece <= ECE_CEILING:
        lines.append(
            f"Below the {ECE_CEILING} line, so the predicted probabilities "
            "mean roughly what they say and can be thresholded for a precision "
            "target."
        )
    else:
        lines.append(
            f"Above the {ECE_CEILING} line. The probabilities can still rank "
            "posts, but they cannot be thresholded for a precision target -- a "
            "0.9 from this model is not a 90% chance."
        )

    if summary:
        lines += [
            "",
            f"Error rate {summary.get('error_rate', 0):.1%} over "
            f"{summary.get('num_samples', 0)} rows "
            f"(FP {summary.get('false_positive', 0)}, "
            f"FN {summary.get('false_negative', 0)}).",
        ]
        if summary.get("one_sided"):
            lines.append(
                "Mistakes run almost entirely one direction, which is a "
                "threshold problem rather than an accuracy problem. Different "
                "fix."
            )
    return "\n".join(lines) + "\n"


def render_provenance(provenance, dataset_path=None):
    if not provenance:
        return (
            f"Dataset: `{dataset_path or 'unknown'}` — no provenance sidecar "
            "found. Fetch with `python -m data.fetch_dataset` to get one.\n"
        )

    lines = [
        "| | |",
        "|---|---|",
        f"| Source | {provenance.get('source', 'unknown')} |",
        f"| Site | `{provenance.get('site', 'n/a')}` |",
        f"| Licence | {provenance.get('licence', 'unknown')} |",
        f"| Rows | {provenance.get('rows', 'n/a')} |",
    ]
    if provenance.get("first_post_utc"):
        lines.append(
            f"| Span (UTC) | {provenance['first_post_utc'][:10]} to "
            f"{provenance['last_post_utc'][:10]} |"
        )
    lines.append(f"| Fetched | {provenance.get('fetched_at_utc', 'n/a')[:19]} |")
    if provenance.get("attribution"):
        lines += ["", f"> {provenance['attribution']}"]
    return "\n".join(lines) + "\n"


UTC_CAVEAT = """Every temporal feature in this run is **UTC**, and that bounds what the result
means. `is_late_night` is "posted in the 00:00–05:00 UTC band", not "posted in
the small hours" — without a per-author timezone, a poster's local clock cannot
be recovered. Someone in California writing at 2am local lands at 09:00 UTC and
is not flagged; someone in Berlin writing at 2am local is.

The feature is still predictive, because UTC hour correlates with local hour and
with how many people are awake to vote. It is not evidence about anyone's sleep
or circadian rhythm, and it must not be written up as such.
"""


SYNTHETIC_BANNER = """> ## Not a result
>
> This document was generated from the synthetic sample corpus, which has a
> deliberately planted time-of-day signal. Every number below is a check that
> the pipeline works end to end. None of it says anything about real posts.
>
> Fetch a real corpus and re-run:
>
> ```
> python -m data.fetch_dataset --site stackoverflow --rows 20000 --out datasets/posts.csv
> python run_experiment.py --dataset datasets/posts.csv --epochs 3
> ```
"""


def render_report(benchmark, error_analysis=None, comparison=None,
                  provenance=None, dataset_path=None, figures=None,
                  synthetic=False):
    """Assemble the whole document. Returns markdown.

    `synthetic` puts a banner at the top. A results document is exactly the
    thing someone skims, and a table of numbers from generated data is
    indistinguishable from a table of real ones at a glance -- so the
    disclaimer goes above the numbers, not in a footnote under them.
    """
    rows = benchmark.get("rows", []) if benchmark else []
    verdicts = classify_outcome(rows, comparison)

    parts = [
        "# Results",
        "",
        "_Generated by `python -m analysis.report`. Do not edit by hand — "
        "re-run it._",
        "",
    ]
    if synthetic:
        parts += [SYNTHETIC_BANNER, ""]
    parts += ["## What happened", ""]

    headline = verdicts.get("headline")
    parts.append(headline if headline else
                 "_Not enough of the grid was run to compare architectures._")
    parts.append("")

    if verdicts.get("temporal_verdict"):
        parts += [
            f"The timestamp: **{verdicts['temporal_verdict']}**.",
            "",
        ]

    for note in verdicts.get("notes", []):
        parts.append(f"> {note}")
        parts.append("")

    parts += ["## Dataset", "", render_provenance(provenance, dataset_path), ""]
    parts += ["## The grid", "", render_grid(rows), ""]
    parts += ["## Does the timestamp add anything?", "",
              render_ablation(verdicts), ""]
    parts += ["## Calibration and errors", "",
              render_calibration(error_analysis), ""]

    if figures:
        parts += ["## Figures", ""]
        parts += [f"- `{figure}`" for figure in figures]
        parts.append("")

    parts += ["## The UTC caveat", "", UTC_CAVEAT]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Artefact loading and CLI
# --------------------------------------------------------------------------

def _repo_relative(path):
    """Path relative to the repo root when it sits inside it, else absolute.

    An --output-dir pointed somewhere else entirely is a legitimate thing to
    do, and it should produce a usable link rather than a crash.
    """
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"  WARNING: {path} is not valid JSON; skipping it.")
        return None


def load_artifacts(output_dir=OUTPUT_DIR, dataset_path=None):
    """Collect everything the report can use. Missing pieces are simply absent."""
    # Resolve first: --output-dir is usually given relative to the cwd, and a
    # relative path cannot be made relative to PROJECT_ROOT below.
    output_dir = Path(output_dir).resolve()

    provenance = None
    if dataset_path:
        from data.fetch_dataset import provenance_path
        provenance = _load_json(provenance_path(dataset_path))

    figures = []
    figure_dir = output_dir / "figures"
    if figure_dir.exists():
        figures = sorted(_repo_relative(p) for p in figure_dir.glob("*.png"))

    return {
        "benchmark": _load_json(output_dir / "benchmark.json"),
        "error_analysis": _load_json(output_dir / "error_analysis.json"),
        "comparison": _load_json(output_dir / "feature_set_comparison.json"),
        "provenance": provenance,
        "dataset_path": dataset_path,
        "figures": figures,
    }


def write_report(path, artifacts, synthetic=False):
    """Render and write RESULTS.md. Returns the path."""
    comparison = artifacts.get("comparison")
    if comparison and "comparison" in comparison:
        comparison = comparison["comparison"]

    markdown = render_report(
        benchmark=artifacts.get("benchmark") or {},
        error_analysis=artifacts.get("error_analysis"),
        comparison=comparison,
        provenance=artifacts.get("provenance"),
        dataset_path=artifacts.get("dataset_path"),
        figures=artifacts.get("figures"),
        synthetic=synthetic or artifacts.get("synthetic", False),
    )
    path = Path(path)
    path.write_text(markdown)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate RESULTS.md from the run artefacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Where benchmark.json and the figures live.")
    parser.add_argument("--dataset", default=None,
                        help="Dataset CSV, so its provenance sidecar is quoted.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "RESULTS.md"))
    parser.add_argument("--synthetic", action="store_true",
                        help="Stamp the document as generated from sample data.")
    args = parser.parse_args(argv)

    artifacts = load_artifacts(args.output_dir, args.dataset)
    if not artifacts["benchmark"]:
        print(f"No benchmark.json under {args.output_dir}. Run benchmark.py first.")
        return None

    path = write_report(args.out, artifacts, synthetic=args.synthetic)
    print(f"Wrote {path}")
    for name in ("error_analysis", "comparison", "provenance"):
        if not artifacts.get(name):
            print(f"  (no {name} artefact -- that section is a placeholder)")
    return path


if __name__ == "__main__":
    main()
