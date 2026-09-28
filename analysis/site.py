"""Turn the run artefacts into a static page a non-specialist can read.

RESULTS.md is written for someone who already knows what a macro F1 is. This
writes the same numbers for someone who does not: the question in a sentence,
the answer in a sentence, and the table underneath for the reader who wants it.

Nothing here recomputes anything. The verdicts, the per-architecture lifts, the
McNemar test and the ECE all come from `analysis.report`. If the page could
derive its own numbers it could disagree with the document, and there would be
no way to tell which one was wrong.

    python -m analysis.site --output-dir outputs --dataset datasets/posts.csv

Writes `docs/index.html`, copies the figures in beside it, and drops a
`.nojekyll` so GitHub Pages serves the directory as-is. Point Pages at
`master` / `docs` once; after that every run republishes by committing.

Before the real run there is nothing to plot, and every section says "not run
yet" rather than rendering an empty table. A blank results page reads as a
broken build; a page that states what is missing reads as an honest one.
"""

import argparse
import html
import shutil
from datetime import datetime, timezone
from pathlib import Path

from analysis.report import (
    ECE_CEILING,
    accuracy_interval,
    classify_outcome,
    load_artifacts,
)
from training.config import OUTPUT_DIR, PROJECT_ROOT

QUESTION = "Does when a post goes up add anything over what it says?"

SITE_DIR = PROJECT_ROOT / "docs"

REPO_URL = "https://github.com/yougijain/text-vs-timing"

# Figures worth putting on the page, in reading order. The run writes more than
# this -- the rest stay in outputs/ for whoever wants them.
FEATURED_FIGURES = (
    ("calibration.png", "Predicted confidence against what actually happened. "
                        "On the diagonal means the probabilities mean what "
                        "they say."),
    ("error_rate_by_hour.png", "Where the model gets it wrong, by hour of day "
                               "(UTC)."),
    ("weekly_heatmap.png", "Posting volume by weekday and hour (UTC)."),
)


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def _esc(value):
    """Escape anything bound for the page.

    Provenance strings come off a remote API -- the site name and the licence
    attribution are whoever-typed-them, not ours. They are rendered as text.
    """
    return html.escape("" if value is None else str(value), quote=True)


def _pct(value, digits=1):
    """0.8230 -> '82.3%'. Percentages, because nobody reads 0.8230."""
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _points(value, digits=1):
    """A signed difference in percentage points, which is the honest unit for
    an accuracy delta. '+2.7 points' is not '+2.7%' and the two get confused."""
    if value is None:
        return "n/a"
    return f"{value * 100:+.{digits}f} points"


def _p_value(value):
    if value is None:
        return "n/a"
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


# --------------------------------------------------------------------------
# The plain-English layer
# --------------------------------------------------------------------------

def plain_answer(verdicts, rows):
    """Translate the verdict dict into sentences with no jargon in them.

    This is the only place the page says anything the document does not. It
    says the same thing in different words -- the mapping is fixed, and every
    number in it comes from the verdicts.

    Returns:
        (str, list[str]): a short answer for the hero, and supporting lines.
    """
    if not rows:
        return ("Not run yet.", [
            "The pipeline is built and tested; the corpus has not been fetched "
            "and the models have not been trained. When they are, the answer "
            "appears here and every number below fills in from the run.",
        ])

    lifts = verdicts.get("temporal_lifts") or {}
    verdict = verdicts.get("temporal_verdict")
    test = verdicts.get("temporal_test")
    lines = []

    if verdict in ("helps", "helps every architecture"):
        short = "Yes — the timestamp adds something."
    elif verdict == "helps some architectures":
        short = "Only for some models, which is a warning sign."
    elif verdict == "hurts":
        short = "No — adding the timestamp made it worse."
    elif verdict == "inconclusive":
        short = "Not measurably."
    else:
        short = "Ran, but not enough of the grid to answer."

    if lifts:
        best = max(lifts.values(), key=lambda lift: lift["delta"])
        lines.append(
            f"Adding the posting time to {_esc(best['model'])} moved accuracy "
            f"from {_pct(best['text_only'])} to {_pct(best['text_temporal'])}, "
            f"a change of {_points(best['delta'])} on "
            f"{best['n_samples']} held-out posts."
        )

    if test:
        # Fixed and broken are the interesting cells: a model can gain accuracy
        # while getting a lot of previously-right rows wrong, and the headline
        # number hides that entirely.
        lines.append(
            f"Post by post, the timestamp corrected {test['fixed']} posts the "
            f"text-only model got wrong and broke {test['broken']} it had got "
            f"right ({_p_value(test.get('p_value'))}, "
            f"{_esc(test.get('method', 'McNemar'))})."
        )

    if verdicts.get("architecture_dependence"):
        lines.append(_esc(verdicts["architecture_dependence"]))

    if verdicts.get("headline"):
        lines.append(_esc(verdicts["headline"]))

    if verdict == "inconclusive":
        lines.append(
            "An inconclusive result is a result. It means that on this corpus "
            "the posting time carries no usable signal once the text is "
            "accounted for, which is worth more than a tuned number that "
            "happened to go up."
        )

    return short, lines


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def _table(header_cells, body_rows, css_class=""):
    """A table in a scroll container.

    A four-column table cannot shrink below the width of its headers, so on a
    phone a bare one pushes the entire page sideways. The container scrolls
    instead, and nothing else moves.
    """
    head = ""
    if header_cells:
        cells = "".join(f"<th>{cell}</th>" for cell in header_cells)
        head = f"  <thead><tr>{cells}</tr></thead>\n"
    attribute = f' class="{css_class}"' if css_class else ""
    return (f'<div class="scroll">\n<table{attribute}>\n{head}'
            f"  <tbody>\n{body_rows}\n  </tbody>\n</table>\n</div>")


def _empty(markup):
    """A placeholder section. Takes markup, not text: these strings name
    commands, and a literal backtick on a rendered page looks like a typo."""
    return f'<p class="empty">{markup}</p>'


def render_hero(verdicts, rows, synthetic=False):
    short, lines = plain_answer(verdicts, rows)
    if synthetic and rows:
        short = f"{short} On generated sample data, so not a finding."
    body = "\n".join(f"<p>{line}</p>" for line in lines)
    tag = "Not run yet" if not rows else ("Sample data" if synthetic else "Result")
    tag_class = "tag warn" if (synthetic or not rows) else "tag"
    return f"""<header class="hero">
  <span class="{tag_class}">{_esc(tag)}</span>
  <h1>{_esc(QUESTION)}</h1>
  <p class="answer">{_esc(short)}</p>
  <div class="prose">
{body}
  </div>
</header>"""


def render_banner(synthetic, rows):
    """The disclaimer goes above the numbers, not in a footnote under them."""
    if not rows:
        return """<div class="banner">
  <strong>No real run yet.</strong> Everything below is the shape of the answer,
  not the answer. The page is generated from the run artefacts, so it fills in
  the moment a real corpus goes through the pipeline.
</div>"""
    if synthetic:
        return """<div class="banner">
  <strong>These numbers are from generated sample data</strong> with a planted
  time-of-day signal. They show the pipeline works end to end. They say nothing
  about real posts, and the large temporal gain below is the plant, not a
  finding.
</div>"""
    return ""


def render_grid(rows):
    """The grid as cards with bars, not as a table of four-decimal floats.

    Each bar is drawn against the majority-class baseline, because an accuracy
    without its baseline is unreadable: 82% is excellent on a balanced target
    and useless on one that is 85% negative.
    """
    if not rows:
        return _empty("The grid runs four cells: TF-IDF and BERT, each with and "
                      "without the timestamp. Nothing has been trained yet.")

    baseline = rows[0]["metrics"].get("majority_baseline", 0.0)
    best = max(row["metrics"]["accuracy"] for row in rows)
    n_samples = rows[0]["metrics"].get("num_samples", 0)

    cards = []
    for row in rows:
        metrics = row["metrics"]
        accuracy = metrics["accuracy"]
        lift = metrics.get("lift_over_baseline", accuracy - baseline)
        # A cell counts as having learned something only when the low end of
        # its confidence interval clears the baseline. A flat threshold would
        # colour a three-point gain on 226 rows green, and three points on 226
        # rows is noise -- exactly the claim this page exists not to make.
        low, _ = accuracy_interval(accuracy, metrics.get("num_samples", 0))
        state = "good" if low > baseline else "flat"
        cards.append(f"""  <article class="card {state}">
    <h3>{_esc(row['model'])}</h3>
    <p class="features">{_esc(row['features'])}</p>
    <p class="metric{' best' if accuracy == best else ''}">{_pct(accuracy)}</p>
    <div class="bar" role="img" aria-label="accuracy {_pct(accuracy)}, baseline {_pct(baseline)}">
      <span class="fill" style="width:{min(accuracy, 1.0) * 100:.1f}%"></span>
      <span class="marker" style="left:{min(baseline, 1.0) * 100:.1f}%"></span>
    </div>
    <p class="sub">{_points(lift)} over the baseline · macro F1 {metrics['macro_f1']:.3f}</p>
  </article>""")

    joined = "\n".join(cards)
    return f"""<p class="note">Majority-class baseline <strong>{_pct(baseline)}</strong>
on {n_samples} held-out posts — the notch on each bar. A bar that does not
reach it means the model learned nothing.</p>
<div class="grid">
{joined}
</div>"""


def render_ablation(verdicts):
    """Fixed versus broken, which is the question the accuracy delta hides."""
    test = verdicts.get("temporal_test")
    lifts = verdicts.get("temporal_lifts") or {}

    if not test and not lifts:
        return _empty("Needs both cells of at least one architecture.")

    parts = []
    if test:
        fixed, broken = test["fixed"], test["broken"]
        total = max(fixed + broken, 1)
        parts.append(f"""<div class="split" role="img"
     aria-label="{fixed} posts fixed, {broken} broken">
  <span class="fixed" style="width:{fixed / total * 100:.1f}%">{fixed} fixed</span>
  <span class="broken" style="width:{broken / total * 100:.1f}%">{broken} broken</span>
</div>
<p class="note">Of the {fixed + broken} posts where the two models disagreed,
the timestamp corrected {fixed} and broke {broken}
({_p_value(test.get('p_value'))}, {_esc(test.get('method', ''))}).
Accuracy alone would have shown a single number and hidden this split.</p>""")

    if lifts:
        rows = "\n".join(
            f"    <tr><td>{_esc(lift['model'])}</td><td>{_pct(lift['text_only'])}</td>"
            f"<td>{_pct(lift['text_temporal'])}</td><td>{_points(lift['delta'])}</td></tr>"
            for lift in lifts.values()
        )
        parts.append(_table(
            ("Architecture", "Text only", "Text + timestamp", "Difference"), rows))

    return "\n".join(parts)


def render_calibration(error_analysis):
    """Whether a 0.9 from this model means a 90% chance."""
    if not error_analysis:
        return _empty("Error analysis has not run.")

    ece = error_analysis.get("ece")
    if ece is None:
        ece = (error_analysis.get("calibration") or {}).get("ece")
    if ece is None:
        return _empty("No calibration data in the error-analysis output.")

    usable = ece <= ECE_CEILING
    reading = (
        "Below the {line} line, so the probabilities mean roughly what they say. "
        "You can set a threshold and get the precision you asked for."
        if usable else
        "Above the {line} line. The scores still rank posts sensibly, but a 0.9 "
        "from this model is not a 90% chance, so a threshold will not hold."
    ).format(line=ECE_CEILING)

    summary = error_analysis.get("summary") or {}
    extra = ""
    if summary:
        extra = (
            f"""<p class="note">Wrong on {_pct(summary.get('error_rate'))} of """
            f"""{summary.get('num_samples', 0)} posts — """
            f"""{summary.get('false_positive', 0)} called popular that were not, """
            f"""{summary.get('false_negative', 0)} the other way.</p>"""
        )
        if summary.get("one_sided"):
            extra += ('<p class="note">The mistakes run almost entirely one '
                      'direction. That is a threshold problem, not an accuracy '
                      'problem, and it has a different fix.</p>')

    return f"""<p class="metric {'good' if usable else 'flat'}">{ece:.3f}</p>
<p class="sub">expected calibration error</p>
<p>{_esc(reading)}</p>
{extra}"""


def render_slices(error_analysis):
    """Where the errors concentrate. An average error rate hides a slice that
    is twice as bad as the rest, and that slice is usually the interesting one."""
    slices = (error_analysis or {}).get("worst_slices") or []
    if not slices:
        return ""
    rows = "\n".join(
        f"    <tr><td>{_esc(item['column'])}</td><td>{_esc(item['bucket'])}</td>"
        f"<td>{item['count']}</td><td>{_pct(item['error_rate'])}</td></tr>"
        for item in slices
    )
    table = _table(("Slice", "Bucket", "Posts", "Error rate"), rows)
    return f"<h3>Where it goes wrong</h3>\n{table}"


def render_dataset(provenance, dataset_path):
    if not provenance:
        named = f" (<code>{_esc(dataset_path)}</code>)" if dataset_path else ""
        return _empty(
            f"No corpus fetched yet{named}. <code>python -m "
            "data.fetch_dataset</code> writes a provenance sidecar beside the "
            "CSV, and this section quotes it."
        )
    rows = [
        ("Source", provenance.get("source")),
        ("Site", provenance.get("site")),
        ("Licence", provenance.get("licence")),
        ("Posts", provenance.get("rows")),
    ]
    if provenance.get("first_post_utc"):
        rows.append(("Span (UTC)", f"{provenance['first_post_utc'][:10]} to "
                                   f"{provenance['last_post_utc'][:10]}"))
    rows.append(("Fetched", (provenance.get("fetched_at_utc") or "")[:19]))

    body = "\n".join(
        f"    <tr><th>{_esc(label)}</th><td>{_esc(value)}</td></tr>"
        for label, value in rows if value
    )
    attribution = ""
    if provenance.get("attribution"):
        attribution = f'<p class="note">{_esc(provenance["attribution"])}</p>'
    return _table((), body, css_class="kv") + attribution


def render_figures(available):
    if not available:
        return _empty("Figures are written by the run.")
    cards = "\n".join(
        f"""  <figure>
    <img src="figures/{_esc(name)}" alt="{_esc(caption)}" loading="lazy">
    <figcaption>{_esc(caption)}</figcaption>
  </figure>"""
        for name, caption in available
    )
    return f'<div class="figures">\n{cards}\n</div>'


CAVEAT = """Every temporal feature here is <strong>UTC</strong>, and that bounds
what the result means. "Late night" is the 00:00–05:00 UTC band, not the small
hours where the poster was sitting. Without a per-author timezone, a local clock
cannot be recovered: someone in California writing at 2am local lands at 09:00
UTC and is not flagged; someone in Berlin writing at 2am local is.
<br><br>
The feature is still predictive, because UTC hour correlates with local hour and
with how many people are awake to vote on a post. It is not evidence about
anyone's sleep, and nothing here should be read as a claim about a person."""


REPRODUCE = f"""<pre><code>git clone {REPO_URL}.git
pip install -r requirements.txt

# a real corpus, with its licence and span recorded beside it
python -m data.fetch_dataset --site stackoverflow --rows 20000 \\
    --out datasets/posts.csv

# the grid, the error analysis, the document, and this page
python run_experiment.py --dataset datasets/posts.csv --epochs 3 \\
    --embeddings --site</code></pre>"""


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

STYLESHEET = """:root {
  --bg: #fbfaf8; --panel: #ffffff; --ink: #1c1b19; --muted: #6b6862;
  --line: #e4e0d9; --accent: #1f5f4f; --warn: #8a5a10; --flat: #9a958c;
  --good: #1f5f4f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14150f; --panel: #1c1e17; --ink: #edeade; --muted: #9c9a8f;
    --line: #2e3126; --accent: #7fc4ae; --warn: #d9a441; --flat: #6e6c63;
    --good: #7fc4ae;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 17px/1.65 ui-serif, Georgia, "Iowan Old Style", serif;
  -webkit-font-smoothing: antialiased;
}
main { max-width: 46rem; margin: 0 auto; padding: 0 16px 6rem; }
h1 { font-size: clamp(1.8rem, 5vw, 2.6rem); line-height: 1.15; margin: .4rem 0 1rem; }
h2 {
  font-size: 1.05rem; text-transform: uppercase; letter-spacing: .09em;
  color: var(--muted); font-family: ui-sans-serif, system-ui, sans-serif;
  margin: 3.5rem 0 1rem; padding-bottom: .5rem; border-bottom: 1px solid var(--line);
}
h3 { font-size: 1.05rem; margin: 2rem 0 .6rem; }
a { color: var(--accent); }
.hero { padding: 4rem 0 0; }
.tag {
  display: inline-block; font: 600 .7rem/1 ui-sans-serif, system-ui, sans-serif;
  letter-spacing: .12em; text-transform: uppercase; color: var(--accent);
  border: 1px solid currentColor; border-radius: 99px; padding: .4rem .7rem;
}
.tag.warn { color: var(--warn); }
.answer {
  font-size: clamp(1.15rem, 3vw, 1.4rem); font-weight: 600;
  color: var(--accent); margin: 0 0 1.2rem;
}
.prose p { color: var(--muted); }
.banner {
  border-left: 3px solid var(--warn); background: var(--panel);
  padding: 1rem 1.2rem; margin: 2rem 0; font-size: .95rem; color: var(--muted);
}
.banner strong { color: var(--ink); }
.empty {
  color: var(--muted); font-style: italic; background: var(--panel);
  border: 1px dashed var(--line); border-radius: 8px; padding: 1.1rem;
}
.note, .sub, figcaption {
  font: .87rem/1.55 ui-sans-serif, system-ui, sans-serif; color: var(--muted);
}
.grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); }
.card {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 1.1rem 1.2rem 1.3rem;
}
.card h3 { margin: 0; font-size: 1.05rem; }
.card .features {
  font: .78rem/1 ui-sans-serif, system-ui, sans-serif; letter-spacing: .06em;
  text-transform: uppercase; color: var(--muted); margin: .35rem 0 .9rem;
}
.metric {
  font: 700 2.1rem/1 ui-sans-serif, system-ui, sans-serif;
  margin: 0 0 .6rem; color: var(--flat);
}
.card.good .metric, .metric.good { color: var(--good); }
.metric.best::after { content: " ★"; font-size: 1rem; vertical-align: super; }
.bar {
  position: relative; height: 8px; border-radius: 99px;
  background: var(--line); margin-bottom: .75rem;
}
.bar .fill { position: absolute; inset: 0 auto 0 0; background: var(--flat); }
.card.good .bar .fill { background: var(--accent); }
.bar { overflow: visible; }
.bar .fill { border-radius: 99px; }
.bar .marker {
  position: absolute; top: -4px; bottom: -4px; width: 2px;
  background: var(--ink); opacity: .75;
}
.split {
  display: flex; height: 2.4rem; border-radius: 6px; overflow: hidden;
  font: 600 .82rem/2.4rem ui-sans-serif, system-ui, sans-serif;
  margin: 1rem 0; color: #fff;
}
.split span { text-align: center; white-space: nowrap; min-width: 4.5rem; }
.split .fixed { background: var(--accent); }
.split .broken { background: var(--warn); }
.scroll { overflow-x: auto; margin: 1.2rem 0; }
table { width: 100%; border-collapse: collapse; font-size: .93rem; }
th, td { text-align: left; padding: .55rem .4rem; border-bottom: 1px solid var(--line); }
th {
  font: 600 .76rem/1.4 ui-sans-serif, system-ui, sans-serif;
  text-transform: uppercase; letter-spacing: .07em; color: var(--muted);
}
table.kv th { width: 34%; }
.figures { display: grid; gap: 1.6rem; }
figure { margin: 0; }
figure img {
  width: 100%; height: auto; border: 1px solid var(--line);
  border-radius: 8px; background: #fff;
}
pre {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 1rem; overflow-x: auto;
}
code { font: .82rem/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; }
footer {
  margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--line);
}
"""


def render_page(artifacts, synthetic=False, figures=(), generated_at=None,
                results_path=None):
    """Assemble the whole page. Pure: takes artefacts, returns HTML."""
    benchmark = artifacts.get("benchmark") or {}
    rows = benchmark.get("rows", [])
    comparison = artifacts.get("comparison")
    if comparison and "comparison" in comparison:
        comparison = comparison["comparison"]
    verdicts = classify_outcome(rows, comparison)
    error_analysis = artifacts.get("error_analysis")

    stamp = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")

    results_path = PROJECT_ROOT / "RESULTS.md" if results_path is None else results_path
    results_link = ""
    if results_path and Path(results_path).exists():
        results_link = (f' · <a href="{REPO_URL}/blob/master/RESULTS.md">'
                        "the full numbers</a>")

    sections = [
        render_banner(synthetic, rows),
        "<h2>The grid</h2>",
        render_grid(rows),
        "<h2>Fixed, or just shuffled?</h2>",
        '<p>A model can gain accuracy while getting a pile of '
        'previously-correct posts wrong. The only way to see that is post by '
        'post, on the rows where the two models disagree.</p>',
        render_ablation(verdicts),
        "<h2>Can you trust the confidence?</h2>",
        render_calibration(error_analysis),
        render_slices(error_analysis),
        "<h2>What this cannot tell you</h2>",
        f"<p>{CAVEAT}</p>",
        "<h2>The data</h2>",
        render_dataset(artifacts.get("provenance"), artifacts.get("dataset_path")),
        "<h2>Figures</h2>",
        render_figures(figures),
        "<h2>Run it yourself</h2>",
        REPRODUCE,
    ]

    body = "\n".join(part for part in sections if part)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Text vs timing — {_esc(QUESTION)}</title>
<meta name="description" content="An engagement-prediction ablation: does a post's timestamp add anything over its text? TF-IDF and BERT, each with and without the clock.">
<style>{STYLESHEET}</style>
</head>
<body>
<main>
{render_hero(verdicts, rows, synthetic)}
{body}
<footer>
  <p class="note">Generated by <code>python -m analysis.site</code> on {_esc(stamp)}.
  Every number on this page is read from the run artefacts — none of it is typed
  by hand.</p>
  <p class="note"><a href="{REPO_URL}">Source and method on GitHub</a>{results_link}</p>
</footer>
</main>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def copy_figures(source_dir, out_dir, featured=FEATURED_FIGURES):
    """Copy the featured figures next to the page. Returns the ones that exist.

    The page has to be self-contained: `docs/` is what GitHub Pages serves, and
    a relative link up into `outputs/` would 404 for every visitor while
    working perfectly on the machine that generated it.
    """
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    target = out_dir / "figures"
    available = []
    for name, caption in featured:
        origin = source_dir / name
        if not origin.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target / name)
        available.append((name, caption))
    return available


def build_site(artifacts, out_dir=SITE_DIR, synthetic=False, figure_dir=None):
    """Write index.html, the figures, and .nojekyll. Returns the page path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available = copy_figures(figure_dir, out_dir) if figure_dir else []

    # Pages runs Jekyll by default, which drops files starting with an
    # underscore and can rewrite what it thinks is markup.
    (out_dir / ".nojekyll").write_text("")

    page = out_dir / "index.html"
    page.write_text(render_page(artifacts, synthetic=synthetic, figures=available))
    return page


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render the run artefacts as a static page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Where benchmark.json and the figures live.")
    parser.add_argument("--dataset", default=None,
                        help="Dataset CSV, so its provenance sidecar is quoted.")
    parser.add_argument("--out", default=str(SITE_DIR),
                        help="Directory to write. GitHub Pages serves docs/.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Stamp the page as generated from sample data.")
    args = parser.parse_args(argv)

    artifacts = load_artifacts(args.output_dir, args.dataset)
    page = build_site(
        artifacts,
        out_dir=args.out,
        synthetic=args.synthetic,
        figure_dir=Path(args.output_dir) / "figures",
    )
    print(f"Wrote {page}")
    if not (artifacts.get("benchmark") or {}).get("rows"):
        print("  No benchmark rows -- the page renders its 'not run yet' state.")
    return page


if __name__ == "__main__":
    main()
