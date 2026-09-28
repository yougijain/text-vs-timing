# Text vs. Timing

**Does *when* a post goes up add anything over *what* it says?**

Everyone accepts that posting time affects engagement — it is folklore on every
platform with a "best time to post" blog article. Almost nobody measures it
against the text. This project does: same rows, same labels, same split, two
architectures, each run with and without the timestamp, and one table at the end
saying how much the clock was worth.

The target is binary: did a post land above typical engagement for its venue.
The comparison is a 2x2 grid.

|                       | text only | text + temporal |
|-----------------------|-----------|-----------------|
| **TF-IDF + LR**       | ablation  | full            |
| **Frozen MiniLM + LR**| ablation  | full            |
| **Zero-shot LLM** *(scaffolded)* | ablation | full |
| **BERT (fine-tuned)** | ablation  | full            |

- **TF-IDF + logistic regression** — word and character n-grams, optionally
  stacked with the temporal features. Fast, interpretable, and a genuinely
  strong competitor on short text. It is the control, not a straw man.
- **Frozen MiniLM + logistic regression** — mean-pooled sentence embeddings,
  no fine-tuning, the same linear head. This row exists because TF-IDF against
  fine-tuned BERT varies two things at once — pretrained semantics *and*
  task-specific training — so a BERT win tells you nothing about which one paid.
  Holding training fixed splits that apart.
- **BERT + temporal fusion** — a BERT encoder for the text; cyclical
  hour-of-day, day-of-week, month, weekend and late-night features (all
  **UTC**, see [Timezones](#timezones)) through a small MLP; the two
  concatenated before the classification head.

Running the same ablation on both architectures is the point. If the timestamp
helps a linear model and a transformer alike, the signal is in the data. If it
only helps one, what you are measuring is the fusion head.

## Status

Working end to end, offline. `python main.py --synthetic` trains, evaluates,
writes figures and a `results.json`, and reports accuracy against the
majority-class baseline. The test suite runs in a few seconds with no network.

| Component | State |
|---|---|
| Data cleaning and labelling | Done |
| Temporal feature engineering | Done |
| BERT + temporal model, attention pooling | Done |
| Training loop (warmup, clipping, AMP, checkpoint selection) | Done |
| Evaluation with baseline comparison | Done |
| TF-IDF + logistic regression baseline | Done |
| Benchmark grid (model x feature set) | Done |
| Frozen-embedding baseline (MiniLM + LR) | Done |
| Zero-shot LLM row | **Scaffolded, never run** |
| Descriptive analysis + figures | Done |
| Error analysis (slices, calibration, model comparison) | Done |
| Corpus fetcher + schema adapter (Stack Exchange, CC BY-SA 4.0) | Done |
| One-command experiment runner + generated `RESULTS.md` | Done |
| Published page generated from the same artefacts | Done — renders *not run yet* until there is a run |
| Test suite (401 tests, offline) | Done |
| Results on a real corpus | **Not run — see [Dataset](#dataset)** |

That last row is the honest one. Everything below the line marked *sample data*
was produced on generated text and is a check that the plumbing works, not a
finding.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs torch from PyPI, which on Linux pulls the CUDA
build (~2.5GB). For a much smaller CPU-only install, do torch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Swap `/cpu` for `/cu124` (or whatever matches your driver) to pick a CUDA build.

## Quick start

The baseline needs no network and no downloads — it generates sample data,
fits, and prints metrics plus the words driving each class:

```bash
python main.py --synthetic --model tfidf --split-strategy random
```

The neural path, on a miniature model so nothing is downloaded:

```bash
python main.py --synthetic --tiny-model --epochs 8 --learning-rate 1e-3 \
    --split-strategy random --max-length 64
```

Both, side by side:

```bash
python benchmark.py --synthetic --tiny-model --epochs 6 --learning-rate 1e-3 \
    --max-length 64 --batch-size 32
python benchmark.py --synthetic --skip-bert     # linear half only, seconds
```

With the frozen-embedding row (needs a model download, so it is opt-in):

```bash
python benchmark.py --dataset datasets/posts.csv --embeddings --epochs 3
python main.py --dataset datasets/posts.csv --model embeddings   # that row alone
```

The real thing — fetch a corpus, then train on it (downloads
`bert-base-uncased`):

```bash
python -m data.fetch_dataset --site stackoverflow --rows 20000 --out datasets/posts.csv
python main.py --dataset datasets/posts.csv --epochs 3
```

Analysis on its own, no training:

```bash
python -m analysis.timestamp_analysis --dataset datasets/sample_posts.csv
python -m analysis.pattern_detection  --dataset datasets/sample_posts.csv
python -m analysis.token_stats        --dataset datasets/sample_posts.csv --offline
python -m analysis.error_analysis     --dataset datasets/sample_posts.csv
python -m analysis.error_analysis     --dataset datasets/sample_posts.csv --compare-feature-sets
```

Tests:

```bash
pytest tests/ -q
```

## What a run produces

```
outputs/
├── benchmark.json                          # the comparison grid
├── checkpoints/bert_temporal_model.pth     # best epoch by validation accuracy
├── figures/hourly_activity.png             # posts per hour + engagement rate overlay
├── figures/weekly_heatmap.png              # weekday x hour posting volume
├── figures/loss_curve.png                  # batch loss + moving average
├── figures/error_rate_by_hour.png          # where the model fails
├── figures/calibration.png                 # confidence vs observed accuracy
├── error_analysis.json                     # slices, ECE, calibration bins
├── feature_set_comparison.json             # fixed vs broken, example by example
├── results.json                            # config, metrics, history, correlations
├── results_tfidf.json                      # baseline metrics and top features
└── results_embeddings.json                 # frozen-embedding row, when run

RESULTS.md                                  # the document, generated from the above
docs/index.html                             # the published page, with --site
```

## Dataset

The pipeline works in one schema — `selftext`, `score`, `created_utc`, with
`title` used when present. You do not have to rename anything: columns are
mapped on load (see [Column mapping](#column-mapping)).

### Getting a corpus

```bash
python -m data.fetch_dataset --site stackoverflow --rows 20000 \
    --from 2023-01-01 --to 2024-01-01 --out datasets/posts.csv
```

**Why Stack Exchange.** The question needs prose, an engagement signal, and a
real posting timestamp — and a licence you can defend out loud. "I found it on
Kaggle" is not a provenance story: a large share of the Reddit dumps there were
scraped against the platform's terms and redistributed with no licence at all.
Stack Exchange gives all three cleanly:

| | |
|---|---|
| **Text** | `body`, real prose, HTML stripped on fetch |
| **Engagement** | `score`, net votes — genuinely two-sided, unlike platforms where posts start at 1 and floor at 0 |
| **Timestamp** | `creation_date`, Unix epoch, UTC |
| **Licence** | CC BY-SA 4.0 — redistribution explicitly permitted with attribution |
| **Access** | Public API, no key needed; 300 requests/day anonymous = 30,000 posts |

That two-sided score matters more than it sounds. On a platform where every
post starts at 1, `score` is almost a count of views and the median label cut is
close to arbitrary. Stack Exchange scores go negative, so "above typical
engagement" is a real distinction.

`--site` takes any Stack Exchange site key. `stackoverflow` is the default;
the smaller sites (`cooking`, `scifi`, `worldbuilding`) have more discursive
prose and a different audience clock, which makes a second run on one of them a
genuine replication rather than a re-roll.

Code blocks are stripped from bodies by default. On a programming site they are
most of the character mass, and a character n-gram model handed a stack trace
will learn to predict engagement from variable names — a leak dressed up as a
feature. `--keep-code` turns that off.

### Provenance

Every fetch writes a sidecar next to the CSV:

```
datasets/posts.csv
datasets/posts.provenance.json    # source, licence, attribution, query, span, row count
```

A CSV with no provenance is a liability — six months later nobody can say what
it is, whether it may be redistributed, or how to reproduce it. The sidecar
answers all three, and it is what you quote when someone asks where the data
came from.

### Column mapping

Any CSV with text, a score, and a timestamp works. Known column names are
mapped automatically:

```
$ python main.py --dataset datasets/posts.csv --model tfidf
  1500 raw rows, columns: ['question_id', 'title', 'body', 'score', 'creation_date']
  column map: 'creation_date' -> 'created_utc' (inferred)
  column map: 'body' -> 'selftext' (inferred)
```

Inference only fills a canonical column that is genuinely **absent**. A frame
that already has `score` keeps it even if it also has `points` — silently
relabelling the target is how you train on the wrong thing and never find out.
When the guess is wrong or missing, say so explicitly:

```bash
python main.py --dataset mine.csv --column-map 'selftext=body_text,score=upvotes'
```

### The synthetic fallback

```bash
python -m data.make_sample_data --rows 2000 --out datasets/sample_posts.csv
```

That is a **testing utility**, not a data source. It writes forum-shaped
template text with a documented planted signal (evening engagement bump,
off-hours penalty, bonus for asking a question) plus the `[removed]`/`[deleted]`
rows a real dump is full of, so the cleaning step has something to remove. It is
fabricated text. It is not data about anyone, and nothing measured on it is a
result.

## The zero-shot LLM row

**Scaffolded, not run.** Every piece is implemented and tested; no API call has
ever been made from this repo and no LLM row appears in any grid. It is a wired
socket, not a result, and this section says so rather than letting a reader
assume otherwise.

It earns a place because it splits an axis the other three rows cannot:

| Row | Representation | Task training |
|---|---|---|
| TF-IDF | lexical | yes |
| Frozen MiniLM | pretrained semantics | yes |
| Fine-tuned BERT | pretrained semantics | yes, end to end |
| **Zero-shot LLM** | pretrained semantics | **none** |

Every other row learns this task from this corpus. The LLM has never seen a
label. If it matches models trained on thousands of them, the labels carry less
information than the corpus size suggests — worth knowing before anyone builds
a labelling pipeline.

### The ablation becomes a prompt

This is the part worth reading. "Text only" and "text + temporal" are not
feature matrices for an LLM — they are two prompts, and the timestamp has to be
rendered into English or the model cannot use it at all:

```
Posted at 17:00 UTC on a Thursday.

Post:
Need a second opinion on this trace...
```

`UTC` is stated explicitly. Without it the model reasonably assumes local time
and reasons about the poster's body clock — exactly the inference
[Timezones](#timezones) says this data cannot support. The prompt has to be as
honest about what it knows as the rest of the pipeline is.

### Guards, because this row bills per post

```bash
python main.py --dataset datasets/posts.csv --model llm     # opt-in, real calls
python benchmark.py --dataset datasets/posts.csv --llm      # adds both arms
```

- **Dry-run by default.** Constructed programmatically without a classifier, it
  refuses to call anything and quotes the price instead:
  `A real pass over 226 prompts would cost roughly $0.12 on claude-opus-5.`
- **Disk-cached** by (model, prompt), so re-running the grid after a bug fix is
  free rather than a second bill.
- **`anthropic` is not in `requirements.txt`** — installing it is part of opting in.
- **Never a default.** No flag path reaches this row without `--llm` or
  `--model llm`.

### Deliberately left for the next iteration

Few-shot examples drawn from the training split (this is zero-shot), the
Batches API (halves the cost), and concurrency — the loop is serial on purpose,
since a fast loop that silently spends money is the wrong default for
scaffolding.

There is also no `predict_proba`: a schema-constrained label carries no
calibrated confidence, so this row reports `loss` as 0.0 and the calibration
section does not apply to it. Inventing a probability would put a meaningless
number in the report.

## Running the experiment

One command, one seed, one split, straight through to the document:

```bash
python -m data.fetch_dataset --site stackoverflow --rows 20000 --out datasets/posts.csv
python run_experiment.py --dataset datasets/posts.csv --epochs 3 --embeddings --site
```

It runs four steps and writes `RESULTS.md`, plus a fifth with `--site`:

1. `benchmark.py` — the grid → `outputs/benchmark.json`
2. error analysis — slices, calibration → `outputs/error_analysis.json`
3. `--compare-feature-sets` — fixed vs broken → `outputs/feature_set_comparison.json`
4. `analysis.report` — the document
5. `analysis.site` — the published page (opt-in; `docs/` is tracked, so a smoke
   run should not rewrite it)

`bert-base-uncased`, `--split-strategy temporal`, `--label-strategy median`,
3 epochs, `--max-length 256`, `--batch-size 16`, AMP on by default — a Colab T4
is enough.

### Two smoke runs before the real one

**Offline, no data, about a minute on CPU** — checks the plumbing:

```bash
python run_experiment.py --synthetic --skip-bert
```

**On real data, capped, a couple of minutes** — do this one before committing to
the full corpus:

```bash
python run_experiment.py --dataset datasets/posts.csv --max-rows 2000 --skip-bert --embeddings
```

Passing the synthetic smoke run proves very little about a real corpus. The
generator writes clean, uniform, template text; a real dump does not. Real
bodies carry markup the cleaner has to strip, encodings it has to survive, rows
where the body is empty and only the title has content, and a score
distribution that decides whether `--label-strategy median` finds a balanced cut
at all. Any of that can fail, and the cheapest place to find out is a two-minute
run rather than a forty-minute one.

Read three things off it before scaling up:

| Check | Where | What is wrong if it looks off |
|---|---|---|
| Rows surviving cleaning | `Cleaned dataset: N in, M out` | A large drop means the body column is mostly markup, placeholders, or under `min_tokens` |
| Class balance | `Labels: {...}` | A lopsided `positive_rate` means the median cut did not land — the score distribution is too concentrated |
| Lift over baseline | end of each grid cell | At or below zero on 2000 rows is usually the label, not the model |

### No GPU?

The BERT rows are the only part that wants one. Everything else runs on a
laptop, and `--skip-bert` still gives you a full four-cell grid — TF-IDF and
frozen MiniLM, each with and without the timestamp:

```bash
python run_experiment.py --dataset datasets/posts.csv --skip-bert --embeddings
```

That is a real result on real data, and it answers the project's question. The
fine-tuned rows sharpen it; they are not what makes it valid.

### Nothing in the report is hand-typed

A results table copied into a README goes stale on the next run and nobody
notices, because a number that is slightly wrong looks exactly like a number
that is right. `analysis/report.py` reads the artefacts and writes the whole
document — grid, lift per cell, ECE, fixed-vs-broken counts, figures, the UTC
caveat.

It also **decides what happened** rather than leaving that to the writer:

```
### The page

`RESULTS.md` is written for someone who already knows what a macro F1 is.
`analysis/site.py` writes the same numbers for someone who does not:

```bash
python -m analysis.site --dataset datasets/posts.csv     # -> docs/index.html
```

One self-contained HTML file, no JavaScript, no CDN, no analytics — plus the
figures copied in beside it so the directory renders on its own. It leads with
the question and the answer in a sentence, then the grid as bars against the
majority-class baseline, then fixed-versus-broken, then calibration, then what
the result cannot tell you.

It reads the same artefacts `analysis.report` does and imports its verdict
functions rather than recomputing anything, so the page and the document cannot
disagree. Where the numbers are missing it renders *not run yet* in each
section, which is the state committed here now.

One detail worth naming: a cell is coloured as a win only when the low end of
its confidence interval clears the baseline. A flat threshold would paint a
three-point gain on 226 rows green, and three points on 226 rows is noise.

To publish, set GitHub Pages to **`master` / `docs`** in the repository
settings once. After that the page updates by committing it, and it serves at
`https://yougijain.github.io/text-vs-timing/`.

## What happened

TF-IDF beats BERT (0.8230 vs 0.6239). On short text with a few thousand rows
this is a common and legitimate outcome, not a bug to tune away.

The timestamp: helps.

> Temporal verdict from McNemar's test on the paired predictions, not on the
> accuracy difference.
```

"BERT won" is the least interesting outcome and the easiest to claim by
eyeballing a table. The four readings the project cares about are selected by a
function with a stated rule, and the report names which rule it used.

**The temporal verdict uses McNemar's test**, which is the correct one for two
classifiers scored over the same rows. A two-sample test on the two accuracies
throws away the pairing — the cases both models get right carry no information
about which is better — and the disagreements it *should* use are exactly the
"fixed" and "broken" cells the feature-set comparison already produces. Below
~25 discordant pairs the chi-square approximation is unreliable, so it falls
back to an exact binomial.

The model-vs-model verdict is weaker and says so: `benchmark.json` stores
metrics rather than predictions, so that comparison falls back to overlapping
confidence intervals.

## Error analysis

An accuracy number says how often a model is right. It does not say whether the
mistakes are spread evenly or piled into one slice, and that difference decides
whether a result is usable.

```bash
python -m analysis.error_analysis --dataset datasets/sample_posts.csv
```

Reports:

- **Error rate by slice** — posting hour, weekday, text length, stylistic
  markers. Buckets under 15 samples are excluded from the ranking, so a 3-row
  bucket at 100% error cannot pose as the model's biggest weakness.
- **Error asymmetry** — mistakes running almost entirely one direction mean a
  threshold problem, not an accuracy problem. Different fix.
- **Calibration** — a reliability table and expected calibration error. Above
  ~0.1 the probabilities do not mean what they say, so they can rank but cannot
  be thresholded for a precision target. The report says so.
- **The most confident mistakes** — a model that is 95% sure and wrong has
  learned something false. These explain an error rate faster than any statistic.

On the sample data the baseline sits at 17.6% error with ECE 0.053, and its
most confident mistakes are all late-night posts that *did* well, called low
engagement at 0.95+. It learned the planted late-night penalty hard enough to
override everything else — which is the kind of thing only error analysis
surfaces.

### Does the ablation fix errors or just move them?

```bash
python -m analysis.error_analysis --dataset datasets/sample_posts.csv --compare-feature-sets
```

The benchmark table says the temporal features raise accuracy. It cannot say
whether they fix predictions or shuffle errors around — a model can gain overall
while getting worse where it matters. This compares the two runs example by
example:

| | Count |
|---|---|
| Both correct | 116 |
| Both wrong | 25 |
| Only text-only correct | 14 |
| Only text+temporal correct | **67** |

67 fixed against 14 broken, a net gain of 53. The features are adding
information, not reshuffling it — a stronger claim than the accuracy delta alone
supports.

`compare_predictions()` does the same for any two models, so TF-IDF and BERT can
be compared the same way once real BERT weights are available.

## Timezones

**Every temporal feature here is UTC, and that is a real limit on what they
mean.**

`created_utc` is all most public dumps give us. There is no per-author timezone,
so a poster's local clock time cannot be recovered. Someone in California
writing at 2am local shows up at 09:00–10:00 UTC and is *not* flagged
late-night; someone in Berlin writing at 2am local is.

So `is_late_night` does not mean "written in the small hours". It means "written
in the 00:00–05:00 UTC band", which selects for a mix of local times that
depends on where the posters live. The feature is still predictive — UTC hour
correlates with both local hour and how many people are awake to vote — but the
circadian interpretation is not supported by this data. Don't describe it as a
sleep or chronotype measure.

Fixing it properly means inferring each author's timezone from the distribution
of their own posting times across a long history. That is real work and is not
implemented. Until it is, call these features UTC posting hour and say so in any
write-up.

## Labels

Post `score` is turned into a binary target. The default is `--label-strategy
median`, which picks the score threshold splitting the data closest to 50/50 —
read it as *did this post land above typical engagement for this venue*.

`--label-strategy positive` reproduces a naive `score > 0` rule and is kept only
for comparison. Do not use it for results: on platforms where posts start at a
score of 1, it puts over 90% of any real dump in one class and the model learns
to answer "1" every time. That is why evaluation always prints the
majority-class baseline next to accuracy.

## Splits

`--split-strategy temporal` (default) sorts chronologically and validates on the
most recent posts — the honest setup when the features include the timestamp,
because it does not let the model see the future. `--split-strategy random`
shuffles, which is the right choice when you only care about linguistic content
and want an i.i.d. split. Class balance for both halves is printed, with a
warning if either lands single-class.

## Running without network access

`--tiny-model` and `--offline-tokenizer` train a WordPiece tokenizer on your own
corpus and size a miniature randomly-initialised BERT to match, so nothing is
downloaded. This is for exercising the pipeline and for CI — the model carries
no pretrained knowledge, so its accuracy is not a result.

## Useful flags

| Flag | Effect |
|---|---|
| `--model` | `bert` (default), `tfidf`, or `embeddings` |
| `--encoder-name` | Encoder for `--model embeddings` (frozen, never fine-tuned) |
| `--llm-model` | Model for `--model llm`. Bills per validation row |
| `--synthetic` | Generate and use sample data |
| `--tiny-model` | Random miniature BERT, no download |
| `--offline-tokenizer` | Corpus-trained tokenizer instead of downloading one |
| `--no-temporal` | Text-only ablation |
| `--no-attention-pooling` | Mean pooling instead of learned attention |
| `--freeze-bert` | Train only the head; much faster on CPU |
| `--max-rows N` | Cap rows for a quick run |
| `--column-map` | `canonical=source` pairs, e.g. `'selftext=body'` |
| `--label-strategy` | `median` (default), `threshold`, `positive` |
| `--split-strategy` | `temporal` (default) or `random` |
| `--device` | `auto`, `cpu`, `cuda` |

`python main.py --help` lists all of them.

## Layout

```
.
├── main.py                        # CLI pipeline: load -> analyse -> train -> evaluate
├── benchmark.py                   # TF-IDF vs BERT x temporal grid, as one table
├── run_experiment.py              # grid + error analysis + RESULTS.md, one command
├── analysis/
│   ├── pattern_detection.py       # stylistic markers, crossed with posting hour
│   ├── error_analysis.py          # error slices, calibration, model comparison
│   ├── report.py                  # artefacts -> RESULTS.md, with the significance tests
│   ├── site.py                    # artefacts -> docs/index.html, the published page
│   ├── timestamp_analysis.py      # temporal EDA and figures
│   └── token_stats.py             # token-length distribution, truncation rates
├── data/
│   ├── data_loader.py             # Dataset and train/val splitting
│   ├── fetch_dataset.py           # Stack Exchange corpus fetcher + provenance
│   ├── inspect_dataset.py         # quick look at a raw CSV
│   ├── make_sample_data.py        # synthetic generator (testing utility)
│   ├── preprocess.py              # cleaning, labelling, tokenization
│   └── schema.py                  # map any CSV onto the canonical columns
├── models/
│   ├── attention_layer.py         # attention pooling over token states
│   ├── bert_temporal_model.py     # BERT + temporal fusion classifier
│   ├── embedding_baseline.py      # frozen sentence embeddings + logistic regression
│   ├── llm_baseline.py            # zero-shot LLM row (scaffolded, opt-in)
│   ├── model_utils.py             # seeding, devices, checkpoints
│   └── tfidf_baseline.py          # TF-IDF + logistic regression baseline
├── training/
│   ├── config.py                  # every tunable, in one dataclass
│   ├── evaluate.py                # metrics with baseline comparison
│   └── train.py                   # training loop
├── docs/                          # the published page (GitHub Pages serves this)
├── utils/
│   ├── loss_utils.py              # loss smoothing
│   └── time_utils.py              # timestamp parsing, temporal features
└── tests/                         # offline test suite
```

## Interpreting results

Accuracy on its own means little on an imbalanced split, so every evaluation
prints the majority-class baseline and the lift over it. **Lift at or below zero
means the model learned nothing**, whatever the accuracy says — the run tells
you so explicitly.

Two ways to inspect what a model keyed on. The neural path's attention weights
(`predict(..., return_attention=True)`) give per-token relevance. The baseline's
`top_features()` gives signed per-word coefficients, which is considerably more
legible — literal words with weights, rather than a distribution over
wordpieces. If the top features look like artefacts, the label is leaking.

Run the baseline first. It takes seconds, it needs no GPU and no downloads, and
if it already gets most of the available accuracy then the transformer is
carrying very little and the honest write-up says so.

Then the frozen-embedding row, which is the cheapest way to find out *why* any
gap exists. Three readings, each pointing somewhere different:

| Pattern | What it means |
|---|---|
| Embeddings > TF-IDF, BERT > embeddings | Both axes pay; the fine-tuning budget is justified |
| Embeddings > TF-IDF, BERT ≈ embeddings | Pretrained semantics are the whole story — ship the embedding model: one forward pass, no training, no GPU at inference |
| Embeddings ≈ TF-IDF | The signal is lexical. A transformer is not reading anything a bag of words cannot |

The middle row is the common outcome on short text, and it is the one worth
knowing before anyone commits to a fine-tuning pipeline.

Then run the error analysis. Accuracy is one number; where a model fails is the
part that decides whether it is usable, and it is usually the more interesting
half of a write-up.

## Why it is built this way

The decisions a reviewer is most likely to ask about, and the answers.

**Attention pooling over mean pooling** (fine-tuned path). Mean pooling weights
every token equally, including the padding-adjacent filler that dominates a
short post. Learned attention lets the classifier put weight where the signal
is. It is a real parameter cost, so `--no-attention-pooling` measures whether
it earned it rather than assuming.

**Mean pooling, not attention or `[CLS]`** (frozen path). Opposite call, same
reasoning. A frozen encoder's `[CLS]` vector only acquires meaning from task
training, and there is none here; the sentence-transformer checkpoints were
trained with mean pooling, so that is what reproduces them.

**Warmup.** Adam's second-moment estimate is near-garbage for the first few
hundred steps, so a full learning rate at step 0 takes large, badly-scaled steps
through a pretrained encoder and undoes some of what it knows. Ramping over the
first 10% keeps early updates small until the optimiser state is worth trusting.

**AMP.** Forward and backward in fp16 with an fp32 master copy of the weights,
and a loss scaler to stop small gradients flushing to zero. Roughly 2x
throughput and half the activation memory on a modern GPU. It changes numerics —
runs are not bitwise reproducible against an fp32 run — so `--no-amp` exists for
when that matters more than speed.

**Chronological split, by default.** The timestamp is a *feature*. A random
split trains on posts from after the ones it validates on, so the model has seen
the future of its own validation set — and since engagement trends drift, that
leaks. `--split-strategy random` is right only when you care about linguistic
content alone.

**The 0.1 line on ECE.** Above it, a model saying 0.9 is not right 90% of the
time, so the probabilities can rank posts but cannot be thresholded for a
precision target. Below ~0.05 they are usable as probabilities. The line is a
convention, not a law, and the report states the number next to it.

**Accuracy printed beside a majority-class baseline.** On a 90/10 split, 90%
accuracy is what you get by answering with the majority class every time.
Reporting accuracy alone on an imbalanced target is how a model that learned
nothing gets written up as a success. Lift at or below zero is the tell.

**Why `--label-strategy positive` is kept but not used.** On platforms where
posts start at a score of 1, `score > 0` puts over 90% of a dump in one class
and the model learns to answer "1" every time — at, and not above, the majority
baseline. It is in the repo because reproducing a degenerate baseline is how you
demonstrate it is degenerate, not because it should be run.

## Scope and limits

This predicts post engagement from text and a UTC timestamp. That is the whole
claim. It is not a diagnostic tool of any kind, it makes no inference about any
author, and it says nothing about any individual.

The temporal features carry the timezone caveat above, so they are not evidence
about anyone's sleep or circadian rhythm. The stylistic markers in
`analysis/pattern_detection.py` are hand-built keyword lists — crude proxies for
writing style, deliberately kept visible and editable rather than hidden behind
a model download, so you can audit exactly what is being counted.

Engagement is also not quality. A score is a measure of what an audience
rewarded at a particular hour on a particular platform, and a model that
predicts it is modelling that audience's behaviour, not the merit of the writing.

---

# Appendix: the sample-data run


> **Sample data.** Generated template text with a deliberately planted
> time-of-day signal. These numbers demonstrate that the pipeline works end to
> end and that the ablation is wired up correctly. They say nothing about real
> posts, and they are not the project's result.

Full grid, `benchmark.py --synthetic --tiny-model --epochs 6 --learning-rate 1e-3`,
majority-class baseline 0.500:

| Model | Features | Accuracy | Macro F1 | Lift |
|---|---|---|---|---|
| TF-IDF | text only | 0.5856 | 0.5850 | +0.0856 |
| TF-IDF | text + temporal | **0.8243** | 0.8239 | +0.3243 |
| BERT (tiny, random) | text only | 0.6081 | 0.6052 | +0.1081 |
| BERT (tiny, random) | text + temporal | 0.6667 | 0.6657 | +0.1667 |

Two things to read off this, and one trap:

**Temporal features help in both architectures** — +0.24 for the linear model,
+0.06 for the neural one. That the gap appears on both is the structure the
generator planted, recovered. On real data, whether it appears on both is the
question.

**The linear baseline wins by a wide margin.** Do not read that as "TF-IDF beats
BERT": the BERT row is a randomly initialised miniature model with no pretrained
weights, because the environment it ran in could not reach huggingface.co. It is
a plumbing check, not a competitor. Re-run without `--tiny-model` before drawing
any conclusion — `benchmark.py` prints this caveat itself when it detects the
tiny model.

The baseline's coefficients recover the planted structure directly: `hour_cos`
at -1.41 and `hour_sin` at +1.37 dominate every word feature, with
`is_late_night` at -0.48 — an evening bump and a late-night penalty, which is
exactly what the generator plants. (In the synthetic data the clock is
unambiguous because the generator defines it; on a real dump read these as UTC
bands, per [Timezones](#timezones).)
