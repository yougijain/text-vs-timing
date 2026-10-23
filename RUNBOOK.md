# Running it for real

Everything in this repository works except the one thing it exists to do: the
grid has never been run on a real corpus. This is the sequence that fixes that.

**You do not need a GPU to get a result.** `--skip-bert` gives a complete
four-cell grid — TF-IDF and frozen MiniLM, each with and without the timestamp
— and it answers the project's central question: does the timestamp add
anything over the text.

It does not answer the other half. Two of [the four
outcomes](#the-four-outcomes) are about BERT against TF-IDF, and without a BERT
row the report says so in as many words:

> _Not enough of the grid was run to compare architectures._

So the result is read twice: once after the linear grid, for the temporal
verdict, and again after the BERT rows, for the whole thing. Stage 5 needs a
free Colab T4 and is the only stage that does.

| Stage | What | Roughly |
|---|---|---|
| 1 | Fetch a corpus | 10–20 min |
| 2 | Capped sanity run | 2 min |
| 3 | The linear grid | 10–30 min |
| 4 | Read the temporal half | as long as it deserves |
| 5 | The BERT rows, on a T4 | 30–60 min |
| 6 | Read it again, whole | — |
| 7 | Commit and publish | 1 min |

Stages 1–4 and 7 are a laptop and about an hour.

---

## Stage 0: check the checkout

```bash
git pull origin master
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

**On Linux, install CPU torch first.** The PyPI default is the CUDA build —
several gigabytes of NVIDIA wheels for a machine that may have no GPU. Skip
this line on macOS, where the PyPI wheel is CPU-only already.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then the rest, and the suite:

```bash
pip install -r requirements.txt
pytest tests/ -q          # expect 411 passed
```

`--skip-bert` skips *training* BERT, not installing it: `data/preprocess.py`
imports `AutoTokenizer` at module level, so transformers and torch come down on
every path, GPU or not. Budget about 3 GB.

If the suite is red before you start, fix that first. Everything downstream
assumes it is green.

---

## Stage 1: fetch a corpus

```bash
python -m data.fetch_dataset --site stackoverflow --rows 20000 \
    --from 2023-01-01 --to 2024-01-01 --out datasets/posts.csv
```

**Watch the quota.** Anonymous access to the Stack Exchange API allows 300
requests per day, and the fetcher takes 100 questions per request. 20,000 posts
is 200 requests — two thirds of the day's allowance in one go. If it runs out
it stops with a message telling you how many it collected and to resume
tomorrow; a free key from Stack Apps passed as `--api-key` raises the ceiling.

**Pin a date window.** `--from` and `--to` are not decoration. A corpus that
runs to today is a corpus whose newest posts have had hours to accumulate votes
while the oldest have had years, and the temporal split then trains on
well-scored old posts and validates on under-scored new ones. Closing the
window at a date well in the past gives every post a comparable chance to be
voted on.

**Code blocks are stripped by default.** On a programming site they are most of
the character mass, and a character n-gram model handed a stack trace learns to
predict engagement from variable names. `--keep-code` turns that off if you
want to see the difference.

### Check what you got

```bash
python -m data.inspect_dataset --dataset datasets/posts.csv
```

Three things matter:

- **Row count.** Well under 20,000 means the fetch stopped early — read the
  message it printed.
- **Score spread.** If nearly every post scores 0 or 1, the median cut will be
  close to arbitrary and the labels will be noise. Try a smaller, more
  discursive site (`scifi`, `cooking`, `worldbuilding`).
- **The sidecar.** `datasets/posts.provenance.json` must exist. It records the
  site, licence, row count, date span and fetch time, and `RESULTS.md` and the
  published page both quote it.

Commit the sidecar now. The CSV is gitignored — it is large and not yours to
redistribute — but the sidecar is a few hundred bytes and it is the record of
where the data came from.

```bash
git add datasets/posts.provenance.json
git commit -m "data: record provenance for the Stack Overflow corpus"
```

---

## Stage 2: a capped sanity run

```bash
python run_experiment.py --dataset datasets/posts.csv --max-rows 2000 \
    --skip-bert --embeddings
```

Two minutes, and it is worth every one of them. Passing the synthetic smoke run
proves very little about a real corpus: the generator writes clean uniform
template text, and a real dump does not. This is the first time the pipeline
sees real markup, real encodings, rows where the body is empty and only the
title has content, and a real score distribution.

You are checking that it reaches step 4 and writes a document. Specifically:

- **all four steps complete** — a crash in step 2 means a column-mapping
  problem, and the error names the fix
- **the label split is not lopsided** — `median` searches for the cut closest
  to an even split, so a heavily skewed result means the score distribution
  itself is degenerate, not the strategy
- **the grid has four rows** — two models, two feature sets

Do not read the numbers. 2,000 rows is not a result.

---

## Stage 3: the linear grid

```bash
python run_experiment.py --dataset datasets/posts.csv \
    --skip-bert --embeddings --site
```

The MiniLM encoding pass dominates the wall clock; the rest is seconds. It
writes:

```
outputs/benchmark.json                 the grid
outputs/error_analysis.json            slices, calibration bins, ECE
outputs/feature_set_comparison.json    fixed vs broken, post by post
outputs/figures/*.png
RESULTS.md                             the document
docs/index.html                        the published page
```

Nothing in either document is typed by hand. Both are generated from those JSON
files, which is why they cannot disagree with each other.

---

## Stage 4: read the temporal half

Read `RESULTS.md` in this order. Skipping to the accuracy is how people talk
themselves into findings.

**1. The majority-class baseline.** Every accuracy is meaningless without it.
82% is excellent on a balanced target and useless on one that is 85% negative.

**2. Lift over baseline, per cell.** At or below zero means that cell learned
nothing, whatever its accuracy reads. The published page colours a cell as a
win only when the low end of its confidence interval clears the baseline — a
three-point gain on a couple of hundred validation rows is noise.

**3. Fixed versus broken.** A model can gain accuracy while getting a pile of
previously-correct posts wrong. `feature_set_comparison.json` splits the
disagreements, and the report runs McNemar's test on them — the correct test
for two classifiers scored over the same rows, because the posts both models
get right carry no information about which is better.

**4. ECE.** Below 0.1 the probabilities mean roughly what they say and you can
threshold them for a precision target. Above it they still rank posts, but a
0.9 is not a 90% chance.

### The four outcomes

The report names which one occurred and says which rule it used. All four are
publishable:

| | What it means |
|---|---|
| Temporal helps both architectures | The timing signal is real and not an artefact of one model's inductive bias. |
| Temporal helps only one | The fusion head is doing the work, not the clock. Say so. |
| BERT clearly beats TF-IDF | The text carries structure a bag of words cannot reach. |
| They tie | The transformer is not earning its compute on short text. Worth knowing before anyone deploys one. |

**The first two are answerable now; the last two are not.** They need a BERT
row, and at this point there isn't one. Write down the temporal verdict and
leave the architecture question open until stage 6.

**If the timestamp adds nothing, that is the result.** It is a cleaner finding
than a marginal gain, and it is the one most people would quietly tune away.
The project is built to report it: the honest negative is why the majority
baseline, the paired test and the calibration check are all in the report
rather than an accuracy number on its own.

What you must not write, whatever the numbers say: anything about when a person
was awake. Every temporal feature here is UTC. "Late night" is the 00:00–05:00
UTC band, not the small hours where the poster was sitting. Without a
per-author timezone a local clock cannot be recovered, and the feature is
predictive because UTC hour correlates with how many people are awake to vote —
not because of anything about the author.

---

## Stage 5: the BERT rows, on a T4

Worth doing once the linear grid is in, and only then — you want to know what
the cheap models say before spending a GPU hour on whether the expensive one
beats them. If TF-IDF ties BERT, that is the more interesting finding, and you
want to walk into it rather than discover it after the fact.

In Colab: Runtime → Change runtime type → T4 GPU, then

```python
!git clone https://github.com/yougijain/text-vs-timing.git
%cd text-vs-timing
!pip install -q -r requirements.txt
```

The corpus is gitignored, so upload `datasets/posts.csv` from the file pane
rather than re-fetching and burning another day's quota. The provenance sidecar
comes down with the clone, so it still matches.

```python
!python run_experiment.py --dataset datasets/posts.csv \
    --epochs 3 --max-length 256 --batch-size 16 \
    --embeddings --site
```

Defaults are already what you want: `bert-base-uncased`, temporal split, median
labels, AMP on. `--no-amp` turns mixed precision off — the fix if the loss goes
to NaN, and worth one deliberate run either way, because the difference is a
number you can quote.

**Write the timing down while it is on screen.** `train_model` prints
`Training finished in Ns` and that is the only place it goes: `benchmark.py`
does not keep the history, so the figure reaches neither `benchmark.json` nor
`RESULTS.md`, and a recycled Colab runtime takes the scrollback with it. Note
that it is total training time across all epochs, not per epoch.

Download `RESULTS.md`, `docs/index.html` and `outputs/` when it finishes, and
commit them from your laptop.

---

## Stage 6: read it again, whole

The BERT run regenerates `RESULTS.md` over the top of the linear one, so the
document you read at stage 4 is gone. Read it again in the same order, and this
time all four outcomes are on the table — the report will have replaced _"not
enough of the grid was run"_ with a headline naming which architecture won, or
saying they are within sampling error of each other.

Check that the temporal verdict did not change when the architectures did. If
the timestamp helped the linear models and not BERT, or the reverse, that is
the "helps only one" outcome and it points at the fusion head rather than the
clock. Say so rather than averaging the two.

---

## Stage 7: publish

```bash
git add RESULTS.md docs/
git commit -m "results: first run on the Stack Overflow corpus"
git push origin master
```

Pages serves `master` / `docs`, so the live page updates on push. Then change
the two lines in `README.md` that currently say the run has not happened: the
status table's last row, and the sample-data appendix note.

Publishing twice is fine, and better than waiting. Commit after stage 4 so the
page stops saying "not run yet", and again after stage 6 when the BERT rows
land. If you never get to a GPU, the stage-4 commit is still a complete,
honest result.

---

## Know cold

Running it is half the work. These are the questions the project invites, and
an answer you have to reconstruct on the spot is an answer you do not have.

1. **Tokenization.** What does BERT's tokenizer do with a word it has never
   seen, and what happens to a post longer than `--max-length 256`?
2. **Attention pooling.** The fine-tuned model pools token states through a
   learned attention layer rather than taking `[CLS]` or a flat mean. What does
   that buy, and what does the frozen MiniLM path do instead?
3. **Warmup and clipping.** Why does the learning rate start near zero and ramp
   up, and what is gradient clipping at 1.0 protecting against?
4. **Cyclical encoding.** Why is hour-of-day two features and not one? What
   specifically breaks if you feed the model the integer 23?
5. **What the frozen baseline isolates.** TF-IDF against fine-tuned BERT varies
   two things at once. Name both, and say which one the MiniLM row holds fixed.
6. **The split.** Why chronological rather than random, and why pin `--from`
   and `--to`? Both answers are about the same failure.
7. **McNemar.** What does it test, why is a two-sample test on the two
   accuracies the wrong tool here, and which cells of the 2x2 carry no
   information about which model is better?
8. **The label.** How is the target defined? `median` is not a plain median —
   say what `_balanced_cut` actually does and why it exists.
9. **Baseline, lift and ECE.** Why is an accuracy without its baseline
   unreadable? What does your ECE number license you to do with the predicted
   probabilities, and what does it not?

And the one that is not a question, because getting it wrong is not a gap in
knowledge but a claim you cannot support: **every temporal feature here is
UTC.** Nothing in this project says anything about when a person was awake.

---

## Appendix: what can go wrong

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: missing required column(s)` | The CSV uses names the schema does not recognise | `--column-map 'selftext=body,created_utc=creation_date'` |
| Quota error mid-fetch | 300 requests/day used | Resume tomorrow, or `--api-key` |
| Labels almost all one class | Score distribution too flat for a median cut | A smaller, more discursive site |
| `Training labels contain a single class` | The capped run sliced a degenerate window | Raise `--max-rows`, or drop the cap |
| Encoder download fails | No network, or the hub is unreachable | `--skip-bert` without `--embeddings`, or `--encoder-name` pointed at a local copy |
| Loss goes to NaN on the BERT rows | AMP numerics | `--no-amp` |
| Accuracy far above baseline on the first try | Usually a leak, not a win | Check the split is `temporal`, and that no engagement-derived column reached the features |

That last row is the one to take seriously. A result that looks too good on the
first attempt is the most common way a project like this goes wrong.
