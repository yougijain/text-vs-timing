# Running it for real

Everything in this repository works except the one thing it exists to do: the
grid has never been run on a real corpus. This is the sequence that fixes that.

**You do not need a GPU.** `--skip-bert` gives a complete four-cell grid —
TF-IDF and frozen MiniLM, each with and without the timestamp — and that grid
answers the question. The fine-tuned BERT rows sharpen the answer; they are not
what makes it valid. [Appendix A](#appendix-a-the-bert-rows-on-colab) covers
them if you want them.

Budget about an hour, most of it waiting. Three commands do the work.

| Stage | What | Roughly |
|---|---|---|
| 1 | Fetch a corpus | 10–20 min |
| 2 | Capped sanity run | 2 min |
| 3 | The real run | 10–30 min |
| 4 | Read the result | as long as it deserves |
| 5 | Commit and publish | 1 min |

---

## Stage 0: check the checkout

```bash
git pull origin master
pip install -r requirements.txt
pytest tests/ -q          # expect 401 passed
```

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

## Stage 3: the real run

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

## Stage 4: read the result honestly

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

## Stage 5: publish

```bash
git add RESULTS.md docs/
git commit -m "results: first run on the Stack Overflow corpus"
git push origin master
```

Pages serves `master` / `docs`, so the live page updates on push. Then change
the two lines in `README.md` that currently say the run has not happened: the
status table's last row, and the sample-data appendix note.

---

## Appendix A: the BERT rows on Colab

Worth doing once the linear grid is in, and only then — you want to know what
the cheap models say before you spend a GPU hour finding out whether the
expensive one beats them.

Runtime → Change runtime type → T4 GPU, then:

```python
!git clone https://github.com/yougijain/text-vs-timing.git
%cd text-vs-timing
!pip install -q -r requirements.txt
```

The corpus is gitignored, so upload `datasets/posts.csv` from the file pane
rather than re-fetching and burning another day's quota. The provenance sidecar
comes down with the clone, so it will still match.

```python
!python run_experiment.py --dataset datasets/posts.csv \
    --epochs 3 --max-length 256 --batch-size 16 \
    --embeddings --site
```

Defaults are already what you want: `bert-base-uncased`, temporal split, median
labels, AMP on. Add `--no-amp` if the loss goes to NaN. Download `RESULTS.md`,
`docs/index.html` and `outputs/` when it finishes, and commit them from your
laptop.

---

## Appendix B: what can go wrong

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
