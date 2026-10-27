"""Zero-shot LLM classification: the row a 2026 reviewer looks for first.

Run once, on claude-haiku-5-5 over the Stack Overflow split: 7,998 calls, one
refusal; the rows are in RESULTS.md. The tests still inject the classifier, so
nothing in the suite spends money. `estimate_cost()` is here so nobody plugs a
larger model in by accident.

Why the row belongs in the grid at all. The other three rows differ in how they
represent text while all learning the task from this corpus. A zero-shot LLM
learns the task from nothing -- it has never seen these labels -- so it splits a
different axis than the frozen-embedding row does:

    TF-IDF          lexical features      + task training
    Frozen MiniLM   pretrained semantics  + task training
    Fine-tuned BERT pretrained semantics  + task training, end to end
    Zero-shot LLM   pretrained semantics  + NO task training   <- this file

If the LLM matches models trained on thousands of labelled rows, the labels
carry less information than the corpus size suggests, and that is worth knowing
before anyone builds a labelling pipeline. If it loses badly, the task is
genuinely corpus-specific and the supervised rows have earned their keep.

The ablation still has to work here, and that is the interesting design
problem. "Text only" and "text + temporal" are not feature matrices for an LLM,
they are two different prompts -- the timestamp has to be *rendered into
English* for the model to use it at all. `build_prompt` does exactly that and
nothing else, which is why it is the part most worth reading.

Deliberately not done yet, so the next iteration has somewhere to go:
  * Few-shot examples selected from the training split (currently zero-shot).
  * The Batches API, which halves the cost on a few thousand rows.
  * Concurrency. The loop here is serial and slow on purpose -- a fast loop
    that silently spends money is the wrong default for scaffolding.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from training.evaluate import compute_metrics, print_metrics

# Haiku 5.5 by default. A binary label over ~8,000 prompts is the job the
# small model is priced for: a full grid costs around a dollar on it and tens
# of dollars on Opus. `--llm-model` switches; the estimate below makes the
# difference visible before anything is spent.
DEFAULT_MODEL = "claude-haiku-5-5"

# Per-million-token rates, only for the pre-run cost estimate. They are a
# convenience, not a source of truth, and they go stale -- the estimate says so.
_PRICING_USD_PER_MTOK = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# No claim about how common each label is. An earlier version said the classes
# were "balanced by construction"; the median cut only searches for the most
# even split the scores allow, and on Stack Overflow that is 38/62. Telling the
# model otherwise pushes it toward 1 and below the majority baseline -- a
# number that would measure the prompt, not the model.
SYSTEM_PROMPT = (
    "You judge whether a forum post landed above or below typical engagement "
    "for its venue.\n"
    "Answer 1 if you expect it scored above the median for that venue, 0 if "
    "at or below.\n"
    "Do not assume the two answers are equally common."
)

# A JSON schema is what keeps this a classifier rather than a text generator.
# Without it the model is free to answer "Probably above average?", and a
# regex over prose is a silent accuracy bug waiting to happen.
LABEL_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "label": {"type": "integer", "enum": [0, 1]},
        },
        "required": ["label"],
        "additionalProperties": False,
    },
}

# Cached in place of a label when the model declines, so a resumed run does not
# pay to be refused twice.
REFUSED = -1

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday")


def render_timestamp(hour, day_of_week):
    """Turn a UTC timestamp into the sentence the model actually reads.

    The whole temporal ablation rests on this function. A feature matrix means
    nothing to an LLM; the clock has to become English or it is not in the
    prompt at all.

    "UTC" is stated explicitly rather than left implied. Without it the model
    reasonably assumes local time and reasons about the poster's body clock --
    which is precisely the inference this project says the data cannot support.
    Naming the timezone keeps the prompt honest about what it knows.
    """
    try:
        hour = int(hour)
        day = _WEEKDAYS[int(day_of_week) % 7]
    except (TypeError, ValueError, IndexError):
        return ""
    return f"Posted at {hour:02d}:00 UTC on a {day}."


def build_prompt(text, hour=None, day_of_week=None, use_temporal=False,
                 max_chars=4000):
    """Build the user turn. This is the ablation, in prompt form.

    Args:
        text (str): The post body.
        hour, day_of_week: UTC components, used only when use_temporal.
        use_temporal (bool): Include the rendered timestamp.
        max_chars (int): Truncation guard. A pathological post should cost a
            predictable amount, not an unbounded one.
    Returns:
        str: the user message.
    """
    body = ("" if text is None else str(text)).strip()
    if len(body) > max_chars:
        body = body[:max_chars] + " [truncated]"

    parts = []
    if use_temporal:
        stamp = render_timestamp(hour, day_of_week)
        if stamp:
            parts.append(stamp)
    parts.append(f"Post:\n{body}")
    return "\n\n".join(parts)


class PromptCache:
    """Disk cache keyed by (model, prompt). Re-runs are free.

    Not an optimisation. Without it, every re-run of the grid -- every seed
    change, every bug fix, every accidental double-invocation -- bills again
    for answers already paid for. A cache is what makes an LLM row safe to
    iterate on.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._entries = {}
        if self.path and self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                print(f"  WARNING: {self.path} is not valid JSON; ignoring it.")

    @staticmethod
    def key(model, prompt):
        digest = hashlib.sha256(f"{model}\x00{prompt}".encode("utf-8"))
        return digest.hexdigest()

    def get(self, model, prompt):
        return self._entries.get(self.key(model, prompt))

    def put(self, model, prompt, label):
        self._entries[self.key(model, prompt)] = int(label)

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=0) + "\n")

    def __len__(self):
        return len(self._entries)


def estimate_cost(prompts, model=DEFAULT_MODEL, output_tokens=150):
    """Roughly what one pass over `prompts` would cost, before running it.

    Chars/4 is a crude token proxy and the rates go stale; this exists to tell
    the difference between "a few cents" and "eighty dollars", which is the
    only distinction that changes a decision. `count_tokens` would be exact and
    is the right call once this row is run for real.

    The system prompt is sent with every request, so it is counted per prompt.
    `output_tokens` is mostly thinking, not the label: thinking is on by
    default on current models and is billed as output. 150 is a guess for a
    binary judgement at low effort; the first real run's usage replaces it.
    """
    input_tokens = sum(len(p) + len(SYSTEM_PROMPT) for p in prompts) / 4.0
    rates = _PRICING_USD_PER_MTOK.get(model)
    if rates is None:
        return {"model": model, "prompts": len(prompts), "usd": None,
                "note": f"No cached rate for {model}; cost unknown."}

    in_rate, out_rate = rates
    usd = (input_tokens / 1e6) * in_rate + (len(prompts) * output_tokens / 1e6) * out_rate
    return {
        "model": model,
        "prompts": len(prompts),
        "estimated_input_tokens": int(input_tokens),
        "usd": round(usd, 2),
        "note": "Rough: chars/4 token proxy, cached rates, no caching or batch "
                "discount applied. Halve it if you route through the Batches API.",
    }


def build_anthropic_classifier(model=DEFAULT_MODEL, system=SYSTEM_PROMPT):
    """Return `prompt -> 0|1|None`, backed by the Messages API.

    Exercised for real once (claude-haiku-5-5, anthropic 2.3.0). The tests
    never call it; they inject a classifier instead.

    Notes on the call shape, since they are the parts that cost money or
    silently degrade:
      * `output_config.format` pins a JSON schema, so the answer is a label
        rather than a sentence that has to be parsed with a regex.
      * `effort: "low"` -- this is a binary judgement, not a reasoning task,
        and effort is the main spend dial. Thinking is left at its default
        (on, adaptive) rather than configured, which keeps the call valid
        across Haiku, Sonnet and Opus: Opus 5.5 rejects disabled thinking.
      * `max_tokens=2048`. The label is one integer, but thinking counts
        against the same limit; 256 could cut a call off before the label.
        Only generated tokens are billed, so the headroom costs nothing.
      * No prompt caching. The system prompt is far below the minimum
        cacheable prefix, so a cache_control marker would silently do nothing.
      * A refusal returns None rather than raising. One declined post would
        otherwise stop the run on the same row every time it is resumed.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The LLM row needs the anthropic package: pip install anthropic\n"
            "It is deliberately not in requirements.txt -- this row is opt-in "
            "and costs money to run."
        ) from exc

    client = anthropic.Anthropic()

    def classify(prompt):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            output_config={"effort": "low", "format": LABEL_SCHEMA},
            messages=[{"role": "user", "content": prompt}],
        )

        if response.stop_reason == "refusal":
            # Refusals are HTTP 200. Reading .content without checking would
            # silently turn a declined row into a guess; the caller decides
            # what a declined row becomes, and counts it.
            return None

        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                "The model ran out of tokens before giving a label "
                f"(max_tokens=2048, {response.usage.output_tokens} used). "
                "Thinking used the budget; raise max_tokens."
            )

        text = next((b.text for b in response.content if b.type == "text"), "")
        return int(json.loads(text)["label"])

    return classify


class LlmBaseline:
    """Zero-shot LLM classifier, shaped like the other baselines.

    Mirrors TfidfBaseline / EmbeddingBaseline so benchmark.py can treat every
    row identically and the metrics come out of the same compute_metrics.

    `fit` is a genuine no-op: there is nothing to train. It exists so the shared
    `run_sklearn_baseline` plumbing works unchanged, and it records the class
    balance -- which is the one thing the training split legitimately tells a
    zero-shot model, and what a future few-shot version would draw examples from.

    Args:
        model (str): Model id.
        use_temporal_features (bool): Render the timestamp into the prompt.
        classify_fn (callable | None): `prompt -> 0|1`. Injected in tests and
            required in any environment without credentials. When None, the
            real Messages API client is built lazily on first use.
        cache_path: Disk cache for prompt -> label.
        max_chars (int): Per-post truncation guard.
        dry_run (bool): Refuse to call the model; report the cost instead. The
            default, because the surprising failure for scaffolding is a bill.
    """

    def __init__(self, model=DEFAULT_MODEL, use_temporal_features=True,
                 classify_fn=None, cache_path=None, max_chars=4000,
                 dry_run=True):
        self.model = model
        self.use_temporal_features = use_temporal_features
        self.classify_fn = classify_fn
        # Whether the caller supplied the classifier, recorded at construction
        # rather than sniffed from the function's name later: the real API
        # classifier is also called `classify`, so a name check cannot tell
        # them apart, and the results file would misreport what ran.
        self._injected = classify_fn is not None
        self.max_chars = max_chars
        self.dry_run = dry_run
        self.cache = PromptCache(cache_path)
        self._fitted = False
        self._train_balance = None
        self._api_calls = 0
        self._refusals = 0

    # -- shared-baseline interface -------------------------------------

    def fit(self, texts, labels, temporal_features=None, verbose=True):
        """No-op by construction: a zero-shot model has nothing to learn."""
        labels = np.asarray(labels)
        counts = np.bincount(labels.astype(int), minlength=2)
        self._train_balance = counts.tolist()
        self._fitted = True
        if verbose:
            print(f"  Zero-shot: no training. Train split balance {counts.tolist()} "
                  "is recorded only for the write-up.")
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before predicting.")

    def _prompts(self, texts, temporal_features=None):
        """Build one prompt per row, honouring the ablation arm."""
        if self.use_temporal_features and temporal_features is None:
            raise ValueError(
                "use_temporal_features=True but no temporal_features were "
                "passed. Provide them, or construct with "
                "use_temporal_features=False for the text-only arm."
            )

        hours = days = None
        if self.use_temporal_features:
            features = np.asarray(temporal_features)
            if len(features) != len(texts):
                raise ValueError(
                    f"Got {len(features)} temporal feature rows for {len(texts)} "
                    "texts. They must align."
                )
            hours, days = self._recover_clock(features)

        return [
            build_prompt(
                text,
                hour=None if hours is None else hours[i],
                day_of_week=None if days is None else days[i],
                use_temporal=self.use_temporal_features,
                max_chars=self.max_chars,
            )
            for i, text in enumerate(texts)
        ]

    @staticmethod
    def _recover_clock(features):
        """Invert the cyclical encoding back to an hour and a weekday.

        The pipeline hands every model the same (sin, cos) matrix, and the
        other rows consume it directly. An LLM needs the clock back in human
        terms, so this undoes the encoding rather than plumbing a second,
        differently-shaped input through the whole stack -- which would make
        this row's data path differ from the others, and a row prepared
        differently is not a comparison.

        Column order matches TEMPORAL_FEATURES: hour_sin, hour_cos, dow_sin,
        dow_cos, ...
        """
        if features.ndim != 2 or features.shape[1] < 4:
            return None, None
        hour = np.round(
            np.arctan2(features[:, 0], features[:, 1]) / (2 * np.pi) * 24
        ).astype(int) % 24
        day = np.round(
            np.arctan2(features[:, 2], features[:, 3]) / (2 * np.pi) * 7
        ).astype(int) % 7
        return hour, day

    def predict(self, texts, temporal_features=None, verbose=False):
        self._check_fitted()
        prompts = self._prompts(texts, temporal_features)

        classify = self.classify_fn
        if classify is None:
            if self.dry_run:
                estimate = estimate_cost(prompts, self.model)
                raise RuntimeError(
                    "LlmBaseline is in dry-run mode and has no classify_fn, so "
                    "nothing was sent.\n"
                    f"  A real pass over {estimate['prompts']} prompts would "
                    f"cost roughly ${estimate['usd']} on {estimate['model']}.\n"
                    f"  {estimate['note']}\n"
                    "  Pass dry_run=False to call the API, or classify_fn=... "
                    "to supply your own."
                )
            classify = build_anthropic_classifier(self.model)
            self.classify_fn = classify
            self._injected = False

        # A declined post gets the training split's majority class -- the
        # answer a model that knows nothing would give -- and is counted, so
        # the write-up can say how many rows were not the model's own judgement.
        fallback = int(np.argmax(self._train_balance))
        self._refusals = 0  # per pass: evaluate() re-predicts the same rows

        labels = []
        for index, prompt in enumerate(prompts):
            cached = self.cache.get(self.model, prompt)
            if cached is None:
                answer = classify(prompt)
                cached = REFUSED if answer is None else int(answer)
                self.cache.put(self.model, prompt, cached)
                self._api_calls += 1
                # Save as it goes. A full grid is thousands of paid calls, and
                # a crash at row 5,000 should not bill rows 1 to 4,999 again.
                if self._api_calls % 100 == 0:
                    self.cache.save()
            if cached == REFUSED:
                self._refusals += 1
                cached = fallback
            labels.append(cached)
            if verbose and (index + 1) % 100 == 0:
                print(f"  {index + 1}/{len(prompts)} classified "
                      f"({self._api_calls} billed)")

        self.cache.save()
        return np.asarray(labels, dtype="int64")

    def evaluate(self, texts, labels, temporal_features=None, verbose=True,
                 class_names=None):
        """Score with the same compute_metrics every other row uses.

        No predict_proba: a schema-constrained label carries no calibrated
        confidence, so `loss` is reported as 0.0 and the calibration section of
        the error analysis does not apply to this row. Inventing a probability
        from a logprob-free answer would put a number in the report that means
        nothing.
        """
        self._check_fitted()
        predictions = self.predict(texts, temporal_features, verbose=verbose)
        metrics = compute_metrics(labels, predictions, loss=0.0)
        metrics["llm_api_calls"] = self._api_calls
        metrics["llm_cache_hits"] = len(predictions) - self._api_calls
        metrics["llm_refusals"] = self._refusals
        if verbose:
            print_metrics(metrics, labels, predictions, class_names,
                          header=f"Zero-shot LLM ({self.model})")
        return metrics

    @property
    def num_features(self):
        """Zero. A zero-shot model fits no parameters to this corpus."""
        self._check_fitted()
        return 0

    def describe_model(self):
        """What actually produced the labels, for the results file.

        A results file naming a model that never ran is worse than one naming
        none: it survives into the write-up and nobody can tell.
        """
        if self._injected:
            name = getattr(self.classify_fn, "__name__", type(self.classify_fn).__name__)
            return f"injected classify_fn ({name})"
        return self.model
