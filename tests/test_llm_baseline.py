"""LLM baseline tests. The classifier is injected, so nothing here spends money.

Two things carry real risk: the cyclical-encoding inversion (silently wrong
hours would make the temporal arm meaningless while still producing a plausible
number), and the dry-run guard (its whole job is to stop an accidental bill).
"""

import json

import numpy as np
import pytest

from models.llm_baseline import (
    DEFAULT_MODEL,
    LABEL_SCHEMA,
    SYSTEM_PROMPT,
    LlmBaseline,
    PromptCache,
    build_prompt,
    estimate_cost,
    render_timestamp,
)
from utils.time_utils import _cyclical


def always(label):
    def classify(prompt):
        return label
    return classify


def by_keyword(keyword="good"):
    """Content-dependent, so a fit test asserts something real."""
    def classify(prompt):
        return 1 if keyword in prompt else 0
    return classify


@pytest.fixture
def texts():
    return [f"post {i} is good" if i % 2 else f"post {i}" for i in range(20)]


@pytest.fixture
def labels(texts):
    return np.array([1 if "good" in t else 0 for t in texts])


@pytest.fixture
def temporal(texts):
    """A real temporal matrix, built the way the pipeline builds one."""
    hours = np.arange(len(texts)) % 24
    days = np.arange(len(texts)) % 7
    hour_sin, hour_cos = _cyclical(hours, 24)
    dow_sin, dow_cos = _cyclical(days, 7)
    rest = np.zeros((len(texts), 4))
    return np.column_stack([hour_sin, hour_cos, dow_sin, dow_cos, rest])


class TestRenderTimestamp:
    def test_states_utc_explicitly(self):
        # Without "UTC" the model reasons about the poster's body clock, which
        # is the exact inference this project says the data cannot support.
        assert "UTC" in render_timestamp(3, 1)

    def test_renders_hour_and_weekday(self):
        assert render_timestamp(3, 1) == "Posted at 03:00 UTC on a Tuesday."

    def test_monday_is_day_zero(self):
        assert "Monday" in render_timestamp(9, 0)

    def test_day_index_wraps(self):
        assert render_timestamp(9, 7) == render_timestamp(9, 0)

    def test_unusable_input_renders_nothing(self):
        assert render_timestamp(None, None) == ""
        assert render_timestamp("x", 1) == ""


class TestBuildPrompt:
    def test_text_only_arm_omits_the_clock(self):
        prompt = build_prompt("hello", hour=3, day_of_week=1, use_temporal=False)
        assert "UTC" not in prompt
        assert "hello" in prompt

    def test_temporal_arm_includes_the_clock(self):
        prompt = build_prompt("hello", hour=3, day_of_week=1, use_temporal=True)
        assert "03:00 UTC" in prompt
        assert "hello" in prompt

    def test_the_two_arms_differ(self):
        # If they did not, the ablation would measure nothing.
        assert build_prompt("x", 3, 1, use_temporal=True) != \
               build_prompt("x", 3, 1, use_temporal=False)

    def test_long_posts_are_truncated(self):
        prompt = build_prompt("a" * 9000, max_chars=100)
        assert "[truncated]" in prompt
        assert len(prompt) < 400

    def test_missing_clock_degrades_to_text_only(self):
        prompt = build_prompt("hello", hour=None, day_of_week=None, use_temporal=True)
        assert "UTC" not in prompt
        assert "hello" in prompt

    def test_none_text_does_not_raise(self):
        assert "Post:" in build_prompt(None)


class TestRecoverClock:
    """The inversion is real maths and a silent error here is invisible."""

    def test_round_trips_every_hour_and_weekday(self, ):
        hours = np.repeat(np.arange(24), 7)
        days = np.tile(np.arange(7), 24)
        hour_sin, hour_cos = _cyclical(hours, 24)
        dow_sin, dow_cos = _cyclical(days, 7)
        features = np.column_stack([hour_sin, hour_cos, dow_sin, dow_cos])

        got_hours, got_days = LlmBaseline._recover_clock(features)
        assert np.array_equal(got_hours, hours)
        assert np.array_equal(got_days, days)

    def test_too_few_columns_returns_nothing(self):
        assert LlmBaseline._recover_clock(np.zeros((3, 2))) == (None, None)

    def test_one_dimensional_input_returns_nothing(self):
        assert LlmBaseline._recover_clock(np.zeros(4)) == (None, None)


class TestPromptCache:
    def test_round_trips_through_disk(self, tmp_path):
        path = tmp_path / "cache.json"
        cache = PromptCache(path)
        cache.put("m", "prompt", 1)
        cache.save()

        assert PromptCache(path).get("m", "prompt") == 1

    def test_keys_are_model_scoped(self, tmp_path):
        cache = PromptCache(tmp_path / "c.json")
        cache.put("model-a", "prompt", 1)
        assert cache.get("model-b", "prompt") is None

    def test_missing_entry_is_none(self):
        assert PromptCache().get("m", "nope") is None

    def test_corrupt_cache_is_ignored_not_fatal(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("{not json")
        assert len(PromptCache(path)) == 0

    def test_pathless_cache_still_works_in_memory(self):
        cache = PromptCache()
        cache.put("m", "p", 0)
        cache.save()  # no-op, must not raise
        assert cache.get("m", "p") == 0


class TestEstimateCost:
    def test_scales_with_prompt_volume(self):
        small = estimate_cost(["x" * 100] * 10)
        large = estimate_cost(["x" * 100] * 1000)
        assert large["usd"] > small["usd"]

    def test_cheaper_model_costs_less(self):
        prompts = ["x" * 1000] * 100
        assert estimate_cost(prompts, "claude-haiku-4-5")["usd"] < \
               estimate_cost(prompts, "claude-opus-5")["usd"]

    def test_unknown_model_reports_unknown_rather_than_guessing(self):
        estimate = estimate_cost(["x"], "some-future-model")
        assert estimate["usd"] is None
        assert "unknown" in estimate["note"].lower()

    def test_states_that_it_is_rough(self):
        assert "Rough" in estimate_cost(["x"])["note"]

    def test_the_default_model_has_a_price(self):
        assert estimate_cost(["x"])["usd"] is not None

    def test_the_system_prompt_is_billed_on_every_call(self):
        # It is sent with each request; an estimate over empty posts is not 0.
        assert estimate_cost([""] * 1000)["estimated_input_tokens"] > 0


class TestPrompt:
    def test_does_not_claim_the_labels_are_balanced(self):
        # On Stack Overflow they are 38/62. Saying otherwise pushes the model
        # toward 1 and below the majority baseline.
        assert "balanced" not in SYSTEM_PROMPT
        assert "equally common" in SYSTEM_PROMPT


class TestDryRun:
    """The guard exists to stop an accidental bill; test it like it matters."""

    def test_dry_run_refuses_and_quotes_a_price(self, texts, labels):
        model = LlmBaseline(use_temporal_features=False, dry_run=True)
        model.fit(texts, labels, verbose=False)
        with pytest.raises(RuntimeError) as excinfo:
            model.predict(texts)

        message = str(excinfo.value)
        assert "dry-run" in message
        assert "$" in message
        assert "classify_fn" in message

    def test_injected_classifier_bypasses_the_guard(self, texts, labels):
        model = LlmBaseline(use_temporal_features=False,
                            classify_fn=always(1), dry_run=True)
        model.fit(texts, labels, verbose=False)
        assert model.predict(texts).tolist() == [1] * len(texts)


class TestLlmBaseline:
    def test_fit_trains_nothing_but_records_balance(self, texts, labels):
        model = LlmBaseline(use_temporal_features=False, classify_fn=always(0))
        model.fit(texts, labels, verbose=False)
        assert model.num_features == 0
        assert sum(model._train_balance) == len(labels)

    def test_predicting_before_fitting_raises(self):
        model = LlmBaseline(classify_fn=always(0))
        with pytest.raises(RuntimeError, match="Call fit"):
            model.predict(["x"])

    def test_predictions_follow_the_prompt_content(self, texts, labels):
        model = LlmBaseline(use_temporal_features=False,
                            classify_fn=by_keyword("good"))
        model.fit(texts, labels, verbose=False)
        assert np.array_equal(model.predict(texts), labels)

    def test_repeated_rows_are_billed_once(self, labels):
        calls = {"n": 0}

        def counting(prompt):
            calls["n"] += 1
            return 1

        repeated = ["same post"] * 10
        model = LlmBaseline(use_temporal_features=False, classify_fn=counting)
        model.fit(repeated, np.ones(10, dtype=int), verbose=False)
        model.predict(repeated)
        assert calls["n"] == 1

    def test_a_refusal_becomes_the_majority_class_and_is_counted(self, texts):
        def refuses_good(prompt):
            return None if "good" in prompt else 1

        # Train balance is mostly 0, so a declined row is predicted 0.
        model = LlmBaseline(use_temporal_features=False, classify_fn=refuses_good)
        model.fit(texts, np.zeros(len(texts), dtype=int), verbose=False)
        predictions = model.predict(texts)
        declined = ["good" in t for t in texts]
        assert all(p == 0 for p, d in zip(predictions, declined) if d)
        metrics = model.evaluate(texts, np.zeros(len(texts), dtype=int),
                                 verbose=False)
        assert metrics["llm_refusals"] == sum(declined)

    def test_a_refusal_is_not_billed_again_on_resume(self, tmp_path, texts):
        calls = {"n": 0}

        def refuses(prompt):
            calls["n"] += 1
            return None

        cache = tmp_path / "cache.json"
        for _ in range(2):
            model = LlmBaseline(use_temporal_features=False, classify_fn=refuses,
                                cache_path=cache)
            model.fit(texts, np.zeros(len(texts), dtype=int), verbose=False)
            model.predict(texts)
        assert calls["n"] == len(set(texts))

    def test_a_crash_mid_run_keeps_what_was_paid_for(self, tmp_path):
        """Thousands of paid calls: a failure at row 150 must not lose 1-149."""
        def dies_at_150(prompt):
            dies_at_150.n += 1
            if dies_at_150.n > 150:
                raise ConnectionError("network went away")
            return 1
        dies_at_150.n = 0

        posts = [f"post {i}" for i in range(300)]
        cache = tmp_path / "cache.json"
        model = LlmBaseline(use_temporal_features=False, classify_fn=dies_at_150,
                            cache_path=cache)
        model.fit(posts, np.ones(300, dtype=int), verbose=False)
        with pytest.raises(ConnectionError):
            model.predict(posts)
        assert len(json.loads(cache.read_text())) >= 100

    def test_temporal_arm_changes_the_prompts(self, texts, labels, temporal):
        seen = []

        def record(prompt):
            seen.append(prompt)
            return 0

        model = LlmBaseline(use_temporal_features=True, classify_fn=record)
        model.fit(texts, labels, temporal, verbose=False)
        model.predict(texts, temporal)
        assert all("UTC" in prompt for prompt in seen)

    def test_missing_temporal_features_raise(self, texts, labels):
        model = LlmBaseline(use_temporal_features=True, classify_fn=always(0))
        model.fit(texts, labels, np.zeros((len(texts), 8)), verbose=False)
        with pytest.raises(ValueError, match="no temporal_features"):
            model.predict(texts)

    def test_misaligned_temporal_features_raise(self, texts, labels):
        model = LlmBaseline(use_temporal_features=True, classify_fn=always(0))
        model.fit(texts, labels, np.zeros((len(texts), 8)), verbose=False)
        with pytest.raises(ValueError, match="must align"):
            model.predict(texts, np.zeros((3, 8)))

    def test_evaluate_matches_the_shared_metric_schema(self, texts, labels):
        model = LlmBaseline(use_temporal_features=False,
                            classify_fn=by_keyword("good"))
        model.fit(texts, labels, verbose=False)
        metrics = model.evaluate(texts, labels, verbose=False)

        for key in ("accuracy", "macro_f1", "majority_baseline",
                    "lift_over_baseline", "loss"):
            assert key in metrics
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_evaluate_reports_what_was_billed(self, texts, labels):
        model = LlmBaseline(use_temporal_features=False, classify_fn=always(1))
        model.fit(texts, labels, verbose=False)
        metrics = model.evaluate(texts, labels, verbose=False)
        assert metrics["llm_api_calls"] + metrics["llm_cache_hits"] == len(texts)

    def test_loss_is_zero_because_there_is_no_probability(self, texts, labels):
        # A schema-constrained label carries no calibrated confidence. Better a
        # stated zero than an invented number in the report.
        model = LlmBaseline(use_temporal_features=False, classify_fn=always(1))
        model.fit(texts, labels, verbose=False)
        assert model.evaluate(texts, labels, verbose=False)["loss"] == 0.0


class TestDescribeModel:
    def test_reports_the_model_when_calling_the_api(self):
        assert LlmBaseline().describe_model() == DEFAULT_MODEL

    def test_reports_the_injection_in_tests(self):
        described = LlmBaseline(classify_fn=by_keyword()).describe_model()
        assert "injected" in described
        assert DEFAULT_MODEL not in described


class TestSchema:
    def test_label_schema_admits_only_zero_and_one(self):
        properties = LABEL_SCHEMA["schema"]["properties"]
        assert properties["label"]["enum"] == [0, 1]
        assert LABEL_SCHEMA["schema"]["additionalProperties"] is False

    def test_schema_is_json_serialisable(self):
        assert json.loads(json.dumps(LABEL_SCHEMA)) == LABEL_SCHEMA
