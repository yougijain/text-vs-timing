"""The TF-IDF baseline and the benchmark grid."""

import json

import numpy as np
import pytest

import benchmark
import main as pipeline
from data.data_loader import split_indices
from data.preprocess import build_labels
from models.tfidf_baseline import TfidfBaseline
from training.config import TEMPORAL_FEATURES, Config
from training.evaluate import compute_metrics
from utils.time_utils import add_temporal_features, temporal_feature_matrix


@pytest.fixture(scope="module")
def dataset(request):
    """Cleaned text, labels, and temporal features from the shared sample frame."""
    clean = request.getfixturevalue("clean_frame")
    framed = add_temporal_features(clean)
    labels, _ = build_labels(framed, "median")
    features = temporal_feature_matrix(framed, TEMPORAL_FEATURES)
    texts = framed["clean_text"].tolist()
    train_idx, val_idx = split_indices(len(framed), 0.2, "random", seed=42)
    return {
        "texts": texts, "labels": labels, "features": features,
        "train": train_idx, "val": val_idx,
    }


def _fit(dataset, **kwargs):
    model = TfidfBaseline(**kwargs)
    idx = dataset["train"]
    features = dataset["features"][idx] if model.use_temporal_features else None
    model.fit([dataset["texts"][i] for i in idx], dataset["labels"][idx], features)
    return model


class TestFitAndPredict:
    def test_predictions_are_valid_labels(self, dataset):
        model = _fit(dataset)
        idx = dataset["val"]
        preds = model.predict([dataset["texts"][i] for i in idx], dataset["features"][idx])
        assert len(preds) == len(idx)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_probabilities_sum_to_one(self, dataset):
        model = _fit(dataset)
        idx = dataset["val"]
        probs = model.predict_proba([dataset["texts"][i] for i in idx],
                                    dataset["features"][idx])
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_beats_the_majority_baseline(self, dataset):
        # If the linear model cannot beat "always answer the bigger class" on
        # data with a planted signal, something upstream is broken.
        model = _fit(dataset)
        idx = dataset["val"]
        metrics = model.evaluate([dataset["texts"][i] for i in idx],
                                 dataset["labels"][idx], dataset["features"][idx],
                                 verbose=False)
        assert metrics["lift_over_baseline"] > 0

    def test_temporal_features_help_on_this_data(self, dataset):
        # The sample generator plants a time-of-day signal, so the temporal
        # variant should win. If it stops winning, the features are not
        # reaching the model.
        idx = dataset["val"]
        texts = [dataset["texts"][i] for i in idx]

        with_temporal = _fit(dataset, use_temporal_features=True).evaluate(
            texts, dataset["labels"][idx], dataset["features"][idx], verbose=False)
        text_only = _fit(dataset, use_temporal_features=False).evaluate(
            texts, dataset["labels"][idx], verbose=False)
        assert with_temporal["accuracy"] > text_only["accuracy"]

    def test_is_deterministic(self, dataset):
        idx = dataset["val"]
        texts = [dataset["texts"][i] for i in idx]
        first = _fit(dataset).predict(texts, dataset["features"][idx])
        second = _fit(dataset).predict(texts, dataset["features"][idx])
        assert np.array_equal(first, second)

    def test_works_without_char_ngrams(self, dataset):
        model = _fit(dataset, use_char_ngrams=False)
        assert model.char_vectorizer is None
        idx = dataset["val"]
        assert len(model.predict([dataset["texts"][i] for i in idx],
                                 dataset["features"][idx])) == len(idx)


class TestErrorHandling:
    def test_predicting_before_fitting_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            TfidfBaseline().predict(["some text"])

    def test_single_class_training_data_raises(self):
        with pytest.raises(ValueError, match="single class"):
            TfidfBaseline(use_temporal_features=False).fit(["a b c", "d e f"], [1, 1])

    def test_missing_temporal_features_raises_a_clear_error(self, dataset):
        model = TfidfBaseline(use_temporal_features=True)
        with pytest.raises(ValueError, match="use_temporal_features=False"):
            model.fit(dataset["texts"][:20], dataset["labels"][:20])

    def test_misaligned_temporal_features_raise(self, dataset):
        model = TfidfBaseline(use_temporal_features=True)
        with pytest.raises(ValueError, match="align"):
            model.fit(dataset["texts"][:20], dataset["labels"][:20],
                      dataset["features"][:5])


class TestMetricsParity:
    def test_uses_the_same_metric_shape_as_the_neural_path(self, dataset):
        """Both models must report the same keys, or the table lies."""
        model = _fit(dataset)
        idx = dataset["val"]
        metrics = model.evaluate([dataset["texts"][i] for i in idx],
                                 dataset["labels"][idx], dataset["features"][idx],
                                 verbose=False)
        reference = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0])
        assert set(metrics) == set(reference)

    def test_compute_metrics_lift_definition(self):
        m = compute_metrics([0, 0, 1, 1], [0, 0, 1, 1])
        assert m["accuracy"] == 1.0
        assert m["lift_over_baseline"] == pytest.approx(0.5)

    def test_compute_metrics_on_empty_input(self):
        assert compute_metrics([], [])["num_samples"] == 0


class TestTopFeatures:
    def test_returns_signed_word_weights(self, dataset):
        top = _fit(dataset).top_features(n=5, temporal_feature_names=TEMPORAL_FEATURES)
        assert len(top["positive"]) == 5 and len(top["negative"]) == 5
        assert all(isinstance(name, str) for name, _ in top["positive"])
        # Positive-class features must outweigh negative-class ones.
        assert top["positive"][0][1] > top["negative"][0][1]

    def test_temporal_weights_are_named(self, dataset):
        top = _fit(dataset).top_features(temporal_feature_names=TEMPORAL_FEATURES)
        assert [name for name, _ in top["temporal"]] == TEMPORAL_FEATURES

    def test_hour_dominates_the_temporal_weights_on_sample_data(self, dataset):
        # The generator plants an hour-of-day effect, so the hour features
        # should carry more weight than the month features.
        top = _fit(dataset).top_features(temporal_feature_names=TEMPORAL_FEATURES)
        weights = dict(top["temporal"])
        hour = max(abs(weights["hour_sin"]), abs(weights["hour_cos"]))
        month = max(abs(weights["month_sin"]), abs(weights["month_cos"]))
        assert hour > month

    def test_no_temporal_key_when_disabled(self, dataset):
        assert "temporal" not in _fit(dataset, use_temporal_features=False).top_features()

    def test_num_features_counts_every_block(self, dataset):
        model = _fit(dataset)
        assert model.num_features > len(model.word_vectorizer.vocabulary_)


class TestMainIntegration:
    def test_tfidf_runs_through_main(self, tmp_path):
        from data.make_sample_data import generate_dataset
        csv = tmp_path / "sample.csv"
        generate_dataset(300, seed=21).to_csv(csv, index=False)

        results = pipeline.main([
            "--dataset", str(csv), "--model", "tfidf",
            "--split-strategy", "random", "--skip-analysis",
            "--output-dir", str(tmp_path / "outputs"),
        ])
        assert results["final_metrics"]["num_samples"] > 0
        assert "top_features" in results

    def test_both_models_get_the_same_split(self, tmp_path):
        """The comparison is void if the two halves differ."""
        from data.make_sample_data import generate_dataset
        csv = tmp_path / "sample.csv"
        generate_dataset(300, seed=23).to_csv(csv, index=False)
        config = Config(dataset_path=csv, split_strategy="random", seed=42)

        data, _, _, _ = pipeline.prepare_frame(config, run_analysis=False)
        first = split_indices(len(data), config.val_split, config.split_strategy,
                              config.seed)
        second = split_indices(len(data), config.val_split, config.split_strategy,
                               config.seed)
        assert np.array_equal(first[1], second[1])


class TestBenchmark:
    def test_tfidf_only_grid(self, tmp_path, monkeypatch):
        from data.make_sample_data import generate_dataset
        csv = tmp_path / "sample.csv"
        generate_dataset(400, seed=29).to_csv(csv, index=False)
        monkeypatch.chdir(tmp_path)

        rows = benchmark.main([
            "--dataset", str(csv), "--skip-bert", "--split-strategy", "random",
            "--out", str(tmp_path / "bench.json"),
        ])
        assert len(rows) == 2
        assert (tmp_path / "checkpoints").is_dir()  # beside --out, not in outputs/
        assert {r["features"] for r in rows} == {"text only", "text + temporal"}
        assert all(r["model"] == "TF-IDF" for r in rows)

    def test_every_row_reports_the_same_baseline(self, tmp_path, monkeypatch):
        from data.make_sample_data import generate_dataset
        csv = tmp_path / "sample.csv"
        generate_dataset(400, seed=31).to_csv(csv, index=False)
        monkeypatch.chdir(tmp_path)

        rows = benchmark.main([
            "--dataset", str(csv), "--skip-bert", "--split-strategy", "random",
            "--out", str(tmp_path / "bench.json"),
        ])
        baselines = {r["metrics"]["majority_baseline"] for r in rows}
        assert len(baselines) == 1

    def _grid_on_disk(self, tmp_path):
        """A TF-IDF grid plus a stand-in BERT row, as a GPU run would leave it."""
        from data.make_sample_data import generate_dataset
        csv = tmp_path / "sample.csv"
        generate_dataset(400, seed=47).to_csv(csv, index=False)
        out = tmp_path / "bench.json"
        benchmark.main(["--dataset", str(csv), "--skip-bert",
                        "--split-strategy", "random", "--out", str(out)])
        grid = json.loads(out.read_text())
        grid["rows"].append({"model": "BERT", "features": "text only",
                             "metrics": grid["rows"][0]["metrics"]})
        out.write_text(json.dumps(grid))
        return csv, out, grid["rows"][0]["metrics"]

    def test_llm_only_adds_to_the_grid_without_dropping_bert(self, tmp_path,
                                                             monkeypatch):
        csv, out, metrics = self._grid_on_disk(tmp_path)
        fake = [{"model": "Zero-shot test", "features": f, "metrics": metrics}
                for f in ("text only", "text + temporal")]
        monkeypatch.setattr(benchmark, "_llm_rows", lambda args: list(fake))
        argv = ["--dataset", str(csv), "--llm-only", "--split-strategy", "random",
                "--out", str(out)]

        rows = benchmark.main(argv)
        models = [r["model"] for r in rows]
        assert "BERT" in models and models.count("Zero-shot test") == 2
        assert len(rows) == 5
        # A second run replaces the LLM pair rather than appending another.
        assert len(benchmark.main(argv)) == 5

    def test_llm_only_refuses_a_different_split_before_billing(self, tmp_path,
                                                               monkeypatch):
        csv, out, _ = self._grid_on_disk(tmp_path)

        def billed(args):
            raise AssertionError("the API was reached")

        monkeypatch.setattr(benchmark, "_llm_rows", billed)
        with pytest.raises(SystemExit, match="Nothing was sent"):
            benchmark.main(["--dataset", str(csv), "--llm-only", "--max-rows",
                            "200", "--split-strategy", "random", "--out", str(out)])

    def test_print_table_handles_an_empty_grid(self, capsys):
        benchmark.print_table([])
        assert capsys.readouterr().out == ""
