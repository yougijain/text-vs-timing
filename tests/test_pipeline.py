"""End-to-end tests: the whole pipeline, offline, on synthetic data."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

import main as main_module
from analysis.pattern_detection import (
    add_marker_columns,
    compare_by_group,
    extract_markers,
    marker_label_correlations,
)
from analysis.timestamp_analysis import (
    plot_hourly_activity,
    plot_loss_curve,
    plot_weekly_heatmap,
    summarize_temporal,
)
from analysis.token_stats import token_length_stats
from data.data_loader import build_dataloaders
from data.make_sample_data import generate_dataset
from data.preprocess import batch_tokenize, build_labels
from models.bert_temporal_model import BertTemporalClassifier
from models.model_utils import set_seed
from training.evaluate import evaluate_model, predict
from training.train import build_optimizer, build_scheduler, train_model
from utils.loss_utils import moving_average
from utils.time_utils import add_temporal_features, temporal_feature_matrix
from training.config import TEMPORAL_FEATURES


class TestSampleDataGenerator:
    def test_has_the_expected_columns(self, raw_frame):
        for column in ("title", "selftext", "score", "created_utc"):
            assert column in raw_frame.columns

    def test_is_deterministic_for_a_seed(self):
        assert generate_dataset(50, seed=3).equals(generate_dataset(50, seed=3))

    def test_includes_the_junk_a_real_dump_has(self):
        frame = generate_dataset(300, seed=11)
        assert frame["selftext"].isin(["[removed]", "[deleted]"]).any()

    def test_timestamps_are_chronological(self, raw_frame):
        assert raw_frame["created_utc"].is_monotonic_increasing

    def test_the_planted_temporal_signal_is_present(self):
        # If this fails, the generator stopped producing a learnable signal and
        # every downstream ablation number is meaningless.
        from data.preprocess import clean_dataset
        frame = add_temporal_features(clean_dataset(generate_dataset(1500, seed=5)))
        labels, _ = build_labels(frame, "median")
        rates = pd.Series(labels).groupby(frame["is_late_night"].to_numpy()).mean()
        assert rates[1.0] < rates[0.0] - 0.1


class TestLinguisticMarkers:
    def test_every_marker_is_produced(self):
        markers = extract_markers("I always forget. Why can't I focus?")
        for key in ("first_person_rate", "negation_rate", "absolutist_rate",
                    "question_rate", "lexical_diversity", "word_count"):
            assert key in markers

    def test_empty_text_yields_zeros_not_nan(self):
        markers = extract_markers("")
        assert all(not np.isnan(v) for v in markers.values())
        assert markers["word_count"] == 0

    def test_rates_are_bounded(self):
        markers = extract_markers("i me my myself i me my")
        assert 0.0 <= markers["first_person_rate"] <= 1.0

    def test_question_rate_responds_to_questions(self):
        assert extract_markers("Why? How? When?")["question_rate"] > \
               extract_markers("It is fine. It is done.")["question_rate"]

    def test_comparison_sorts_by_gap_size(self, clean_frame):
        framed = add_marker_columns(add_temporal_features(clean_frame))
        comparison = compare_by_group(framed, "is_late_night")
        gaps = comparison["diff"].abs().to_numpy()
        assert np.all(np.diff(gaps) <= 1e-9)

    def test_constant_marker_correlates_zero_without_warning(self, clean_frame):
        framed = add_marker_columns(clean_frame)
        framed["constant_rate"] = 1.0
        labels, _ = build_labels(framed)
        correlations = marker_label_correlations(
            framed, labels, marker_columns=["constant_rate", "question_rate"]
        )
        assert correlations["constant_rate"] == 0.0

    def test_markers_require_cleaning_first(self, raw_frame):
        with pytest.raises(KeyError, match="clean_dataset"):
            add_marker_columns(raw_frame, text_column="clean_text")


class TestFigures:
    def test_hourly_activity_is_written(self, clean_frame, tmp_path):
        framed = add_temporal_features(clean_frame)
        labels, _ = build_labels(framed)
        path = plot_hourly_activity(framed, tmp_path, labels)
        assert path.exists() and path.stat().st_size > 0

    def test_weekly_heatmap_is_written(self, clean_frame, tmp_path):
        path = plot_weekly_heatmap(add_temporal_features(clean_frame), tmp_path)
        assert path.exists() and path.stat().st_size > 0

    def test_loss_curve_handles_a_series_shorter_than_the_window(self, tmp_path):
        assert plot_loss_curve([0.7, 0.6, 0.5], tmp_path, window=25).exists()

    def test_summarize_temporal_is_json_serialisable(self, clean_frame):
        framed = add_temporal_features(clean_frame)
        labels, _ = build_labels(framed)
        json.dumps(summarize_temporal(framed, labels))


class TestTrainingLoop:
    @pytest.fixture
    def loaders(self, local_tokenizer, clean_frame):
        framed = add_temporal_features(clean_frame)
        labels, _ = build_labels(framed)
        encodings = batch_tokenize(framed["clean_text"].tolist(),
                                   tokenizer=local_tokenizer, max_length=32, verbose=False)
        features = temporal_feature_matrix(framed, TEMPORAL_FEATURES)
        return build_dataloaders(encodings, labels, features, batch_size=8,
                                 strategy="random")

    @pytest.fixture
    def model(self, local_tokenizer):
        set_seed(0)
        return BertTemporalClassifier.tiny_for_testing(
            num_temporal_features=len(TEMPORAL_FEATURES),
            vocab_size=len(local_tokenizer), max_position_embeddings=64,
        )

    def test_training_runs_and_returns_history(self, loaders, model):
        train, val = loaders
        history = train_model(train, model, epochs=1, val_loader=val,
                              device=torch.device("cpu"), log_every=0)
        assert len(history["batch_losses"]) == len(train)
        assert len(history["val_metrics"]) == 1
        assert history["seconds"] >= 0

    def test_loss_decreases_over_several_epochs(self, loaders, model):
        train, _ = loaders
        optimizer = build_optimizer(model, learning_rate=1e-3)
        history = train_model(train, model, optimizer=optimizer, epochs=4,
                              device=torch.device("cpu"), log_every=0)
        assert history["epoch_losses"][-1] < history["epoch_losses"][0]

    def test_best_checkpoint_is_selected_on_validation(self, loaders, model, tmp_path):
        train, val = loaders
        path = tmp_path / "best.pth"
        history = train_model(train, model, epochs=2, val_loader=val,
                              save_path=path, device=torch.device("cpu"), log_every=0)
        assert path.exists()
        assert history["best_epoch"] in (1, 2)

    def test_amp_is_off_on_cpu(self, loaders, model):
        # Requesting AMP on a CPU box must not crash or silently pretend.
        train, _ = loaders
        history = train_model(train, model, epochs=1, use_amp=True,
                              device=torch.device("cpu"), log_every=0)
        assert len(history["batch_losses"]) > 0

    def test_scheduler_warms_up_then_decays(self, model):
        optimizer = build_optimizer(model, learning_rate=1e-3)
        scheduler = build_scheduler(optimizer, num_training_steps=100, warmup_ratio=0.1)
        seen = []
        for _ in range(100):
            seen.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        assert seen[0] < seen[10]        # warming up
        assert seen[10] > seen[-1]       # then decaying
        assert seen[-1] < 1e-4

    def test_optimizer_excludes_bias_and_layernorm_from_decay(self, model):
        groups = build_optimizer(model, weight_decay=0.01).param_groups
        assert {g["weight_decay"] for g in groups} == {0.01, 0.0}
        assert all(len(g["params"]) > 0 for g in groups)


class TestEvaluation:
    @pytest.fixture
    def trained(self, local_tokenizer, clean_frame):
        set_seed(0)
        framed = add_temporal_features(clean_frame)
        labels, _ = build_labels(framed)
        encodings = batch_tokenize(framed["clean_text"].tolist(),
                                   tokenizer=local_tokenizer, max_length=32, verbose=False)
        features = temporal_feature_matrix(framed, TEMPORAL_FEATURES)
        train, val = build_dataloaders(encodings, labels, features, batch_size=8,
                                       strategy="random")
        model = BertTemporalClassifier.tiny_for_testing(
            num_temporal_features=len(TEMPORAL_FEATURES),
            vocab_size=len(local_tokenizer), max_position_embeddings=64,
        )
        train_model(train, model, optimizer=build_optimizer(model, 1e-3), epochs=1,
                    device=torch.device("cpu"), log_every=0)
        return model, val

    def test_metrics_are_returned_not_just_printed(self, trained):
        model, val = trained
        metrics = evaluate_model(val, model, device=torch.device("cpu"), verbose=False)
        for key in ("accuracy", "macro_f1", "loss", "confusion_matrix",
                    "majority_baseline", "lift_over_baseline"):
            assert key in metrics

    def test_accuracy_is_a_probability(self, trained):
        model, val = trained
        metrics = evaluate_model(val, model, device=torch.device("cpu"), verbose=False)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_lift_is_accuracy_minus_baseline(self, trained):
        model, val = trained
        m = evaluate_model(val, model, device=torch.device("cpu"), verbose=False)
        assert m["lift_over_baseline"] == pytest.approx(
            m["accuracy"] - m["majority_baseline"]
        )

    def test_confusion_matrix_totals_match_the_sample_count(self, trained):
        model, val = trained
        m = evaluate_model(val, model, device=torch.device("cpu"), verbose=False)
        assert np.array(m["confusion_matrix"]).sum() == m["num_samples"]

    def test_predict_returns_probabilities_that_sum_to_one(self, trained):
        model, val = trained
        result = predict(model, val, device=torch.device("cpu"), return_attention=True)
        assert np.allclose(result["probabilities"].sum(axis=1), 1.0, atol=1e-5)
        assert "attention_weights" in result

    def test_empty_loader_does_not_crash(self, trained):
        from torch.utils.data import DataLoader
        model, val = trained
        empty = DataLoader(torch.utils.data.Subset(val.dataset, []), batch_size=4)
        assert evaluate_model(empty, model, device=torch.device("cpu"),
                              verbose=False)["num_samples"] == 0


class TestMainEntryPoint:
    def test_full_run_offline(self, tmp_path):
        """The whole pipeline, end to end, with no network access."""
        csv = tmp_path / "sample.csv"
        generate_dataset(300, seed=13).to_csv(csv, index=False)
        out = tmp_path / "outputs"

        results = main_module.main([
            "--dataset", str(csv), "--tiny-model", "--epochs", "1",
            "--batch-size", "16", "--max-length", "32", "--split-strategy", "random",
            "--skip-analysis", "--output-dir", str(out),
        ])
        assert results["final_metrics"]["num_samples"] > 0
        assert results["config"]["use_temporal_features"] is True
        assert len(results["history"]["batch_losses"]) > 0
        # A test run left in the real outputs/ gets picked up by the report.
        assert (out / "results.json").exists()
        assert (out / "figures" / "loss_curve.png").exists()

    def test_text_only_ablation_builds_no_temporal_branch(self, tmp_path):
        csv = tmp_path / "sample.csv"
        generate_dataset(200, seed=17).to_csv(csv, index=False)

        results = main_module.main([
            "--dataset", str(csv), "--tiny-model", "--epochs", "1",
            "--batch-size", "16", "--max-length", "32", "--no-temporal",
            "--split-strategy", "random", "--skip-analysis",
            "--output-dir", str(tmp_path / "outputs"),
        ])
        assert results["config"]["use_temporal_features"] is False

    def test_missing_dataset_names_the_path(self, tmp_path):
        from training.config import Config
        with pytest.raises(FileNotFoundError, match="--synthetic"):
            main_module.load_and_prepare_data(Config(dataset_path=tmp_path / "nope.csv"))


class TestTokenStats:
    def test_percentiles_and_truncation_rates(self, clean_frame, local_tokenizer):
        stats = token_length_stats(clean_frame["clean_text"].tolist(),
                                   tokenizer=local_tokenizer)
        assert stats["min"] <= stats["p50"] <= stats["max"]
        assert set(stats["truncation_rate"]) == {64, 128, 256, 512}

    def test_empty_input(self, local_tokenizer):
        assert token_length_stats([], tokenizer=local_tokenizer)["count"] == 0


class TestLossUtils:
    def test_moving_average_smooths(self):
        assert len(moving_average([1, 2, 3, 4, 5], window_size=3)) == 3

    def test_window_longer_than_data_returns_empty(self):
        assert len(moving_average([1.0, 2.0], window_size=10)) == 0
