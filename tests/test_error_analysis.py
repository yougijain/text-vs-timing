"""Error analysis: slicing, calibration, and model comparison."""

import json

import numpy as np
import pandas as pd
import pytest

from analysis.error_analysis import (
    MIN_SLICE_COUNT,
    build_error_frame,
    calibration_table,
    compare_feature_sets,
    compare_predictions,
    error_rate_by_slice,
    error_summary,
    plot_calibration,
    plot_error_rate_by_hour,
    print_report,
    worst_errors,
    worst_slices,
)


@pytest.fixture
def rows():
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame({
        "clean_text": [f"post number {i} with some words" for i in range(n)],
        "hour": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "is_late_night": rng.integers(0, 2, n).astype(float),
        "word_count": rng.integers(5, 200, n).astype(float),
    })


@pytest.fixture
def frame(rows):
    rng = np.random.default_rng(1)
    n = len(rows)
    y_true = rng.integers(0, 2, n)
    y_pred = y_true.copy()
    y_pred[:24] = 1 - y_pred[:24]  # 20% error rate
    prob = rng.uniform(0.5, 1.0, n)
    probabilities = np.column_stack([1 - prob, prob])
    return build_error_frame(rows, y_true, y_pred, probabilities)


class TestBuildErrorFrame:
    def test_adds_the_expected_columns(self, frame):
        for column in ("y_true", "y_pred", "correct", "error_type",
                       "prob_positive", "confidence"):
            assert column in frame.columns

    def test_keeps_the_source_columns(self, frame):
        assert "clean_text" in frame.columns and "hour" in frame.columns

    def test_correct_flag_matches(self, frame):
        assert (frame["correct"] == (frame["y_true"] == frame["y_pred"])).all()

    def test_error_types_are_consistent(self, frame):
        fp = frame[frame["error_type"] == "false_positive"]
        assert ((fp["y_true"] == 0) & (fp["y_pred"] == 1)).all()
        fn = frame[frame["error_type"] == "false_negative"]
        assert ((fn["y_true"] == 1) & (fn["y_pred"] == 0)).all()

    def test_confidence_is_at_least_half(self, frame):
        # For a binary classifier, confidence in the predicted class cannot
        # fall below 0.5 -- if it does, the sign convention is wrong.
        assert (frame["confidence"] >= 0.5).all()
        assert (frame["confidence"] <= 1.0).all()

    def test_works_without_probabilities(self, rows):
        built = build_error_frame(rows, np.zeros(len(rows)), np.zeros(len(rows)))
        assert "confidence" not in built.columns
        assert built["correct"].all()

    def test_accepts_1d_probabilities(self, rows):
        n = len(rows)
        built = build_error_frame(rows, np.zeros(n), np.zeros(n),
                                  np.full(n, 0.3))
        assert built["prob_positive"].iloc[0] == pytest.approx(0.3)
        assert built["confidence"].iloc[0] == pytest.approx(0.7)

    def test_length_mismatch_raises(self, rows):
        with pytest.raises(ValueError, match="Length mismatch"):
            build_error_frame(rows, np.zeros(5), np.zeros(5))

    def test_does_not_mutate_the_input(self, rows):
        before = list(rows.columns)
        build_error_frame(rows, np.zeros(len(rows)), np.zeros(len(rows)))
        assert list(rows.columns) == before


class TestErrorSummary:
    def test_counts_add_up(self, frame):
        summary = error_summary(frame)
        total = (summary["true_positive"] + summary["true_negative"]
                 + summary["false_positive"] + summary["false_negative"])
        assert total == summary["num_samples"]

    def test_error_rate_matches_the_frame(self, frame):
        summary = error_summary(frame)
        assert summary["error_rate"] == pytest.approx(1 - frame["correct"].mean())

    def test_detects_one_sided_errors(self, rows):
        # Every mistake in the same direction: a bias problem, not an
        # accuracy problem, and the report must distinguish them.
        n = len(rows)
        y_true = np.zeros(n, dtype=int)
        y_pred = np.zeros(n, dtype=int)
        y_pred[:30] = 1  # all false positives
        summary = error_summary(build_error_frame(rows, y_true, y_pred))
        assert summary["one_sided"] is True
        assert summary["false_positive_share"] == 1.0

    def test_balanced_errors_are_not_flagged(self, rows):
        n = len(rows)
        y_true = np.array([0, 1] * (n // 2))
        y_pred = y_true.copy()
        y_pred[:10] = 1 - y_pred[:10]  # 5 each way
        assert error_summary(build_error_frame(rows, y_true, y_pred))["one_sided"] is False

    def test_perfect_predictions(self, rows):
        n = len(rows)
        summary = error_summary(build_error_frame(rows, np.zeros(n), np.zeros(n)))
        assert summary["num_errors"] == 0
        assert summary["error_skew"] == 0.0


class TestSlicing:
    def test_counts_sum_to_the_frame(self, frame):
        table = error_rate_by_slice(frame, "is_late_night")
        assert table["count"].sum() == len(frame)

    def test_binned_slice_produces_labelled_buckets(self, frame):
        table = error_rate_by_slice(frame, "hour",
                                    bins=[-0.5, 11.5, 23.5], labels=["am", "pm"])
        assert set(table.index.astype(str)) == {"am", "pm"}

    def test_small_buckets_are_marked_unreliable(self, rows):
        rows = rows.copy()
        rows["tiny_group"] = [0] * (len(rows) - 3) + [1, 1, 1]
        n = len(rows)
        built = build_error_frame(rows, np.zeros(n), np.zeros(n))
        table = error_rate_by_slice(built, "tiny_group")
        assert table.loc[1, "count"] == 3
        assert not bool(table.loc[1, "reliable"])
        assert bool(table.loc[0, "reliable"])

    def test_missing_column_raises(self, frame):
        with pytest.raises(KeyError, match="nonexistent"):
            error_rate_by_slice(frame, "nonexistent")

    def test_worst_slices_excludes_tiny_buckets(self, rows):
        # A 2-row bucket at 100% error must never top the ranking.
        rows = rows.copy()
        rows["day_of_week"] = [0] * (len(rows) - 2) + [6, 6]
        n = len(rows)
        y_true = np.zeros(n, dtype=int)
        y_pred = np.zeros(n, dtype=int)
        y_pred[-2:] = 1  # only the 2-row bucket is wrong
        worst = worst_slices(build_error_frame(rows, y_true, y_pred))
        assert "6" not in worst["bucket"].tolist()

    def test_worst_slices_is_sorted(self, frame):
        worst = worst_slices(frame, top_n=5)
        rates = worst["error_rate"].to_numpy()
        assert np.all(np.diff(rates) <= 1e-9)

    def test_worst_slices_respects_min_count(self, frame):
        worst = worst_slices(frame, top_n=10)
        assert (worst["count"] >= MIN_SLICE_COUNT).all()

    def test_worst_slices_on_a_perfect_model(self, rows):
        n = len(rows)
        worst = worst_slices(build_error_frame(rows, np.zeros(n), np.zeros(n)))
        assert (worst["error_rate"] == 0).all() or worst.empty


class TestWorstErrors:
    def test_returns_only_mistakes(self, frame):
        errors = worst_errors(frame, n=10)
        assert not errors.empty
        assert (errors["y_true"] != errors["y_pred"]).all()

    def test_sorted_by_confidence(self, frame):
        confidences = worst_errors(frame, n=10)["confidence"].to_numpy()
        assert np.all(np.diff(confidences) <= 1e-9)

    def test_empty_when_nothing_is_wrong(self, rows):
        n = len(rows)
        assert worst_errors(build_error_frame(rows, np.zeros(n), np.zeros(n))).empty


class TestCalibration:
    def test_perfect_calibration_scores_near_zero(self, rows):
        # Confidence c, right exactly c of the time, in every bin.
        rng = np.random.default_rng(3)
        n = 4000
        confidence = rng.uniform(0.5, 1.0, n)
        correct = rng.random(n) < confidence
        y_true = np.zeros(n, dtype=int)
        y_pred = np.where(correct, 0, 1)
        data = pd.DataFrame({"clean_text": ["x"] * n})
        frame = build_error_frame(data, y_true, y_pred,
                                  np.column_stack([confidence, 1 - confidence]))
        _, ece = calibration_table(frame)
        assert ece < 0.05

    def test_overconfident_model_scores_badly(self, rows):
        # Always 99% sure, right half the time.
        n = 400
        rng = np.random.default_rng(4)
        y_true = rng.integers(0, 2, n)
        y_pred = rng.integers(0, 2, n)
        data = pd.DataFrame({"clean_text": ["x"] * n})
        frame = build_error_frame(data, y_true, y_pred,
                                  np.column_stack([np.full(n, 0.01), np.full(n, 0.99)]))
        _, ece = calibration_table(frame)
        assert ece > 0.3

    def test_table_columns(self, frame):
        table, _ = calibration_table(frame, bins=5)
        for column in ("count", "mean_confidence", "accuracy", "gap"):
            assert column in table.columns

    def test_requires_probabilities(self, rows):
        n = len(rows)
        built = build_error_frame(rows, np.zeros(n), np.zeros(n))
        with pytest.raises(KeyError, match="y_prob"):
            calibration_table(built)


class TestComparePredictions:
    @pytest.fixture
    def pair(self, rows):
        n = len(rows)
        rng = np.random.default_rng(5)
        y_true = rng.integers(0, 2, n)
        a, b = y_true.copy(), y_true.copy()
        a[:20] = 1 - a[:20]
        b[10:35] = 1 - b[10:35]
        return (build_error_frame(rows, y_true, a),
                build_error_frame(rows, y_true, b))

    def test_cells_sum_to_the_sample_count(self, pair):
        fa, fb = pair
        c = compare_predictions(fa, fb, "a", "b")
        total = (c["both_correct"] + c["both_wrong"]
                 + c["only_a_correct"] + c["only_b_correct"])
        assert total == c["num_samples"]

    def test_identical_models_agree_completely(self, pair):
        fa, _ = pair
        c = compare_predictions(fa, fa, "a", "b")
        assert c["agreement_rate"] == 1.0
        assert c["num_disagreements"] == 0
        assert c["complementary"] is False

    def test_flags_complementary_models(self, pair):
        fa, fb = pair
        assert compare_predictions(fa, fb, "a", "b")["complementary"] is True

    def test_length_mismatch_raises(self, pair):
        fa, fb = pair
        with pytest.raises(ValueError, match="different numbers of rows"):
            compare_predictions(fa, fb.iloc[:50], "a", "b")

    def test_different_ground_truth_raises(self, rows):
        """Comparing predictions over different rows is meaningless, so it must
        not be silently possible."""
        n = len(rows)
        y_true = np.zeros(n, dtype=int)
        fa = build_error_frame(rows, y_true, y_true)
        fb = build_error_frame(rows, 1 - y_true, y_true)
        with pytest.raises(ValueError, match="different ground-truth"):
            compare_predictions(fa, fb, "a", "b")


class TestFigures:
    def test_hour_plot_is_written(self, frame, tmp_path):
        path = plot_error_rate_by_hour(frame, tmp_path)
        assert path.exists() and path.stat().st_size > 0

    def test_calibration_plot_is_written(self, frame, tmp_path):
        path = plot_calibration(frame, tmp_path)
        assert path.exists() and path.stat().st_size > 0


class TestReport:
    def test_returns_its_pieces(self, frame, capsys):
        report = print_report(frame)
        assert "summary" in report and "worst_slices" in report
        assert report["calibration"]["ece"] >= 0
        assert "Error summary" in capsys.readouterr().out

    def test_runs_without_probabilities(self, rows):
        n = len(rows)
        built = build_error_frame(rows, np.zeros(n), np.zeros(n))
        assert print_report(built)["calibration"] is None


class TestFeatureSetComparison:
    def test_temporal_features_fix_more_than_they_break(self, tmp_path):
        """The ablation's accuracy gain should come from fixed predictions, not
        from a reshuffle that happens to land better."""
        from data.make_sample_data import generate_dataset
        csv = tmp_path / "sample.csv"
        generate_dataset(800, seed=37).to_csv(csv, index=False)

        result = compare_feature_sets(csv, split_strategy="random")
        comparison = result["comparison"]
        assert (comparison["only_text_temporal_correct"]
                > comparison["only_text_only_correct"])

    def test_records_which_model_the_comparison_used(self, tmp_path):
        from data.make_sample_data import generate_dataset
        csv = tmp_path / "sample.csv"
        generate_dataset(500, seed=41).to_csv(csv, index=False)

        result = compare_feature_sets(csv, split_strategy="random",
                                      json_out=tmp_path / "cmp.json")
        assert result["comparison"]["model"] == "TF-IDF"
        saved = json.loads((tmp_path / "cmp.json").read_text())
        assert saved["comparison"]["model"] == "TF-IDF"

    def test_a_small_net_gain_is_not_called_information(self, tmp_path,
                                                         monkeypatch, capsys):
        """34 fixed against 32 broken is noise. The printout used to call any
        net gain 'adding information'; it now defers to McNemar."""
        import analysis.error_analysis as ea
        from data.make_sample_data import generate_dataset
        csv = tmp_path / "sample.csv"
        generate_dataset(300, seed=43).to_csv(csv, index=False)

        def near_tie(frame_a, frame_b, name_a, name_b):
            return {"num_samples": 3999, "agreement_rate": 0.98,
                    "both_correct": 2407, "both_wrong": 1526,
                    f"only_{name_a}_correct": 32, f"only_{name_b}_correct": 34}

        monkeypatch.setattr(ea, "compare_predictions", near_tie)
        compare_feature_sets(csv, split_strategy="random")
        out = capsys.readouterr().out
        assert "within noise" in out
        assert "adding information" not in out

    def test_returns_both_frames_and_summaries(self, tmp_path):
        from data.make_sample_data import generate_dataset
        csv = tmp_path / "sample.csv"
        generate_dataset(500, seed=41).to_csv(csv, index=False)

        result = compare_feature_sets(csv, split_strategy="random")
        assert set(result["frames"]) == {"text_only", "text_temporal"}
        assert set(result["summaries"]) == {"text_only", "text_temporal"}
