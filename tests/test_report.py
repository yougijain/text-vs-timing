"""Report generator tests.

The statistics get the most attention here. A report that renders a slightly
wrong p-value looks exactly like one that is right, and it is the number a
reader will quote back at you.
"""

import json

import pytest

from analysis.report import (
    ECE_CEILING,
    accuracy_interval,
    classify_outcome,
    load_artifacts,
    mcnemar,
    render_ablation,
    render_calibration,
    render_grid,
    render_provenance,
    render_report,
    temporal_lift,
    write_report,
)


def _metrics(accuracy, n=400, baseline=0.5):
    return {
        "accuracy": accuracy,
        "macro_f1": accuracy - 0.002,
        "num_samples": n,
        "majority_baseline": baseline,
        "lift_over_baseline": accuracy - baseline,
    }


def _grid(tfidf_text=0.60, tfidf_both=0.70, bert_text=0.64, bert_both=0.74, n=400):
    return {"rows": [
        {"model": "TF-IDF", "features": "text only", "metrics": _metrics(tfidf_text, n)},
        {"model": "TF-IDF", "features": "text + temporal", "metrics": _metrics(tfidf_both, n)},
        {"model": "BERT", "features": "text only", "metrics": _metrics(bert_text, n)},
        {"model": "BERT", "features": "text + temporal", "metrics": _metrics(bert_both, n)},
    ]}


class TestAccuracyInterval:
    def test_interval_brackets_the_estimate(self):
        lo, hi = accuracy_interval(0.8, 400)
        assert lo < 0.8 < hi

    def test_more_samples_narrow_the_interval(self):
        narrow = accuracy_interval(0.8, 4000)
        wide = accuracy_interval(0.8, 100)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_zero_samples_does_not_divide_by_zero(self):
        assert accuracy_interval(0.8, 0) == (0.0, 0.0)

    def test_bounds_are_clamped_to_zero_one(self):
        lo, hi = accuracy_interval(0.99, 10)
        assert lo >= 0.0 and hi <= 1.0


class TestMcNemar:
    def test_lopsided_disagreement_is_significant(self):
        result = mcnemar(only_a_correct=5, only_b_correct=60)
        assert result["significant"]
        assert result["p_value"] < 0.05
        assert result["n_discordant"] == 65

    def test_even_disagreement_is_not_significant(self):
        result = mcnemar(only_a_correct=40, only_b_correct=38)
        assert not result["significant"]
        assert result["p_value"] > 0.05

    def test_no_disagreement_has_no_p_value(self):
        result = mcnemar(0, 0)
        assert result["p_value"] is None
        assert not result["significant"]
        assert "never disagree" in result["method"]

    def test_few_pairs_use_the_exact_test(self):
        # The chi-square approximation is unreliable below ~25 discordant pairs.
        result = mcnemar(1, 9)
        assert "exact binomial" in result["method"]
        assert result["n_discordant"] == 10

    def test_many_pairs_use_chi_square(self):
        assert "chi-square" in mcnemar(10, 40)["method"]

    def test_is_symmetric_in_magnitude(self):
        assert mcnemar(5, 60)["statistic"] == mcnemar(60, 5)["statistic"]

    def test_continuity_correction_is_applied(self):
        # (|b - c| - 1)^2 / n, not (b - c)^2 / n.
        assert mcnemar(10, 40)["statistic"] == pytest.approx((30 - 1) ** 2 / 50)


class TestTemporalLift:
    def test_computes_the_delta(self):
        lift = temporal_lift(_grid()["rows"], "TF-IDF")
        assert lift["delta"] == pytest.approx(0.10)

    def test_half_a_grid_returns_none(self):
        # A grid missing one cell must not be written up as an ablation.
        rows = [{"model": "TF-IDF", "features": "text only", "metrics": _metrics(0.6)}]
        assert temporal_lift(rows, "TF-IDF") is None


class TestClassifyOutcome:
    def test_clear_bert_win_is_reported_as_such(self):
        verdicts = classify_outcome(_grid(tfidf_both=0.60, bert_both=0.85, n=2000)["rows"])
        assert verdicts["model_verdict"] == "bert_wins"
        assert "cannot reach" in verdicts["headline"]

    def test_close_accuracies_are_a_tie(self):
        verdicts = classify_outcome(_grid(tfidf_both=0.700, bert_both=0.705, n=300)["rows"])
        assert verdicts["model_verdict"] == "tie"
        assert "not earning its compute" in verdicts["headline"]

    def test_tfidf_win_is_not_framed_as_a_bug(self):
        verdicts = classify_outcome(_grid(tfidf_both=0.85, bert_both=0.60, n=2000)["rows"])
        assert verdicts["model_verdict"] == "tfidf_wins"
        assert "legitimate outcome" in verdicts["headline"]

    def test_mcnemar_is_used_when_a_comparison_is_supplied(self):
        comparison = {"only_text-only_correct": 12,
                      "only_text+temporal_correct": 70}
        verdicts = classify_outcome(_grid()["rows"], comparison)
        assert verdicts["temporal_verdict"] == "helps"
        assert verdicts["temporal_test"]["fixed"] == 70
        assert verdicts["temporal_test"]["broken"] == 12
        assert any("McNemar" in n for n in verdicts["notes"])

    def test_temporal_verdict_can_come_out_inconclusive(self):
        comparison = {"only_text-only_correct": 30,
                      "only_text+temporal_correct": 32}
        verdicts = classify_outcome(_grid()["rows"], comparison)
        assert verdicts["temporal_verdict"] == "inconclusive"

    def test_falls_back_to_intervals_without_a_comparison(self):
        verdicts = classify_outcome(_grid(n=4000)["rows"])
        assert "temporal_verdict" in verdicts
        assert any("confidence intervals" in n for n in verdicts["notes"])

    def test_architecture_dependence_is_called_out(self):
        # Timestamp helps TF-IDF, does nothing for BERT.
        rows = _grid(tfidf_text=0.60, tfidf_both=0.75,
                     bert_text=0.70, bert_both=0.70)["rows"]
        verdicts = classify_outcome(rows)
        assert "fusion head" in verdicts["architecture_dependence"]

    def test_help_across_architectures_is_called_out(self):
        verdicts = classify_outcome(_grid()["rows"])
        assert "every architecture" in verdicts["architecture_dependence"]

    def test_tiny_model_caveat_leads_the_notes(self):
        rows = _grid()["rows"]
        for row in rows:
            row["model"] = row["model"].replace("BERT", "BERT (tiny, random)")
        verdicts = classify_outcome(rows)
        assert "plumbing check" in verdicts["notes"][0]

    def test_empty_grid_does_not_raise(self):
        verdicts = classify_outcome([])
        assert "headline" not in verdicts

    def test_passing_the_whole_document_fails_loudly(self):
        # Iterating a dict yields keys, so without this guard the mistake
        # surfaces several frames later as "string indices must be integers".
        with pytest.raises(TypeError, match="benchmark\\['rows'\\]"):
            classify_outcome(_grid())


class TestRendering:
    def test_grid_states_the_baseline_and_every_lift(self):
        text = render_grid(_grid()["rows"])
        assert "Majority-class baseline" in text
        assert text.count("|") > 20
        for accuracy in ("0.6000", "0.7000", "0.6400", "0.7400"):
            assert accuracy in text
        assert "+0.2400" in text  # best cell's lift

    def test_grid_bolds_the_best_cell(self):
        assert "**0.7400**" in render_grid(_grid()["rows"])

    def test_empty_grid_renders_a_placeholder(self):
        assert "No benchmark rows" in render_grid([])

    def test_ablation_includes_the_mcnemar_line(self):
        comparison = {"only_text-only_correct": 12,
                      "only_text+temporal_correct": 70}
        verdicts = classify_outcome(_grid()["rows"], comparison)
        text = render_ablation(verdicts)
        assert "fixes 70" in text and "breaks 12" in text
        assert "McNemar" in text

    def test_ablation_names_the_pair_the_test_was_run_on(self):
        # The table above it has a row per architecture; the paired test
        # covers one of them, and the reader has to be told which.
        comparison = {"only_text-only_correct": 32,
                      "only_text+temporal_correct": 34, "model": "TF-IDF"}
        text = render_ablation(classify_outcome(_grid()["rows"], comparison))
        assert "On the **TF-IDF** pair" in text
        assert "other rows (BERT)" in text

    def test_an_older_comparison_without_a_model_is_tfidf(self):
        comparison = {"only_text-only_correct": 32,
                      "only_text+temporal_correct": 34}
        verdicts = classify_outcome(_grid()["rows"], comparison)
        assert verdicts["temporal_test"]["model"] == "TF-IDF"

    def test_calibration_flags_a_bad_ece(self):
        text = render_calibration({"ece": ECE_CEILING + 0.2,
                                   "summary": {"error_rate": 0.3, "num_samples": 100}})
        assert "cannot be thresholded" in text

    def test_calibration_accepts_a_good_ece(self):
        text = render_calibration({"ece": 0.02, "summary": {}})
        assert "mean roughly what they say" in text

    def test_calibration_reads_a_nested_ece(self):
        assert "0.0300" in render_calibration({"calibration": {"ece": 0.03}})

    def test_calibration_without_a_run_says_so(self):
        assert "not run" in render_calibration(None)

    def test_one_sided_errors_are_called_a_threshold_problem(self):
        text = render_calibration({"ece": 0.02, "summary": {"one_sided": True,
                                                            "error_rate": 0.2,
                                                            "num_samples": 100}})
        assert "threshold problem" in text

    def test_provenance_quotes_the_licence_and_attribution(self):
        text = render_provenance({
            "source": "Stack Exchange API v2.3", "site": "cooking",
            "licence": "CC BY-SA 4.0", "rows": 2000,
            "attribution": "Content from the Stack Exchange 'cooking' site",
            "fetched_at_utc": "2026-01-01T00:00:00+00:00",
            "first_post_utc": "2023-01-01T00:00:00+00:00",
            "last_post_utc": "2023-12-31T00:00:00+00:00",
        })
        assert "CC BY-SA 4.0" in text
        assert "2023-01-01 to 2023-12-31" in text
        assert "> Content from" in text

    def test_missing_provenance_says_how_to_get_one(self):
        assert "fetch_dataset" in render_provenance(None, "datasets/mine.csv")


class TestRenderReport:
    def test_carries_the_utc_caveat_forward(self):
        text = render_report(_grid())
        assert "UTC" in text
        assert "circadian" in text

    def test_includes_every_section(self):
        text = render_report(_grid(), error_analysis={"ece": 0.04},
                             figures=["outputs/figures/calibration.png"])
        for heading in ("## What happened", "## Dataset", "## The grid",
                        "## Does the timestamp add anything?",
                        "## Calibration and errors", "## Figures",
                        "## The UTC caveat"):
            assert heading in text

    def test_says_so_when_the_grid_is_too_thin_to_compare(self):
        thin = {"rows": [{"model": "TF-IDF", "features": "text only",
                          "metrics": _metrics(0.6)}]}
        assert "Not enough of the grid" in render_report(thin)

    def test_warns_the_document_is_generated(self):
        assert "Do not edit by hand" in render_report(_grid())


class TestArtifacts:
    def test_missing_artifacts_are_absent_not_fatal(self, tmp_path):
        artifacts = load_artifacts(tmp_path)
        assert artifacts["benchmark"] is None
        assert artifacts["error_analysis"] is None
        assert artifacts["figures"] == []

    def test_loads_what_is_present(self, tmp_path):
        (tmp_path / "benchmark.json").write_text(json.dumps(_grid()))
        artifacts = load_artifacts(tmp_path)
        assert len(artifacts["benchmark"]["rows"]) == 4

    def test_invalid_json_is_skipped_rather_than_raised(self, tmp_path):
        (tmp_path / "benchmark.json").write_text("{not json")
        assert load_artifacts(tmp_path)["benchmark"] is None

    def test_write_report_unwraps_a_nested_comparison(self, tmp_path):
        # compare_feature_sets writes {"comparison": {...}, "summaries": {...}}.
        out = tmp_path / "RESULTS.md"
        write_report(out, {
            "benchmark": _grid(),
            "comparison": {"comparison": {"only_text-only_correct": 12,
                                          "only_text+temporal_correct": 70}},
        })
        assert "fixes 70" in out.read_text()

    def test_write_report_returns_the_path(self, tmp_path):
        out = tmp_path / "RESULTS.md"
        assert write_report(out, {"benchmark": _grid()}) == out


class TestSyntheticBanner:
    """A table of generated numbers looks exactly like a table of real ones."""

    def test_banner_leads_the_document(self):
        text = render_report(_grid(), synthetic=True)
        assert "Not a result" in text
        assert text.index("Not a result") < text.index("## The grid")

    def test_no_banner_on_a_real_run(self):
        assert "Not a result" not in render_report(_grid())

    def test_write_report_passes_the_flag_through(self, tmp_path):
        out = tmp_path / "RESULTS.md"
        write_report(out, {"benchmark": _grid()}, synthetic=True)
        assert "Not a result" in out.read_text()

    def test_artifacts_can_carry_the_flag(self, tmp_path):
        out = tmp_path / "RESULTS.md"
        write_report(out, {"benchmark": _grid(), "synthetic": True})
        assert "Not a result" in out.read_text()
