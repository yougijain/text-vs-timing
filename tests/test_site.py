"""Static-page tests.

Three things here carry real risk. The page is published, so anything from the
corpus that reaches it is untrusted text and must be escaped. It renders before
any run exists, so the empty state has to be a complete page rather than a
stack of exceptions. And it must not soften a number: a gain inside the noise
has to look like a gain inside the noise.
"""

from html.parser import HTMLParser

import pytest

from analysis.site import (
    QUESTION,
    build_site,
    copy_figures,
    plain_answer,
    render_grid,
    render_page,
    _pct,
    _points,
)

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "source", "track", "wbr"}


class TagBalance(HTMLParser):
    """An unclosed div silently wrecks the layout and nothing else notices."""

    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> with nothing open")
        elif self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes <{self.stack[-1]}>")
        else:
            self.stack.pop()

    @classmethod
    def check(cls, markup):
        parser = cls()
        parser.feed(markup)
        return parser.errors + [f"<{t}> never closed" for t in parser.stack]


def _handler_attributes(markup):
    """Every on* attribute the parser actually sees. Escaped text yields none."""
    found = []

    class Collect(HTMLParser):
        def handle_starttag(self, tag, attrs):
            found.extend(f"{tag}.{name}" for name, _ in attrs
                         if name.startswith("on"))

    Collect().feed(markup)
    return found


def metrics(accuracy, n=226, baseline=0.5, macro_f1=None):
    return {
        "accuracy": accuracy,
        "macro_f1": accuracy if macro_f1 is None else macro_f1,
        "num_samples": n,
        "majority_baseline": baseline,
        "lift_over_baseline": accuracy - baseline,
    }


@pytest.fixture
def rows():
    return [
        {"model": "TF-IDF", "features": "text only", "metrics": metrics(0.62)},
        {"model": "TF-IDF", "features": "text + temporal", "metrics": metrics(0.71)},
    ]


@pytest.fixture
def artifacts(rows):
    return {
        "benchmark": {"rows": rows},
        "error_analysis": {
            "ece": 0.04,
            "summary": {"num_samples": 226, "error_rate": 0.29,
                        "false_positive": 30, "false_negative": 36},
            "worst_slices": [{"column": "hour", "bucket": "00-03",
                              "count": 40, "error_rate": 0.4}],
        },
        "comparison": {"comparison": {"only_text_only_correct": 10,
                                      "only_text_temporal_correct": 40}},
        "provenance": {"source": "Stack Exchange", "site": "stackoverflow",
                       "licence": "CC BY-SA 4.0", "rows": 20000,
                       "fetched_at_utc": "2026-01-02T03:04:05Z"},
        "dataset_path": "datasets/posts.csv",
        "figures": [],
    }


class TestFormatting:
    def test_accuracies_render_as_percentages(self):
        assert _pct(0.8230) == "82.3%"

    def test_deltas_render_as_points_not_percent(self):
        # "+2.7%" and "+2.7 points" are different claims and get confused.
        assert _points(0.027) == "+2.7 points"
        assert "%" not in _points(0.027)

    def test_missing_values_do_not_render_as_zero(self):
        assert _pct(None) == "n/a"
        assert _points(None) == "n/a"


class TestPlainAnswer:
    def test_no_rows_says_not_run_rather_than_guessing(self):
        short, lines = plain_answer({}, [])
        assert "Not run yet" in short
        assert lines

    def test_a_positive_verdict_reads_as_yes(self, rows):
        short, _ = plain_answer({"temporal_verdict": "helps"}, rows)
        assert short.startswith("Yes")

    def test_a_negative_verdict_is_not_softened(self, rows):
        short, _ = plain_answer({"temporal_verdict": "hurts"}, rows)
        assert "No" in short and "worse" in short

    def test_inconclusive_is_stated_as_a_result(self, rows):
        _, lines = plain_answer({"temporal_verdict": "inconclusive"}, rows)
        assert any("is a result" in line for line in lines)

    def test_fixed_and_broken_counts_are_both_quoted(self, rows):
        _, lines = plain_answer(
            {"temporal_verdict": "helps",
             "temporal_test": {"fixed": 40, "broken": 10, "p_value": 0.0001,
                               "method": "chi-square"}},
            rows,
        )
        joined = " ".join(lines)
        assert "40" in joined and "10" in joined


class TestGrid:
    def test_a_gain_inside_the_noise_is_not_coloured_as_a_win(self):
        # 3 points over baseline on 226 rows is inside the interval. Colouring
        # it green would be the page making a claim the numbers do not support.
        rows = [{"model": "TF-IDF", "features": "text only",
                 "metrics": metrics(0.55, n=226, baseline=0.52)}]
        assert 'class="card flat"' in render_grid(rows)

    def test_a_gain_outside_the_noise_is(self):
        rows = [{"model": "TF-IDF", "features": "text + temporal",
                 "metrics": metrics(0.82, n=226, baseline=0.52)}]
        assert 'class="card good"' in render_grid(rows)

    def test_the_baseline_is_always_on_the_page(self, rows):
        # An accuracy without its baseline is unreadable.
        assert "baseline" in render_grid(rows).lower()

    def test_bar_widths_stay_inside_the_bar(self):
        rows = [{"model": "M", "features": "text only",
                 "metrics": metrics(1.0, baseline=1.0)}]
        markup = render_grid(rows)
        assert "width:100.0%" in markup
        assert not any(f"width:{n}" in markup for n in range(101, 200))

    def test_zero_samples_does_not_divide_by_zero(self):
        rows = [{"model": "M", "features": "text only",
                 "metrics": metrics(0.5, n=0)}]
        assert render_grid(rows)

    def test_empty_grid_says_so(self):
        assert "Nothing has been trained yet" in render_grid([])


class TestPage:
    def test_the_question_is_the_headline(self, artifacts):
        assert QUESTION in render_page(artifacts)

    def test_markup_is_balanced(self, artifacts):
        assert TagBalance.check(render_page(artifacts)) == []

    def test_empty_markup_is_balanced_too(self):
        assert TagBalance.check(render_page({})) == []

    def test_renders_a_whole_page_with_no_artefacts(self):
        page = render_page({})
        for heading in ("The grid", "Can you trust the confidence?",
                        "What this cannot tell you", "The data"):
            assert heading in page
        assert "Not run yet" in page

    def test_numbers_come_from_the_artefacts(self, artifacts):
        page = render_page(artifacts)
        assert "71.0%" in page and "62.0%" in page

    def test_the_utc_caveat_survives_onto_the_page(self, artifacts):
        page = render_page(artifacts)
        assert "UTC" in page
        assert "timezone" in page

    def test_the_synthetic_warning_sits_above_the_numbers(self, artifacts):
        page = render_page(artifacts, synthetic=True)
        assert page.index("generated sample data") < page.index("The grid")

    def test_a_synthetic_run_does_not_read_as_a_finding(self, artifacts):
        assert "not a finding" in render_page(artifacts, synthetic=True)

    def test_a_real_run_carries_no_warning_banner(self, artifacts):
        assert "generated sample data" not in render_page(artifacts)

    def test_provenance_and_licence_are_on_the_page(self, artifacts):
        page = render_page(artifacts)
        assert "CC BY-SA 4.0" in page and "stackoverflow" in page

    def test_the_page_is_self_contained(self, artifacts):
        # No CDN, no remote font, no analytics. It has to render from a
        # directory of files and nothing else.
        page = render_page(artifacts)
        for tag in ("<script", "src=\"http", "@import"):
            assert tag not in page

    def test_results_link_is_omitted_when_there_is_no_results_file(
            self, artifacts, tmp_path):
        page = render_page(artifacts, results_path=tmp_path / "nope.md")
        assert "the full numbers" not in page

    def test_results_link_appears_once_the_file_exists(self, artifacts, tmp_path):
        results = tmp_path / "RESULTS.md"
        results.write_text("# Results")
        assert "the full numbers" in render_page(artifacts, results_path=results)


class TestEscaping:
    """Provenance strings come off a remote API. They are text, not markup."""

    def test_provenance_is_escaped(self, artifacts):
        artifacts["provenance"]["attribution"] = "<script>alert(1)</script>"
        page = render_page(artifacts)
        assert "<script>" not in page
        assert "&lt;script&gt;" in page

    def test_a_hostile_site_name_cannot_open_a_tag(self, artifacts):
        # Asserting the substring is absent would be wrong: escaped, the text
        # still reads "onerror=alert(1)" and is inert. What matters is that the
        # parser sees no tag and no handler attribute.
        artifacts["provenance"]["site"] = '"><img src=x onerror=alert(1)>'
        page = render_page(artifacts)
        assert not _handler_attributes(page)
        assert TagBalance.check(page) == []

    def test_a_hostile_model_name_is_escaped(self, artifacts):
        artifacts["benchmark"]["rows"][0]["model"] = "<b>TF-IDF</b>"
        page = render_page(artifacts)
        assert "<b>TF-IDF</b>" not in page


class TestBuildSite:
    def test_writes_a_page_and_a_nojekyll(self, artifacts, tmp_path):
        page = build_site(artifacts, out_dir=tmp_path)
        assert page.exists()
        # Pages runs Jekyll by default and drops files it does not recognise.
        assert (tmp_path / ".nojekyll").exists()

    def test_copies_only_the_figures_that_exist(self, tmp_path):
        source = tmp_path / "figures"
        source.mkdir()
        (source / "calibration.png").write_bytes(b"png")
        available = copy_figures(source, tmp_path / "site")
        assert [name for name, _ in available] == ["calibration.png"]
        assert (tmp_path / "site" / "figures" / "calibration.png").exists()

    def test_no_figures_directory_is_not_an_error(self, tmp_path):
        assert copy_figures(tmp_path / "missing", tmp_path / "site") == []

    def test_figures_are_referenced_relatively(self, artifacts, tmp_path):
        source = tmp_path / "figures"
        source.mkdir()
        (source / "calibration.png").write_bytes(b"png")
        build_site(artifacts, out_dir=tmp_path / "site", figure_dir=source)
        page = (tmp_path / "site" / "index.html").read_text()
        # An absolute path works on the machine that built it and 404s for
        # everyone else.
        assert 'src="figures/calibration.png"' in page

    def test_rebuilding_overwrites_rather_than_accumulates(self, artifacts,
                                                           tmp_path):
        build_site(artifacts, out_dir=tmp_path)
        build_site(artifacts, out_dir=tmp_path, synthetic=True)
        page = (tmp_path / "index.html").read_text()
        assert page.count("<h1>") == 1
        assert "generated sample data" in page
