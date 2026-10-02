"""The one-command driver's flag plumbing.

`run_experiment.py` builds an argv and hands it to `benchmark.main`. Nothing
type-checks that hand-off, so a flag can exist on the driver, be accepted
without complaint, and never reach the thing it was meant to control. That is
what happened to `--no-amp`: it lived on `main.py` only, and the documented
Colab path went through `run_experiment.py`, where there was no way to turn
mixed precision off at all.
"""

import pytest

import benchmark
import run_experiment


class Captured(Exception):
    """Stops the driver once the grid argv has been seen."""

    def __init__(self, argv):
        super().__init__("captured")
        self.argv = argv


def grid_argv(monkeypatch, driver_argv):
    """Run the driver far enough to see what it would pass to the grid."""
    def fake_main(argv=None):
        raise Captured(list(argv or []))

    monkeypatch.setattr(benchmark, "main", fake_main)
    with pytest.raises(Captured) as excinfo:
        run_experiment.main(driver_argv)
    return excinfo.value.argv


class TestAmpEscapeHatch:
    """A NaN loss under AMP on a T4 is a real failure with one known fix. It
    has to be reachable from the command the runbook tells people to run."""

    def test_benchmark_exposes_it(self):
        args = benchmark.parse_args(["--dataset", "x", "--no-amp"])
        assert benchmark._build_config(args, False).use_amp is False

    def test_mixed_precision_is_on_by_default(self):
        args = benchmark.parse_args(["--dataset", "x"])
        assert benchmark._build_config(args, False).use_amp is True

    def test_the_driver_forwards_it(self, monkeypatch):
        argv = grid_argv(monkeypatch, ["--synthetic", "--no-amp"])
        assert "--no-amp" in argv

    def test_the_driver_does_not_forward_it_unasked(self, monkeypatch):
        argv = grid_argv(monkeypatch, ["--synthetic"])
        assert "--no-amp" not in argv


class TestGridArgv:
    """The rest of the hand-off, since the same gap could open anywhere in it."""

    def test_store_true_flags_are_passed_through(self, monkeypatch):
        argv = grid_argv(monkeypatch, ["--synthetic", "--skip-bert",
                                       "--embeddings", "--tiny-model"])
        for flag in ("--skip-bert", "--embeddings", "--tiny-model"):
            assert flag in argv

    def test_valued_options_keep_their_values(self, monkeypatch):
        argv = grid_argv(monkeypatch, ["--synthetic", "--epochs", "7",
                                       "--batch-size", "4", "--seed", "99"])
        for flag, value in (("--epochs", "7"), ("--batch-size", "4"),
                            ("--seed", "99")):
            assert argv[argv.index(flag) + 1] == value

    def test_the_temporal_split_is_the_default(self, monkeypatch):
        # A random split lets a model with timestamp features see the future.
        argv = grid_argv(monkeypatch, ["--synthetic"])
        assert argv[argv.index("--split-strategy") + 1] == "temporal"

    def test_every_forwarded_flag_is_one_the_grid_accepts(self, monkeypatch):
        # The failure this whole file exists for: a flag the driver sends and
        # the grid has never heard of dies with "unrecognized arguments"
        # halfway through a long run.
        argv = grid_argv(monkeypatch, [
            "--synthetic", "--skip-bert", "--embeddings", "--tiny-model",
            "--offline-tokenizer", "--no-amp", "--llm",
            "--encoder-name", "enc", "--llm-model", "m",
        ])
        benchmark.parse_args(argv)  # raises SystemExit on an unknown flag


class TestResolveDataset:
    def test_refuses_to_run_without_a_corpus_or_the_synthetic_flag(self):
        with pytest.raises(SystemExit, match="--dataset"):
            run_experiment.resolve_dataset(
                run_experiment.parse_args([]))

    def test_a_missing_dataset_path_is_named(self, tmp_path):
        args = run_experiment.parse_args(["--dataset", str(tmp_path / "nope.csv")])
        with pytest.raises(FileNotFoundError, match="nope.csv"):
            run_experiment.resolve_dataset(args)
