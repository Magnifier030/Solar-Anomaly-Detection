from __future__ import annotations

import pytest
import torch

from research_training.backends import configure_backends
from research_training.config import config_hash


@pytest.fixture(autouse=True)
def restore_backend_flags():
    previous = (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        yield
    finally:
        (
            torch.backends.cudnn.enabled,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
        ) = previous


def test_explicit_cudnn_tuning_and_legacy_defaults():
    config = {"runtime": {"cudnn": {"enabled": True, "benchmark": True, "deterministic": False}}}
    metadata = configure_backends(config, torch.device("cpu"))
    assert torch.backends.cudnn.enabled
    assert torch.backends.cudnn.benchmark
    assert not metadata["cudnn_active"]
    assert metadata["cudnn_benchmark"]
    legacy = configure_backends({}, torch.device("cpu"))
    assert not torch.backends.cudnn.benchmark
    assert legacy["cudnn_enabled"]
    assert not legacy["cudnn_deterministic"]


def test_backend_change_participates_in_resume_compatibility():
    original = {"name": "run"}
    tuned = {"name": "run", "runtime": {"cudnn": {"benchmark": True}}}
    assert config_hash(original) != config_hash(tuned)


@pytest.mark.parametrize(
    "flags",
    [
        {"enabled": False, "benchmark": True},
        {"benchmark": True, "deterministic": True},
        {"benchmark": "false"},
    ],
)
def test_reject_ambiguous_or_contradictory_flags(flags):
    with pytest.raises(ValueError):
        configure_backends({"runtime": {"cudnn": flags}}, torch.device("cpu"))
