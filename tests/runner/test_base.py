"""Tests for evalflow.runner.base: the minimal Runner interface.

Kept deliberately thin -- this is just the shape the Ray runner (week 3) will
also implement. Behavior lives in LocalRunner (tests/runner/test_local.py).
"""

from __future__ import annotations

import pytest

from evalflow.runner.base import Runner


def test_runner_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Runner()  # type: ignore[abstract]


def test_local_runner_is_a_runner() -> None:
    from evalflow.runner.local import LocalRunner

    assert issubclass(LocalRunner, Runner)
