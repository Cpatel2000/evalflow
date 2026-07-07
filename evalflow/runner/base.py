"""Runner abstraction: same interface for local (asyncio) and future Ray execution.

Callers never know which runner they got. Keep this minimal -- the Ray runner
(week 3) implements this same interface without changes here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from evalflow.results import RunSummary, SampleResult
from evalflow.spec import EvalSpec


class Runner(ABC):
    """Executes an eval spec end to end: render, cache, complete, score."""

    @abstractmethod
    async def run(self, spec: EvalSpec) -> tuple[list[SampleResult], RunSummary]:
        """Run every sample in spec's dataset and return results plus a summary."""
