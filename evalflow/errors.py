"""Exception hierarchy for evalflow.

All evalflow code raises these instead of bare exceptions so callers can
distinguish spec problems from runtime problems.
"""

from __future__ import annotations


class EvalflowError(Exception):
    """Base class for all evalflow errors."""


class SpecError(EvalflowError):
    """The eval spec is invalid. Message names the field and the fix."""


class DatasetError(EvalflowError):
    """The dataset file is missing, malformed, or incompatible with the spec."""


class ProviderError(EvalflowError):
    """A model provider call failed after retries were exhausted."""


class JudgeError(EvalflowError):
    """A judge response could not be parsed or the judge file is invalid."""


class CacheError(EvalflowError):
    """The response cache was misused (e.g. accessed before connect())."""
