"""Exception hierarchy for gradetrail.

All gradetrail code raises these instead of bare exceptions so callers can
distinguish spec problems from runtime problems.
"""

from __future__ import annotations


class GradetrailError(Exception):
    """Base class for all gradetrail errors."""


class SpecError(GradetrailError):
    """The eval spec is invalid. Message names the field and the fix."""


class DatasetError(GradetrailError):
    """The dataset file is missing, malformed, or incompatible with the spec."""


class ProviderError(GradetrailError):
    """A model provider call failed after retries were exhausted."""


class JudgeError(GradetrailError):
    """A judge response could not be parsed or the judge file is invalid."""


class CacheError(GradetrailError):
    """The response cache was misused (e.g. accessed before connect())."""


class ResultsError(GradetrailError):
    """A SampleResult had a state outside the closed set summarize() knows how to count."""
