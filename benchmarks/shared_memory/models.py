"""Pydantic data models for the shared memory evaluation harness.

Defines all structured data shapes used across the harness: synthetic data
generation inputs, judge scores, per-trial results, and experiment-level
aggregation output. No business logic — just data and validation.
"""

from typing import List, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Synthetic data models
# ---------------------------------------------------------------------------

class SyntheticProfile(BaseModel):
    """Matches ProfileAgent extraction schema."""

    user_name: str
    preferred_tools: List[str]
    preferred_language: str
    response_style: str


class SyntheticTaskContext(BaseModel):
    """Matches TaskAgent extraction schema."""

    current_project: str
    active_experiment: str
    goals: List[str]
    blockers: List[str]
    next_steps: List[str]


class SyntheticTrialData(BaseModel):
    """All generated data for one trial."""

    profile: SyntheticProfile
    task_context: SyntheticTaskContext
    follow_up_query: str


# ---------------------------------------------------------------------------
# Metric and result models
# ---------------------------------------------------------------------------

class JudgeScores(BaseModel):
    """LLM judge output."""

    relevance_score: Optional[int] = None
    personalization_score: Optional[int] = None
    relevance_reasoning: Optional[str] = None
    personalization_reasoning: Optional[str] = None


class MemoryCounts(BaseModel):
    """Memory creation counts for a trial."""

    total: int = 0
    shared: int = 0
    private: int = 0


class TrialResult(BaseModel):
    """Complete result for one trial."""

    trial_index: int
    condition: str
    relevance_score: Optional[int] = None
    personalization_score: Optional[int] = None
    memory_counts: MemoryCounts = MemoryCounts()
    latency_seconds: Optional[float] = None
    follow_up_query: str = ""
    assistant_response: str = ""
    synthetic_profile: Optional[SyntheticProfile] = None
    synthetic_task_context: Optional[SyntheticTaskContext] = None
    failed: bool = False
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Experiment output models
# ---------------------------------------------------------------------------

class SummaryStatistics(BaseModel):
    """Aggregated stats for one metric."""

    mean: float
    std: float
    min: float
    max: float


class ConditionSummary(BaseModel):
    """Summary statistics for all metrics in one condition."""

    relevance: SummaryStatistics
    personalization: SummaryStatistics
    latency: SummaryStatistics
    memory_total: SummaryStatistics
    memory_shared: SummaryStatistics
    memory_private: SummaryStatistics
    total_trials: int
    failed_trials: int


class ExperimentMetadata(BaseModel):
    """Top-level experiment configuration."""

    trials_per_condition: int
    timestamp: str
    kernel_url: str
    conditions_run: List[str]


class ConditionResults(BaseModel):
    """All trials and summary for one condition."""

    condition: str
    trials: List[TrialResult]
    summary: ConditionSummary


class ExperimentResults(BaseModel):
    """Top-level output structure for the Results_File."""

    experiment_metadata: ExperimentMetadata
    conditions: List[ConditionResults]
