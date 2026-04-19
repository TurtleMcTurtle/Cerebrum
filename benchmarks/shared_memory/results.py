"""Results aggregation and output for the shared memory evaluation harness.

Provides the ResultsWriter class for computing summary statistics over
trial results and writing experiment output as JSON and optionally CSV.
"""

import csv
import json
import os
import statistics
from typing import List, Optional

from benchmarks.shared_memory.models import (
    ConditionSummary,
    ExperimentResults,
    SummaryStatistics,
    TrialResult,
)


class ResultsWriter:
    """Aggregates trial results and writes JSON/CSV output."""

    def __init__(self, output_dir: str, write_csv: bool = False):
        """Configure output directory and CSV flag.

        Args:
            output_dir: Directory path for writing result files.
            write_csv: If True, also write a CSV file alongside JSON.
        """
        self.output_dir = output_dir
        self.write_csv_flag = write_csv

    def _compute_metric_stats(self, values: List[float]) -> SummaryStatistics:
        """Compute summary statistics for a list of numeric values.

        Args:
            values: Non-empty list of float values.

        Returns:
            SummaryStatistics with mean, std, min, max.
        """
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return SummaryStatistics(
            mean=mean,
            std=std,
            min=min(values),
            max=max(values),
        )

    def compute_summary_statistics(self, trials: List[TrialResult]) -> ConditionSummary:
        """Compute summary statistics for all metrics, excluding failed trials.

        Args:
            trials: List of TrialResult objects for a single condition.

        Returns:
            ConditionSummary with statistics for each metric.
        """
        total_trials = len(trials)
        failed_trials = sum(1 for t in trials if t.failed)
        non_failed = [t for t in trials if not t.failed]

        relevance_vals = [t.relevance_score for t in non_failed if t.relevance_score is not None]
        personalization_vals = [t.personalization_score for t in non_failed if t.personalization_score is not None]
        latency_vals = [t.latency_seconds for t in non_failed if t.latency_seconds is not None]
        memory_total_vals = [float(t.memory_counts.total) for t in non_failed]
        memory_shared_vals = [float(t.memory_counts.shared) for t in non_failed]
        memory_private_vals = [float(t.memory_counts.private) for t in non_failed]

        def _safe_stats(vals: List[float]) -> SummaryStatistics:
            if not vals:
                return SummaryStatistics(mean=0.0, std=0.0, min=0.0, max=0.0)
            return self._compute_metric_stats(vals)

        return ConditionSummary(
            relevance=_safe_stats([float(v) for v in relevance_vals]),
            personalization=_safe_stats([float(v) for v in personalization_vals]),
            latency=_safe_stats(latency_vals),
            memory_total=_safe_stats(memory_total_vals),
            memory_shared=_safe_stats(memory_shared_vals),
            memory_private=_safe_stats(memory_private_vals),
            total_trials=total_trials,
            failed_trials=failed_trials,
        )

    def write_json(self, experiment: ExperimentResults) -> str:
        """Write the full experiment results to results.json.

        Args:
            experiment: Complete experiment results to serialize.

        Returns:
            File path of the written JSON file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, "results.json")
        with open(file_path, "w") as f:
            json.dump(experiment.model_dump(), f, indent=2)
        return file_path

    def write_csv(self, experiment: ExperimentResults) -> str:
        """Write per-trial CSV with one row per trial across all conditions.

        Args:
            experiment: Complete experiment results to export.

        Returns:
            File path of the written CSV file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, "results.csv")
        columns = [
            "condition",
            "trial_index",
            "relevance_score",
            "personalization_score",
            "memory_total",
            "memory_shared",
            "memory_private",
            "latency_seconds",
            "query",
            "response",
        ]
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for condition_result in experiment.conditions:
                for trial in condition_result.trials:
                    writer.writerow([
                        trial.condition,
                        trial.trial_index,
                        trial.relevance_score,
                        trial.personalization_score,
                        trial.memory_counts.total,
                        trial.memory_counts.shared,
                        trial.memory_counts.private,
                        trial.latency_seconds,
                        trial.follow_up_query,
                        trial.assistant_response,
                    ])
        return file_path
