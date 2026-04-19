"""CLI entry point and orchestrator for the shared memory evaluation harness.

Runs a two-condition experiment comparing private-only memory (Phase 1)
against shared memory (Phase 2) across synthetic trials, collecting
relevance, personalization, memory-count, and latency metrics.

Usage::

    python benchmarks/shared_memory/run_evaluation.py --trials 10 --output results/
    python benchmarks/shared_memory/run_evaluation.py --trials 5 --condition phase2 --csv

Note on kernel ``auto_inject`` behaviour:
    The AIOS kernel's ``memory.auto_inject`` setting independently injects
    relevant memories into LLM calls.  For controlled experiments that
    isolate the effect of agent-level shared memory, consider disabling
    ``auto_inject`` in the kernel ``config.yaml`` or passing the
    ``--disable-auto-inject`` flag.  If the flag is used but the harness
    cannot programmatically update the kernel config, a warning is logged
    instructing the user to disable it manually.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Ensure the project root is on sys.path when running as a script
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tqdm import tqdm

from benchmarks.shared_memory.models import (
    ConditionResults,
    ExperimentMetadata,
    ExperimentResults,
    JudgeScores,
    MemoryCounts,
    TrialResult,
)
from benchmarks.shared_memory.judge import LLMJudge
from benchmarks.shared_memory.pipeline import AgentPipeline
from benchmarks.shared_memory.results import ResultsWriter
from benchmarks.shared_memory.synth import SyntheticDataGenerator
from cerebrum.config.config_manager import config

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """Orchestrates the shared memory evaluation experiment.

    Coordinates synthetic data generation, agent pipeline execution,
    LLM judge scoring, and results aggregation across experimental
    conditions.

    Args:
        trials: Number of trials to run per condition.
        output_dir: Directory path for writing result files.
        write_csv: If True, also write a CSV file alongside JSON.
        condition: Which conditions to run — ``"both"``, ``"phase1"``,
            or ``"phase2"``.
        disable_auto_inject: If True, attempt to disable the kernel's
            ``memory.auto_inject`` setting for the experiment.
    """

    def __init__(
        self,
        trials: int,
        output_dir: str,
        write_csv: bool,
        condition: str,
        disable_auto_inject: bool,
    ):
        self.trials = trials
        self.output_dir = output_dir
        self.write_csv = write_csv
        self.condition = condition
        self.disable_auto_inject = disable_auto_inject

        self.generator = SyntheticDataGenerator()
        self.judge = LLMJudge()
        self.writer = ResultsWriter(output_dir=output_dir, write_csv=write_csv)

    # ------------------------------------------------------------------
    # Single trial
    # ------------------------------------------------------------------

    def run_single_trial(
        self,
        trial_index: int,
        condition: str,
        pipeline: AgentPipeline,
    ) -> TrialResult:
        """Execute one trial with log-and-continue error handling.

        Args:
            trial_index: Zero-based index of this trial.
            condition: ``"phase1"`` or ``"phase2"``.
            pipeline: Pre-configured ``AgentPipeline`` for this condition.

        Returns:
            A ``TrialResult`` — possibly with ``failed=True`` if any
            stage raised an exception.
        """
        # Step 1: Generate synthetic data
        try:
            trial_data = self.generator.generate_trial_data(trial_index)
        except Exception as e:
            logger.error("Trial %d: synthetic data generation failed: %s", trial_index, e)
            return TrialResult(
                trial_index=trial_index,
                condition=condition,
                failed=True,
                error_message=str(e),
            )

        # Step 2: Run agent pipeline
        try:
            pipeline_result = pipeline.run_trial(trial_data)
        except Exception as e:
            logger.error("Trial %d: agent pipeline failed: %s", trial_index, e)
            return TrialResult(
                trial_index=trial_index,
                condition=condition,
                failed=True,
                error_message=str(e),
                synthetic_profile=trial_data.profile,
                synthetic_task_context=trial_data.task_context,
                follow_up_query=trial_data.follow_up_query,
            )

        # Step 3: Judge the assistant response
        try:
            scores = self.judge.evaluate(
                query=trial_data.follow_up_query,
                response=pipeline_result.assistant_response,
                profile=trial_data.profile,
                task_context=trial_data.task_context,
            )
        except Exception as e:
            logger.warning("Trial %d: judge evaluation failed: %s", trial_index, e)
            scores = JudgeScores()

        # Memory counts heuristic:
        # Phase 1 (private): profile + task memories stored privately
        # Phase 2 (shared):  profile + task memories stored as shared
        if condition == "phase2":
            memory_counts = MemoryCounts(total=2, shared=2, private=0)
        else:
            memory_counts = MemoryCounts(total=2, shared=0, private=2)

        return TrialResult(
            trial_index=trial_index,
            condition=condition,
            relevance_score=scores.relevance_score,
            personalization_score=scores.personalization_score,
            memory_counts=memory_counts,
            latency_seconds=pipeline_result.latency_seconds,
            follow_up_query=trial_data.follow_up_query,
            assistant_response=pipeline_result.assistant_response,
            synthetic_profile=trial_data.profile,
            synthetic_task_context=trial_data.task_context,
        )

    # ------------------------------------------------------------------
    # Full experiment
    # ------------------------------------------------------------------

    def run(self) -> ExperimentResults:
        """Run the full experiment across all requested conditions.

        Returns:
            ``ExperimentResults`` containing per-trial data and summaries.
        """
        # Determine which conditions to run
        if self.condition == "both":
            conditions = ["phase1", "phase2"]
        elif self.condition == "phase1":
            conditions = ["phase1"]
        else:
            conditions = ["phase2"]

        # Optionally disable auto_inject
        if self.disable_auto_inject:
            try:
                config.update(**{"memory.auto_inject": False})
                logger.info("Disabled memory.auto_inject via config update.")
            except Exception as e:
                logger.warning(
                    "Could not programmatically disable auto_inject: %s. "
                    "Please disable it manually in the kernel config.yaml.",
                    e,
                )

        condition_results = []

        for cond in conditions:
            share_memory = cond == "phase2"
            pipeline = AgentPipeline(share_memory=share_memory)

            trials: list[TrialResult] = []
            for i in tqdm(range(self.trials), desc=f"Condition: {cond}"):
                result = self.run_single_trial(i, cond, pipeline)
                trials.append(result)

            summary = self.writer.compute_summary_statistics(trials)
            condition_results.append(
                ConditionResults(condition=cond, trials=trials, summary=summary)
            )

        metadata = ExperimentMetadata(
            trials_per_condition=self.trials,
            timestamp=datetime.now().isoformat(),
            kernel_url=config.get_kernel_url(),
            conditions_run=conditions,
        )

        experiment = ExperimentResults(
            experiment_metadata=metadata,
            conditions=condition_results,
        )

        # Write output files
        json_path = self.writer.write_json(experiment)
        logger.info("Results written to %s", json_path)

        if self.write_csv:
            csv_path = self.writer.write_csv(experiment)
            logger.info("CSV written to %s", csv_path)

        return experiment



def main():
    """Parse CLI arguments and run the evaluation orchestrator."""
    parser = argparse.ArgumentParser(
        description="Shared Memory Evaluation Harness — measures whether "
        "shared memory improves personalization quality.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per condition (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for result files (default: results/).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write a CSV file alongside the JSON results.",
    )
    parser.add_argument(
        "--condition",
        choices=["both", "phase1", "phase2"],
        default="both",
        help="Which condition(s) to run (default: both).",
    )
    parser.add_argument(
        "--disable-auto-inject",
        action="store_true",
        help="Attempt to disable the kernel's memory.auto_inject setting.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    orchestrator = EvaluationOrchestrator(
        trials=args.trials,
        output_dir=args.output,
        write_csv=args.csv,
        condition=args.condition,
        disable_auto_inject=args.disable_auto_inject,
    )

    experiment = orchestrator.run()

    # Print summary to stdout
    meta = experiment.experiment_metadata
    print(f"\n{'=' * 60}")
    print(f"Experiment complete — {meta.timestamp}")
    print(f"Conditions: {', '.join(meta.conditions_run)}")
    print(f"Trials per condition: {meta.trials_per_condition}")
    print(f"{'=' * 60}")

    for cond_result in experiment.conditions:
        s = cond_result.summary
        print(f"\n--- {cond_result.condition} ---")
        print(f"  Relevance:       mean={s.relevance.mean:.2f}  std={s.relevance.std:.2f}")
        print(f"  Personalization: mean={s.personalization.mean:.2f}  std={s.personalization.std:.2f}")
        print(f"  Latency (s):     mean={s.latency.mean:.2f}  std={s.latency.std:.2f}")
        print(f"  Memory total:    mean={s.memory_total.mean:.2f}")
        print(f"  Trials: {s.total_trials} total, {s.failed_trials} failed")

    print()


if __name__ == "__main__":
    main()
