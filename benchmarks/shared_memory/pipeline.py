"""Agent execution pipeline for the shared memory evaluation harness.

Handles loading agents locally and running them in sequence:
ProfileAgent → TaskAgent → AssistantAgent. The share_memory flag
controls whether agents store memories as shared or private,
corresponding to the Phase 2 and Phase 1 experimental conditions.
"""

import json
import time
from dataclasses import dataclass

from cerebrum.example.agents.profile_agent.agent import ProfileAgent
from cerebrum.example.agents.task_agent.agent import TaskAgent
from cerebrum.example.agents.assistant_agent.agent import AssistantAgent

from benchmarks.shared_memory.models import SyntheticTrialData


@dataclass
class PipelineResult:
    """Result from a single trial's agent pipeline execution."""

    profile_result: dict
    task_result: dict
    assistant_result: dict
    assistant_response: str
    latency_seconds: float


class AgentPipeline:
    """Runs the three-agent pipeline for a single trial.

    Instantiates ProfileAgent, TaskAgent, and AssistantAgent in sequence,
    configuring the share_memory attribute based on the experimental
    condition. Measures AssistantAgent latency for metric collection.

    Args:
        share_memory: If True, agents use sharing_policy="shared" (Phase 2).
            If False, agents use sharing_policy="private" (Phase 1).
    """

    def __init__(self, share_memory: bool):
        self.share_memory = share_memory

    def run_trial(self, trial_data: SyntheticTrialData) -> PipelineResult:
        """Execute the full agent pipeline for one trial.

        Args:
            trial_data: The synthetic data for this trial, containing
                profile, task_context, and follow_up_query.

        Returns:
            PipelineResult with all agent outputs and assistant latency.
        """
        # Step 1: Run ProfileAgent with synthetic profile data
        profile_agent = ProfileAgent("profile_agent")
        profile_agent.share_memory = self.share_memory
        profile_result = profile_agent.run(
            json.dumps(trial_data.profile.model_dump())
        )

        # Step 2: Run TaskAgent with synthetic task context data
        task_agent = TaskAgent("task_agent")
        task_agent.share_memory = self.share_memory
        task_result = task_agent.run(
            json.dumps(trial_data.task_context.model_dump())
        )

        # Step 3: Run AssistantAgent with follow-up query, measuring latency
        assistant_agent = AssistantAgent("assistant_agent")
        assistant_agent.share_memory = self.share_memory

        start_time = time.time()
        assistant_result = assistant_agent.run(trial_data.follow_up_query)
        latency_seconds = time.time() - start_time

        # Extract the response text from the assistant result
        assistant_response = assistant_result.get("result", "")

        return PipelineResult(
            profile_result=profile_result,
            task_result=task_result,
            assistant_result=assistant_result,
            assistant_response=assistant_response,
            latency_seconds=latency_seconds,
        )
