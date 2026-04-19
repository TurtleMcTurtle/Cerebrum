"""Synthetic data generation for the shared memory evaluation harness.

Uses LLM calls via ``llm_chat_with_json_output`` to produce unique
synthetic profiles, task contexts, and follow-up queries for each trial.
"""

import json
import logging
from typing import Dict, Any, List

from cerebrum.llm.apis import llm_chat_with_json_output
from cerebrum.config.config_manager import config

from benchmarks.shared_memory.models import (
    SyntheticProfile,
    SyntheticTaskContext,
    SyntheticTrialData,
)

logger = logging.getLogger(__name__)


def _unwrap_nested(data: dict, required_keys: List[str]) -> dict:
    """Unwrap a potentially nested LLM response to find the expected keys.

    Some LLMs wrap the response in an extra layer like
    ``{"developer": {"user_name": ...}}``. This function checks if the
    required keys are present at the top level; if not, it looks one
    level deeper for a dict value that contains them.

    Args:
        data: Parsed JSON dict from the LLM.
        required_keys: Keys that must be present in the result.

    Returns:
        A dict containing the required keys (may be the original or
        an inner dict).
    """
    if all(k in data for k in required_keys):
        return data
    # Try one level deeper
    for value in data.values():
        if isinstance(value, dict) and all(k in value for k in required_keys):
            return value
    # Give up — return original and let Pydantic raise a clear error
    return data


class SyntheticDataGenerator:
    """Generates synthetic trial data (profile, task context, query) via LLM."""

    def __init__(self, agent_name: str = "eval_harness"):
        """Initialise the generator.

        Args:
            agent_name: Agent identity used for SDK LLM calls.
        """
        self.agent_name = agent_name
        self.kernel_url = config.get_kernel_url()

    # ------------------------------------------------------------------
    # Profile generation
    # ------------------------------------------------------------------

    def generate_profile(self, trial_index: int) -> SyntheticProfile:
        """Generate a synthetic user profile for a single trial.

        Args:
            trial_index: Zero-based trial number, included in the prompt
                to encourage diversity across trials.

        Returns:
            A validated ``SyntheticProfile`` instance.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data generator. You MUST return a flat JSON "
                    "object with exactly these keys: user_name, "
                    "preferred_tools, preferred_language, response_style. "
                    "Do NOT nest the object inside another key."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate a realistic software developer profile for "
                    f"trial #{trial_index}. Return a JSON object with:\n"
                    f'- "user_name": a realistic full name\n'
                    f'- "preferred_tools": a list of 2-5 developer tool names\n'
                    f'- "preferred_language": a programming language\n'
                    f'- "response_style": one of "concise", "detailed", '
                    f'"casual", or "formal"'
                ),
            },
        ]

        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "synthetic_profile",
                "schema": {
                    "type": "object",
                    "properties": {
                        "user_name": {"type": "string"},
                        "preferred_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "preferred_language": {"type": "string"},
                        "response_style": {"type": "string"},
                    },
                    "required": [
                        "user_name",
                        "preferred_tools",
                        "preferred_language",
                        "response_style",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        llm_response = llm_chat_with_json_output(
            agent_name=self.agent_name,
            messages=messages,
            base_url=self.kernel_url,
            response_format=response_format,
        )

        raw = llm_response["response"]["response_message"]
        data = json.loads(raw) if isinstance(raw, str) else raw
        data = _unwrap_nested(data, ["user_name", "preferred_tools", "preferred_language", "response_style"])
        return SyntheticProfile(**data)

    # ------------------------------------------------------------------
    # Task-context generation
    # ------------------------------------------------------------------

    def generate_task_context(
        self, trial_index: int, profile: SyntheticProfile
    ) -> SyntheticTaskContext:
        """Generate a synthetic task context informed by the profile.

        Args:
            trial_index: Zero-based trial number for diversity.
            profile: The previously generated profile for this trial.

        Returns:
            A validated ``SyntheticTaskContext`` instance.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data generator. You MUST return a flat JSON "
                    "object with exactly these keys: current_project, "
                    "active_experiment, goals, blockers, next_steps. "
                    "Do NOT nest the object inside another key."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate a realistic working context for a developer "
                    f"named {profile.user_name} who uses "
                    f"{profile.preferred_language} for trial #{trial_index}. "
                    f"Return a JSON object with:\n"
                    f'- "current_project": name of the project\n'
                    f'- "active_experiment": what they are currently testing\n'
                    f'- "goals": list of 2-4 goal strings\n'
                    f'- "blockers": list of 0-2 blocker strings\n'
                    f'- "next_steps": list of 2-4 next step strings'
                ),
            },
        ]

        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "synthetic_task_context",
                "schema": {
                    "type": "object",
                    "properties": {
                        "current_project": {"type": "string"},
                        "active_experiment": {"type": "string"},
                        "goals": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "blockers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "next_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "current_project",
                        "active_experiment",
                        "goals",
                        "blockers",
                        "next_steps",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        llm_response = llm_chat_with_json_output(
            agent_name=self.agent_name,
            messages=messages,
            base_url=self.kernel_url,
            response_format=response_format,
        )

        raw = llm_response["response"]["response_message"]
        data = json.loads(raw) if isinstance(raw, str) else raw
        data = _unwrap_nested(data, ["current_project", "active_experiment", "goals", "blockers", "next_steps"])
        return SyntheticTaskContext(**data)

    # ------------------------------------------------------------------
    # Follow-up query generation
    # ------------------------------------------------------------------

    def generate_follow_up_query(
        self,
        profile: SyntheticProfile,
        task_context: SyntheticTaskContext,
    ) -> str:
        """Generate a natural follow-up query that benefits from personalisation.

        Args:
            profile: The synthetic profile for this trial.
            task_context: The synthetic task context for this trial.

        Returns:
            A plain-text query string.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data generator. You MUST return a flat JSON "
                    'object with exactly one key: "query". '
                    "Do NOT nest the object inside another key."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate a natural question that {profile.user_name} "
                    f"would ask about their project "
                    f"'{task_context.current_project}' that would benefit "
                    f"from knowing their tool preferences "
                    f"({', '.join(profile.preferred_tools)}) and current "
                    f"goals ({', '.join(task_context.goals)}). "
                    f'Return JSON: {{"query": "your question here"}}'
                ),
            },
        ]

        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "follow_up_query",
                "schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        llm_response = llm_chat_with_json_output(
            agent_name=self.agent_name,
            messages=messages,
            base_url=self.kernel_url,
            response_format=response_format,
        )

        raw = llm_response["response"]["response_message"]
        data = json.loads(raw) if isinstance(raw, str) else raw
        data = _unwrap_nested(data, ["query"])
        return data["query"]

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def generate_trial_data(self, trial_index: int) -> SyntheticTrialData:
        """Generate all synthetic data for a single trial.

        Orchestrates profile → task context → follow-up query generation.

        Args:
            trial_index: Zero-based trial number.

        Returns:
            A ``SyntheticTrialData`` bundle with profile, task context,
            and follow-up query.
        """
        profile = self.generate_profile(trial_index)
        task_context = self.generate_task_context(trial_index, profile)
        follow_up_query = self.generate_follow_up_query(profile, task_context)

        return SyntheticTrialData(
            profile=profile,
            task_context=task_context,
            follow_up_query=follow_up_query,
        )
