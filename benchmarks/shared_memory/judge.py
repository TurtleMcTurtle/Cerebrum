"""LLM-as-judge evaluation for the shared memory evaluation harness.

Uses ``llm_chat_with_json_output`` to score assistant responses on
relevance and personalization using a structured rubric.
"""

import json
import logging
from typing import Any, Dict, List

from cerebrum.llm.apis import llm_chat_with_json_output
from cerebrum.config.config_manager import config

from benchmarks.shared_memory.models import (
    JudgeScores,
    SyntheticProfile,
    SyntheticTaskContext,
)
from benchmarks.shared_memory.synth import _unwrap_nested

logger = logging.getLogger(__name__)


def _normalize_judge_keys(data: dict) -> dict:
    """Normalize LLM judge response keys to the expected format.

    LLMs often return variations like "Relevance", "relevance",
    "relevance_score", etc. This maps them to the canonical keys.

    Args:
        data: Parsed JSON dict from the LLM judge.

    Returns:
        Dict with normalized keys.
    """
    key_map = {
        "relevance": "relevance_score",
        "relevance_score": "relevance_score",
        "personalization": "personalization_score",
        "personalization_score": "personalization_score",
        "relevance_reasoning": "relevance_reasoning",
        "personalization_reasoning": "personalization_reasoning",
        "reasoning": None,  # skip nested reasoning objects
    }
    normalized = {}
    for k, v in data.items():
        canonical = key_map.get(k.lower().strip())
        if canonical and not isinstance(v, (dict, list)):
            normalized[canonical] = v
        elif canonical is None:
            # Try to extract reasoning from nested dicts
            if isinstance(v, dict):
                for rk, rv in v.items():
                    rk_lower = rk.lower().strip()
                    if "relevance" in rk_lower and "relevance_reasoning" not in normalized:
                        normalized["relevance_reasoning"] = str(rv)
                    elif "personal" in rk_lower and "personalization_reasoning" not in normalized:
                        normalized["personalization_reasoning"] = str(rv)
            elif isinstance(v, list):
                # Some LLMs return reasoning as a list of strings
                normalized.setdefault("relevance_reasoning", str(v))
    return normalized


class LLMJudge:
    """Evaluates assistant responses using an LLM judge with a scoring rubric."""

    def __init__(self, agent_name: str = "eval_judge"):
        """Initialise the judge.

        Args:
            agent_name: Agent identity used for SDK LLM calls.
        """
        self.agent_name = agent_name
        self.kernel_url = config.get_kernel_url()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_judge_prompt(
        self,
        query: str,
        response: str,
        profile: SyntheticProfile,
        task_context: SyntheticTaskContext,
    ) -> List[Dict[str, str]]:
        """Build the messages list for the LLM judge call.

        Separated from ``evaluate`` so that prompt contents can be
        inspected and tested independently (Property 4).

        Args:
            query: The follow-up query posed to the assistant.
            response: The assistant's response text.
            profile: The synthetic user profile for this trial.
            task_context: The synthetic task context for this trial.

        Returns:
            A list of message dicts suitable for ``llm_chat_with_json_output``.
        """
        user_content = (
            "Evaluate the following AI assistant response.\n\n"
            "--- USER PROFILE ---\n"
            f"Name: {profile.user_name}\n"
            f"Preferred Tools: {', '.join(profile.preferred_tools)}\n"
            f"Preferred Language: {profile.preferred_language}\n"
            f"Response Style: {profile.response_style}\n\n"
            "--- TASK CONTEXT ---\n"
            f"Current Project: {task_context.current_project}\n"
            f"Active Experiment: {task_context.active_experiment}\n"
            f"Goals: {', '.join(task_context.goals)}\n"
            f"Blockers: {', '.join(task_context.blockers)}\n"
            f"Next Steps: {', '.join(task_context.next_steps)}\n\n"
            "--- FOLLOW-UP QUERY ---\n"
            f"{query}\n\n"
            "--- ASSISTANT RESPONSE ---\n"
            f"{response}\n\n"
            "--- SCORING RUBRIC ---\n"
            "Relevance (1-5):\n"
            "  5 = Directly and completely addresses the query\n"
            "  4 = Addresses the query with minor gaps\n"
            "  3 = Partially addresses the query\n"
            "  2 = Tangentially related to the query\n"
            "  1 = Does not address the query\n\n"
            "Personalization (1-5):\n"
            "  5 = Correctly references specific profile attributes AND task context details\n"
            "  4 = References most profile/task details correctly\n"
            "  3 = References some profile/task details\n"
            "  2 = Vague or incorrect references to profile/task\n"
            "  1 = No personalization evident\n\n"
            "Return your scores and reasoning as JSON."
        )

        return [
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator assessing the quality "
                    "of an AI assistant's response."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: str,
        response: str,
        profile: SyntheticProfile,
        task_context: SyntheticTaskContext,
    ) -> JudgeScores:
        """Score an assistant response on relevance and personalization.

        Args:
            query: The follow-up query posed to the assistant.
            response: The assistant's response text.
            profile: The synthetic user profile for this trial.
            task_context: The synthetic task context for this trial.

        Returns:
            A ``JudgeScores`` instance. On any failure the scores are
            ``None`` (the ``JudgeScores`` defaults).
        """
        messages = self._build_judge_prompt(query, response, profile, task_context)

        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_scores",
                "schema": {
                    "type": "object",
                    "properties": {
                        "relevance_score": {"type": "integer"},
                        "personalization_score": {"type": "integer"},
                        "relevance_reasoning": {"type": "string"},
                        "personalization_reasoning": {"type": "string"},
                    },
                    "required": [
                        "relevance_score",
                        "personalization_score",
                        "relevance_reasoning",
                        "personalization_reasoning",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        try:
            llm_response = llm_chat_with_json_output(
                agent_name=self.agent_name,
                messages=messages,
                base_url=self.kernel_url,
                response_format=response_format,
            )

            raw = llm_response["response"]["response_message"]
            data = json.loads(raw) if isinstance(raw, str) else raw
            data = _normalize_judge_keys(data)

            relevance = data.get("relevance_score")
            personalization = data.get("personalization_score")

            if relevance is None or personalization is None:
                logger.warning("Judge returned incomplete scores: %s", data)
                return JudgeScores()

            # Clamp scores to [1, 5]
            if not isinstance(relevance, int) or relevance < 1 or relevance > 5:
                logger.warning("Relevance score %s out of range, clamping", relevance)
                relevance = max(1, min(5, int(relevance)))
            if not isinstance(personalization, int) or personalization < 1 or personalization > 5:
                logger.warning("Personalization score %s out of range, clamping", personalization)
                personalization = max(1, min(5, int(personalization)))

            return JudgeScores(
                relevance_score=relevance,
                personalization_score=personalization,
                relevance_reasoning=data.get("relevance_reasoning"),
                personalization_reasoning=data.get("personalization_reasoning"),
            )

        except Exception as e:
            logger.warning("Judge evaluation failed: %s", e)
            return JudgeScores()
