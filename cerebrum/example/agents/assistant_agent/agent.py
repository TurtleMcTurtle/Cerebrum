import os
import json

from cerebrum.llm.apis import llm_chat
from cerebrum.memory.apis import create_memory, search_memories
from cerebrum.config.config_manager import config

from cerebrum.example.agents.shared_memory_utils import (
    build_memory_metadata,
    filter_shared_memories,
    FIELD_OWNER_AGENT,
    MEMORY_TYPE_CONVERSATION,
    MEMORY_TYPE_PROFILE,
    MEMORY_TYPE_TASK_CONTEXT,
    POLICY_PRIVATE,
)

aios_kernel_url = config.get_kernel_url()


class AssistantAgent:
    """A personalized assistant agent that helps users with queries.

    Phase 1: Uses private memory and LLM for responses.
    Phase 2: Also retrieves shared profile/task memories for personalization.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.config = self.load_config()
        self.messages = []
        self.rounds = 0

    def load_config(self) -> dict:
        """Load agent configuration from config.json in the agent's directory."""
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        config_file = os.path.join(script_dir, "config.json")

        with open(config_file, "r") as f:
            config = json.load(f)
        return config

    def run(self, task_input: str) -> dict:
        """Process user query and return a result dictionary.

        Args:
            task_input: The user's query string.

        Returns:
            A dict with agent_name, result, and rounds.
        """
        try:
            # Build system instruction from config description
            system_instruction = "".join(self.config.get("description", []))

            # Phase 2: Retrieve shared context (additive, no-op if empty)
            shared_context = self._retrieve_shared_context()
            if shared_context:
                system_instruction = (
                    f"{system_instruction}\n\n"
                    f"Relevant context from other agents:\n{shared_context}"
                )

            self.messages.append({"role": "system", "content": system_instruction})

            # Append user query
            self.messages.append({"role": "user", "content": task_input})

            # Call LLM
            response = llm_chat(
                agent_name=self.agent_name,
                messages=self.messages,
                base_url=aios_kernel_url,
            )

            result_text = response["response"]["response_message"] if response else ""
            self.messages.append({"role": "assistant", "content": result_text})
            self.rounds += 1

            # Store conversation as private memory
            try:
                self._store_conversation_memory(
                    user_id=self.agent_name,
                    content=f"User: {task_input}\nAssistant: {result_text}",
                )
            except Exception:
                pass  # Memory storage failure is non-critical

            return {
                "agent_name": self.agent_name,
                "result": result_text,
                "rounds": self.rounds,
            }

        except Exception as e:
            return {
                "agent_name": self.agent_name,
                "result": f"Error: {e}",
                "rounds": self.rounds,
            }

    def _store_conversation_memory(self, user_id: str, content: str) -> None:
        """Store conversation turn as private memory.

        Args:
            user_id: Identifier for the user this memory pertains to.
            content: The conversation content to store.
        """
        metadata = build_memory_metadata(
            owner_agent=self.agent_name,
            user_id=user_id,
            memory_type=MEMORY_TYPE_CONVERSATION,
            sharing_policy=POLICY_PRIVATE,
        )
        create_memory(
            agent_name=self.agent_name,
            content=content,
            metadata=metadata,
            base_url=aios_kernel_url,
        )

    def _retrieve_shared_context(self) -> str:
        """Phase 2: Retrieve shared profile and task context memories.

        Calls search_memories twice (for profile and task_context types),
        filters for sharing_policy="shared" using filter_shared_memories,
        and formats results into a context string with owner_agent
        attribution.

        Returns:
            A formatted context string, or empty string if no shared
            memories are found or on error.
        """
        try:
            context_parts = []

            # Retrieve shared profile memories
            profile_response = search_memories(
                agent_name=self.agent_name,
                query="user profile preferences",
                base_url=aios_kernel_url,
            )
            profile_results = []
            if profile_response and isinstance(profile_response, dict):
                resp = profile_response.get("response", {})
                if resp and isinstance(resp, dict):
                    profile_results = resp.get("search_results", []) or []

            shared_profiles = filter_shared_memories(
                profile_results,
                memory_type=MEMORY_TYPE_PROFILE,
                exclude_owner=self.agent_name,
            )
            for mem in shared_profiles:
                owner = mem.get("metadata", {}).get(FIELD_OWNER_AGENT, "unknown")
                content = mem.get("content", "")
                if content:
                    context_parts.append(
                        f"[Profile from {owner}]: {content}"
                    )

            # Retrieve shared task context memories
            task_response = search_memories(
                agent_name=self.agent_name,
                query="current task context goals",
                base_url=aios_kernel_url,
            )
            task_results = []
            if task_response and isinstance(task_response, dict):
                resp = task_response.get("response", {})
                if resp and isinstance(resp, dict):
                    task_results = resp.get("search_results", []) or []

            shared_tasks = filter_shared_memories(
                task_results,
                memory_type=MEMORY_TYPE_TASK_CONTEXT,
                exclude_owner=self.agent_name,
            )
            for mem in shared_tasks:
                owner = mem.get("metadata", {}).get(FIELD_OWNER_AGENT, "unknown")
                content = mem.get("content", "")
                if content:
                    context_parts.append(
                        f"[Task context from {owner}]: {content}"
                    )

            return "\n".join(context_parts)

        except Exception:
            return ""
