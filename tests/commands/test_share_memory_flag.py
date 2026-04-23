"""
Tests for the --share-memory CLI flag: parsing, AgentConfig defaults,
AgentRunner propagation, and __slots__ graceful degradation.

Run: python tests/commands/test_share_memory_flag.py

Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 3.1, 3.3, 3.4, 8.4
"""

import unittest
import logging
from unittest.mock import patch, MagicMock

from cerebrum.commands.run_agent import AgentConfig, AgentRunner, parse_arguments


class TestShareMemoryCLIParsing(unittest.TestCase):
    """Tests for --share-memory argument parsing."""

    def test_parse_args_with_share_memory_flag(self):
        """Parse args with --share-memory → config.share_memory is True.
        Validates: Requirements 1.1, 2.2
        """
        test_argv = [
            "run-agent",
            "--agent_path", "/tmp/fake_agent",
            "--mode", "local",
            "--share-memory",
        ]
        with patch("sys.argv", test_argv):
            config = parse_arguments()
        self.assertTrue(config.share_memory)

    def test_parse_args_without_share_memory_flag(self):
        """Parse args without --share-memory → config.share_memory is False.
        Validates: Requirements 1.2, 2.3
        """
        test_argv = [
            "run-agent",
            "--agent_path", "/tmp/fake_agent",
            "--mode", "local",
        ]
        with patch("sys.argv", test_argv):
            config = parse_arguments()
        self.assertFalse(config.share_memory)

    def test_agent_config_default_share_memory(self):
        """AgentConfig() default → share_memory is False.
        Validates: Requirement 2.1
        """
        config = AgentConfig()
        self.assertFalse(config.share_memory)


class TestAgentRunnerPropagation(unittest.TestCase):
    """Tests for AgentRunner propagating share_memory to agent instances."""

    def _make_runner(self, share_memory: bool) -> AgentRunner:
        """Create an AgentRunner with a minimal local-mode config."""
        config = AgentConfig(
            agent_path="/tmp/fake_agent",
            mode="local",
            share_memory=share_memory,
        )
        runner = AgentRunner(config)
        return runner

    def test_share_memory_set_on_agent_before_run(self):
        """Mock agent class, run AgentRunner.run() with share_memory=True
        → verify attribute set on agent before run() called.
        Validates: Requirements 3.1, 3.3
        """
        # Track the value of share_memory at the moment run() is called
        captured = {}

        class FakeAgent:
            def __init__(self, name):
                self.agent_name = name

            def run(self_agent, task_input):
                captured["share_memory"] = self_agent.share_memory
                return {"result": "ok"}

        runner = self._make_runner(share_memory=True)

        # Patch _load_local_agent to return our fake agent class + config dict
        with patch.object(runner, "_load_local_agent", return_value=(FakeAgent, {"name": "fake"})):
            with patch.object(runner, "_load_json_config", return_value={}):
                runner.run()

        self.assertTrue(captured["share_memory"])

    def test_share_memory_false_propagated(self):
        """AgentRunner.run() with share_memory=False → agent.share_memory is False.
        Validates: Requirements 3.1, 3.4
        """
        captured = {}

        class FakeAgent:
            def __init__(self, name):
                self.agent_name = name

            def run(self_agent, task_input):
                captured["share_memory"] = self_agent.share_memory
                return {"result": "ok"}

        runner = self._make_runner(share_memory=False)

        with patch.object(runner, "_load_local_agent", return_value=(FakeAgent, {"name": "fake"})):
            with patch.object(runner, "_load_json_config", return_value={}):
                runner.run()

        self.assertFalse(captured["share_memory"])

    def test_slots_agent_logs_warning_and_continues(self):
        """Mock agent class with __slots__ (no share_memory slot)
        → verify warning logged and execution completes.
        Validates: Requirement 8.4
        """

        class SlotsAgent:
            __slots__ = ("agent_name",)

            def __init__(self, name):
                self.agent_name = name

            def run(self, task_input):
                return {"result": "ok"}

        runner = self._make_runner(share_memory=True)

        with patch.object(runner, "_load_local_agent", return_value=(SlotsAgent, {"name": "slots_agent"})):
            with patch.object(runner, "_load_json_config", return_value={}):
                with self.assertLogs("cerebrum.commands.run_agent", level=logging.WARNING) as cm:
                    result = runner.run()

        # Verify warning was logged about __slots__
        self.assertTrue(
            any("share_memory" in msg and "__slots__" in msg for msg in cm.output),
            f"Expected warning about share_memory/__slots__, got: {cm.output}",
        )
        # Verify execution completed successfully
        self.assertEqual(result, {"result": "ok"})


if __name__ == "__main__":
    unittest.main()
