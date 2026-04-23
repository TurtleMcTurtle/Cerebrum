"""Shared Memory Evaluation Harness.

Benchmark tool that quantitatively measures whether shared memory improves
personalization quality in a multi-agent system. Compares Phase 1 (private
memory only) against Phase 2 (shared memory enabled) across synthetic trials.

Usage:
    python benchmarks/shared_memory/run_evaluation.py --trials 10 --output results/

Note on kernel auto_inject:
    The AIOS kernel's ``memory.auto_inject`` setting independently injects
    relevant memories into LLM calls. For controlled experiments isolating
    the effect of agent-level shared memory, consider disabling auto_inject
    in the kernel config or using the ``--disable-auto-inject`` flag.
"""
