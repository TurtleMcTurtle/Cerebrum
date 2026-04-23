# System-Wide Personalization via Controlled Shared Memory

## Overview

System-wide personalization is a multi-agent architecture pattern where specialized agents independently capture different facets of a user's identity and working context, then selectively share those memories through a centralized kernel memory layer so that any consuming agent can deliver personalized responses — without direct inter-agent communication or a monolithic user model.

This document describes the architecture, agents, memory model, and sharing mechanism implemented in the Cerebrum AIOS Agent SDK.

## Motivation

Traditional single-agent personalization stores all user context in one agent's local memory. This creates several problems:

- **Monolithic coupling** — One agent must handle profile extraction, task tracking, and response generation, making it hard to test or improve any single capability in isolation.
- **No separation of concerns** — Profile data (stable, long-lived) and task context (volatile, short-lived) are mixed in the same memory store with no semantic distinction.
- **Opaque personalization** — When a response is personalized, it's unclear whether the personalization came from stored preferences, recent task context, or the LLM's own inference.

Shared memory solves these problems by decomposing personalization into independent, testable agents that communicate through explicit, metadata-tagged memory items.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AIOS Kernel                          │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │            Kernel Memory Layer                    │  │
│  │                                                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │  │
│  │  │ Profile  │  │  Task    │  │ Conversation │   │  │
│  │  │ Memories │  │ Context  │  │  Memories    │   │  │
│  │  │ (shared) │  │ Memories │  │  (private)   │   │  │
│  │  │          │  │ (shared) │  │              │   │  │
│  │  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │  │
│  │       │              │               │           │  │
│  └───────┼──────────────┼───────────────┼───────────┘  │
│          │              │               │              │
└──────────┼──────────────┼───────────────┼──────────────┘
           │              │               │
     ┌─────┴─────┐ ┌─────┴─────┐ ┌───────┴───────┐
     │  Profile  │ │   Task    │ │   Assistant   │
     │   Agent   │ │   Agent   │ │     Agent     │
     │           │ │           │ │               │
     │ Extracts  │ │ Extracts  │ │ Responds to   │
     │ stable    │ │ working   │ │ user queries  │
     │ user      │ │ context   │ │ with shared   │
     │ attributes│ │           │ │ context       │
     └───────────┘ └───────────┘ └───────────────┘
```

### Key Principles

1. **Private by default** — All memories are created with `sharing_policy="private"`. An agent must explicitly opt in to sharing.
2. **Kernel-mediated sharing** — Shared memories live in the kernel memory layer, not in agent-local state. Agents never communicate directly.
3. **Metadata-driven visibility** — Each memory carries structured metadata (`owner_agent`, `user_id`, `memory_type`, `sharing_policy`) that controls who can see it.
4. **Graceful degradation** — If shared memories are unavailable, agents fall back to their own private memory without error.

## The Three Agents

### Profile Agent

**Location:** `cerebrum/example/agents/profile_agent/`

**Purpose:** Captures relatively stable user attributes that change infrequently.

**Extracted fields:**

| Field | Type | Example |
|-------|------|---------|
| `user_name` | string | "Alice Chen" |
| `preferred_tools` | list of strings | ["vim", "pytest", "docker"] |
| `preferred_language` | string | "Python" |
| `response_style` | string | "concise" |

**How it works:**
1. Receives natural-language input describing the user (e.g., "My name is Alice, I prefer Python and vim")
2. Calls `llm_chat_with_json_output` with a structured JSON schema to extract profile fields
3. Searches existing memories to avoid duplicates (`search_memories`)
4. Creates new memory or updates existing one (`create_memory` / `update_memory`)
5. When `share_memory` is enabled, memories are created with `sharing_policy="shared"`

**Memory type:** `"profile"`

### Task Agent

**Location:** `cerebrum/example/agents/task_agent/`

**Purpose:** Captures short-to-medium-term working context that changes as the user's focus shifts.

**Extracted fields:**

| Field | Type | Example |
|-------|------|---------|
| `current_project` | string | "AIOS v2 migration" |
| `active_experiment` | string | "testing mem0 provider" |
| `goals` | list of strings | ["finish migration by Friday"] |
| `blockers` | list of strings | ["API rate limits"] |
| `next_steps` | list of strings | ["run benchmarks", "update docs"] |

**How it works:** Same pattern as Profile Agent, but with task context fields and `memory_type="task_context"`.

### Assistant Agent

**Location:** `cerebrum/example/agents/assistant_agent/`

**Purpose:** The user-facing agent that responds to queries, enriched with shared context from the other agents.

**How it works:**
1. Calls `_retrieve_shared_context()` to search the kernel for shared memories
2. Filters results using `filter_shared_memories()` for `sharing_policy="shared"` items
3. Separates results by `memory_type` (profile vs task_context)
4. Formats shared context with `owner_agent` attribution into a context string
5. Prepends the shared context to the system prompt before calling `llm_chat`
6. Stores its own conversation as private memory

**Fallback behavior:** If `_retrieve_shared_context()` fails or returns nothing, the agent proceeds normally with just its own context — no error, no interruption.

## Memory Model

### Memory Metadata Schema

Every memory item created by any agent carries this metadata:

```python
{
    "owner_agent": "profile_agent",      # Who created it
    "user_id": "alice",                  # Who it's about
    "memory_type": "profile",            # What kind of data
    "sharing_policy": "shared"           # Who can see it
}
```

### Sharing Policy Values

| Value | Meaning |
|-------|---------|
| `"private"` | Only the creating agent can access this memory (default) |
| `"shared"` | Any agent can discover this memory via `search_memories` |

### Memory Types

| Type | Produced By | Consumed By | Lifetime |
|------|------------|-------------|----------|
| `"profile"` | Profile Agent | Assistant Agent | Long-lived (stable attributes) |
| `"task_context"` | Task Agent | Assistant Agent | Medium-lived (changes with projects) |
| `"conversation"` | Assistant Agent | Assistant Agent only | Short-lived (per-session) |

## Sharing Mechanism

### How Memories Become Shared

There are two ways to enable sharing:

**1. CLI flag (`--share-memory`):**
```bash
run-agent --mode local --agent_path cerebrum/example/agents/profile_agent \
  --task "My name is Alice, I prefer Python" --share-memory
```

When `--share-memory` is passed, the `AgentRunner` sets `agent.share_memory = True` on the agent instance before calling `run()`. The agent reads this via `getattr(self, 'share_memory', False)` and passes `sharing_policy=POLICY_SHARED` to `build_memory_metadata()`.

**2. Programmatic sharing after creation:**
```python
from cerebrum.example.agents.profile_agent.agent import ProfileAgent

agent = ProfileAgent("profile_agent")
result = agent.run("My name is Alice, I prefer Python")
# result contains memory IDs

agent.share_memory("mem_abc123")    # Make specific memory shared
agent.revoke_sharing("mem_abc123")  # Revoke sharing later
```

### How the Assistant Consumes Shared Memories

```python
def _retrieve_shared_context(self) -> str:
    # 1. Search for shared profile memories
    profile_results = search_memories(agent_name=self.agent_name, query="user profile preferences")
    shared_profiles = filter_shared_memories(profile_results, memory_type="profile")

    # 2. Search for shared task context memories
    task_results = search_memories(agent_name=self.agent_name, query="current task context goals")
    shared_tasks = filter_shared_memories(task_results, memory_type="task_context")

    # 3. Format with attribution
    # "[Profile from profile_agent]: {content}"
    # "[Task context from task_agent]: {content}"
```

The formatted context is prepended to the system prompt:

```
You are a personalized assistant agent. You help users with their queries...

Relevant context from other agents:
[Profile from profile_agent]: {"user_name": "Alice", "preferred_tools": ["vim", "pytest"], ...}
[Task context from task_agent]: {"current_project": "AIOS v2 migration", ...}
```

## Shared Utilities Module

**Location:** `cerebrum/example/agents/shared_memory_utils.py`

Provides constants and helpers used by all three agents:


### Constants

```python
# Field names
FIELD_OWNER_AGENT = "owner_agent"
FIELD_USER_ID = "user_id"
FIELD_MEMORY_TYPE = "memory_type"
FIELD_SHARING_POLICY = "sharing_policy"

# Sharing policies
POLICY_PRIVATE = "private"
POLICY_SHARED = "shared"

# Memory types
MEMORY_TYPE_CONVERSATION = "conversation"
MEMORY_TYPE_PROFILE = "profile"
MEMORY_TYPE_TASK_CONTEXT = "task_context"
```

### Helper Functions

**`build_memory_metadata(owner_agent, user_id, memory_type, sharing_policy="private", **extra)`**

Constructs a metadata dict with all required fields. Accepts additional provider-specific keys (e.g., `agent_id` for mem0).

**`filter_shared_memories(search_results, memory_type=None, exclude_owner=None)`**

Filters search results to only include items where `sharing_policy="shared"`, optionally restricting by `memory_type` and excluding a specific `owner_agent`.

## End-to-End Workflow

### Phase 1: Private Memory (Agent Isolation)

```
User → "My name is Alice, I use Python and vim"
         │
         ▼
   ProfileAgent.run()
         │
         ▼
   create_memory(content, metadata={
       owner_agent: "profile_agent",
       sharing_policy: "private",    ← private by default
       memory_type: "profile"
   })
```

Each agent operates independently. The Assistant Agent has no access to profile or task memories from other agents.

### Phase 2: Shared Memory (Cross-Agent Personalization)

```
User → "My name is Alice, I use Python and vim"
         │
         ▼
   ProfileAgent.run()  (with share_memory=True)
         │
         ▼
   create_memory(content, metadata={
       owner_agent: "profile_agent",
       sharing_policy: "shared",     ← explicitly shared
       memory_type: "profile"
   })

         ... later ...

User → "Help me with my current project"
         │
         ▼
   AssistantAgent.run()
         │
         ├─→ _retrieve_shared_context()
         │     ├─→ search_memories("user profile preferences")
         │     │     └─→ finds shared profile from ProfileAgent
         │     ├─→ search_memories("current task context goals")
         │     │     └─→ finds shared task context from TaskAgent
         │     └─→ returns formatted context string
         │
         ├─→ Prepend shared context to system prompt
         │
         ├─→ llm_chat(enriched_messages)
         │
         └─→ "Based on your Python expertise and vim setup,
              here's how to debug your AIOS v2 migration..."
```

## Design Properties

The system guarantees the following properties:

1. **Memory metadata completeness** — Every memory item always contains `owner_agent`, `user_id`, `memory_type`, and `sharing_policy`.
2. **Private by default** — If `sharing_policy` is not explicitly set to `"shared"`, it defaults to `"private"`.
3. **Upsert semantics** — Profile and Task agents search before creating, updating existing memories rather than duplicating.
4. **Filter correctness** — `filter_shared_memories` returns only items where `sharing_policy="shared"`, preserving `owner_agent` attribution.
5. **Graceful degradation** — Shared memory retrieval failures are caught and logged; the Assistant Agent continues with private memory only.
6. **Backward compatibility** — The `run(task_input)` signature is unchanged. Agents that don't know about `share_memory` default to private.

## Interaction with Kernel Auto-Inject

The AIOS kernel has its own memory personalization mechanism (`memory.auto_inject`). When enabled, the kernel independently retrieves and injects relevant memories into LLM calls before the agent sees them.

| Configuration | Agent-Level Sharing | Kernel Auto-Inject | Personalization Source |
|--------------|--------------------|--------------------|----------------------|
| Phase 1 + auto_inject off | Private only | Disabled | None (baseline) |
| Phase 1 + auto_inject on | Private only | Enabled | Kernel injection only |
| Phase 2 + auto_inject off | Shared | Disabled | Agent-level shared memory only |
| Phase 2 + auto_inject on | Shared | Enabled | Both (overlapping) |

For controlled experiments, disable `auto_inject` to isolate the effect of agent-level shared memory.

## File Structure

```
cerebrum/example/agents/
├── shared_memory_utils.py          # Constants and helpers
├── assistant_agent/
│   ├── agent.py                    # AssistantAgent class
│   ├── config.json                 # Agent metadata
│   └── meta_requirements.txt
├── profile_agent/
│   ├── agent.py                    # ProfileAgent class
│   ├── config.json
│   └── meta_requirements.txt
└── task_agent/
    ├── agent.py                    # TaskAgent class
    ├── config.json
    └── meta_requirements.txt
```

## Quick Start

```bash
# 1. Start the AIOS kernel
# (from the AIOS repo)
python -m aios.core.server

# 2. Store a user profile (shared)
run-agent --mode local --agent_path cerebrum/example/agents/profile_agent \
  --task "My name is Alice, I prefer Python, I use vim and pytest, I like concise responses" \
  --share-memory

# 3. Store working context (shared)
run-agent --mode local --agent_path cerebrum/example/agents/task_agent \
  --task "I'm working on the AIOS v2 migration, testing the mem0 provider, goal is to finish by Friday" \
  --share-memory

# 4. Ask the assistant (it picks up shared context automatically)
run-agent --mode local --agent_path cerebrum/example/agents/assistant_agent \
  --task "Help me debug my current project"
```

The Assistant Agent's response will reference Alice's Python/vim preferences and the AIOS v2 migration context — demonstrating system-wide personalization through controlled cross-agent shared memory.
