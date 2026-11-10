# Agent State Checkpoints: Deterministic Multi-Agent State Persistence

> **Author:** [Vaibhav Kadam](https://github.com/vaibhav7087) | Co-founder @ Bylance • AI Intern @ Eldea Solutions  
> **Stack:** LangGraph, Python 3.11+, Pydantic v2, SQLite, Docker  
> **Architecture:** Explicit Directed Acyclic State Graphs (`StateGraph`) with ACID-compliant SQLite Checkpoint Persistence and Isolated Supervisor Error Boundaries.

---

## 1. Architectural Motivation

A common anti-pattern when developers first build autonomous agents is wrapping a Large Language Model in an open `while True:` loop.

While this works for simple prototypes, in multi-step production environments three catastrophic failure modes emerge:
1. **Unbounded Retry Loops:** An intermittent network or tool error causes runaway retries, burning API token budgets within minutes (e.g. the $14 infinite loop post-mortem).
2. **Context Window Contamination:** Accumulating raw exception traces bloats the prompt context until argument parsing degrades into hallucinations.
3. **Zero Process Durability:** If the host runtime restarts mid-execution, all progress is lost, forcing destructive tool calls (payments, emails, database mutations) to re-run from scratch.

This repository implements **deterministic state machines** using **LangGraph** and **SQLite checkpointing**, establishing strict software boundaries around non-deterministic AI models.

---

## 2. Key Features

- **Typed State Representation:** Uses `TypedDict` and Pydantic `BaseModel` schemas for state verification at every node edge.
- **Durable SQLite Checkpoints:** Every state transition is serialized to local or edge SQLite (`checkpoints.db`) using `SqliteSaver`.
- **Thread-Aware Resumption:** Workflows can pause indefinitely for human-in-the-loop review and resume seamlessly by `thread_id`.
- **Isolated Error Boundary:** Third-party timeouts trigger clean state rewind, pruning raw error dumps into structured JSON codes before re-prompting.

---

## 3. Quickstart

### Prerequisites
- Python 3.11+
- Virtualenv or Docker

### Installation
```bash
git clone https://github.com/vaibhav7087/agent-state-checkpoints.git
cd agent-state-checkpoints
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the State Machine
```bash
python -m src.state_machine
```

---

## 4. Architecture Breakdown

```
+───────────────────+      (Valid Request)      +───────────────────+
|   parse_intent    | ────────────────────────> |   execute_tool    |
+───────────────────+                           +───────────────────+
          |                                               |
          | (Schema Error)                                | (Timeout / Error)
          v                                               v
+───────────────────+                           +───────────────────+
|  handle_schema_err|                           | supervisor_error_ |
|                   | <──────────────────────── |      boundary     |
+───────────────────+   (Rewind to Checkpoint)  +───────────────────+
```

### Isolated Error Boundary Implementation:
```python
def handle_tool_error(state: AgentState) -> dict:
    clean_error = {
        "code": "TIMEOUT_504",
        "retries_left": state["retry_budget"] - 1,
        "detail": "Upstream service timeout; rewound to last healthy checkpoint."
    }
    return {
        "messages": [ToolMessage(content=json.dumps(clean_error))],
        "current_node": "recover_checkpoint",
        "retry_budget": state["retry_budget"] - 1
    }
```

---

## 5. License
MIT License. Built by Vaibhav Kadam.
