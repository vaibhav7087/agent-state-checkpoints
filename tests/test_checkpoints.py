"""Unit tests for deterministic state checkpointing"""

import pytest
from src.state_machine import (
    AgentState,
    parse_intent_node,
    execute_tool_node,
    supervisor_error_boundary,
    render_response_node
)

def test_parse_intent_transitions():
    state: AgentState = {
        "messages": [],
        "current_node": "init",
        "retry_budget": 3,
        "thread_id": "test_1",
        "last_checkpoint_node": None
    }
    updated = parse_intent_node(state)
    assert updated["current_node"] == "execute_tool"
    assert updated["last_checkpoint_node"] == "parse_intent"

def test_supervisor_error_boundary_pruning():
    state: AgentState = {
        "messages": [],
        "current_node": "execute_tool",
        "retry_budget": 3,
        "thread_id": "test_1",
        "last_checkpoint_node": "parse_intent"
    }
    boundary_state = supervisor_error_boundary(state)
    assert boundary_state["retry_budget"] == 2
    assert boundary_state["current_node"] == "execute_tool"
    assert "TIMEOUT_504" in boundary_state["messages"][0]["content"]

def test_full_recovery_flow():
    state: AgentState = {
        "messages": [{"role": "user", "content": "Fetch records"}],
        "current_node": "parse_intent",
        "retry_budget": 1,  # Allows successful execute_tool
        "thread_id": "test_2",
        "last_checkpoint_node": "parse_intent"
    }
    res = execute_tool_node(state)
    assert res["current_node"] == "render_response"
    
    final_res = render_response_node(res)
    assert final_res["current_node"] == "completed"
