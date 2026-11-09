"""Explicit comparison profiles preserve native model and runtime settings."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from workflows.agent_team.profile import load_swarm_profile


def local_profile(name, *, overrides=None):
    path = Path(__file__).parents[1] / "local_profiles" / f"{name}.yaml"
    return load_swarm_profile(
        "benchmark", inline=yaml.safe_load(path.read_text(encoding="utf-8")),
        overrides=overrides,
    )


@pytest.mark.parametrize(
    ("name", "expected_agent_changes"),
    [
        ("effort_high_v1", {"team_effort": "high"}),
        ("shared_search_v1", {"share_search_results": True}),
        ("integrated_v1", {"team_effort": "high", "share_search_results": True}),
    ],
)
def test_comparison_profile_changes_only_declared_controls(name, expected_agent_changes):
    baseline = load_swarm_profile("benchmark")
    actual = local_profile(name)
    expected = deepcopy(baseline)
    expected["agent"].update(expected_agent_changes)
    assert actual == expected
    assert baseline["agent"]["team_effort"] == "max"
    assert "share_search_results" not in baseline["agent"]


def test_model_reasoning_settings_are_independent_of_team_effort():
    overrides = {
        "llm": {
            "extra_body": {
                "top_k": None,
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            },
        },
        "agent": {"reporter": False, "thinking_format": "reasoning_content"},
    }
    baseline = load_swarm_profile("benchmark", overrides=overrides)
    adjusted = local_profile("effort_high_v1", overrides=overrides)
    assert adjusted["llm"] == baseline["llm"]
    assert adjusted["llm"]["extra_body"]["thinking"] == {"type": "enabled"}
    assert adjusted["llm"]["extra_body"]["reasoning_effort"] == "high"
    assert adjusted["agent"]["team_effort"] == "high"
