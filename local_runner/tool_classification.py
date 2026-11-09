"""One classification of native control/receipt tools at the source boundary.

These results originate inside the current run. Classification preserves their
control protocol; it does not endorse their claims. Local file and unknown tool
results can contain fetched evidence and retain the external-content guard.
"""

INTERNAL_TOOLS = frozenset({
    "create_subagent", "assign_task", "collect_reports", "stop_subagent",
    "add_task", "update_task", "finish_planning", "review_research",
    "submit_report", "delegate_subtask", "check_tasks", "wait_for_subtasks",
    "list_subtasks",
})


def is_internal_tool(name: str) -> bool:
    return name in INTERNAL_TOOLS
