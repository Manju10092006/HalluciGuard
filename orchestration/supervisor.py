from __future__ import annotations

from typing import Any

ACTIVE_AGENTS = ("detector", "verifier", "memory")
DISABLED_AGENTS = ("judge", "corrector")


def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
    """Advance the control plane without making factual decisions."""
    phase = state.get("supervisor_phase", "start")
    if phase == "start":
        route = "detector"
    elif phase == "after_detector":
        if state.get("route") == "verify":
            route = "verifier"
        elif state.get("route") == "accept":
            route = "memory"
        else:
            route = "failure"
    elif phase == "after_verifier":
        route = "memory" if state.get("verification_status") == "verified" else "failure"
    elif phase == "after_memory":
        route = "end"
    else:
        route = "failure"

    trace = list(state.get("trace", []))
    trace.append(
        {
            "execution_id": state.get("execution_id", ""),
            "node": "supervisor",
            "status": "completed",
            "timestamp": state.get("updated_at"),
            "retry_count": int(state.get("retry_count", 0)),
            "details": {
                "phase": phase,
                "route": route,
                "active_agents": list(ACTIVE_AGENTS),
                "disabled_agents": list(DISABLED_AGENTS),
            },
        }
    )
    return {"supervisor_route": route, "trace": trace}


def supervisor_route(state: dict[str, Any]) -> str:
    route = state.get("supervisor_route")
    return route if route in {"detector", "verifier", "memory", "failure", "end"} else "failure"
