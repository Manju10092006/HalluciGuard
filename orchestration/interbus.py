from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class AgentMessage:
    """Auditable in-process message exchanged through the LangGraph bus."""

    source_agent: str
    target_agent: str
    message_type: str
    payload: dict[str, Any]
    execution_id: str
    message_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "emitted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "execution_id": self.execution_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "status": self.status,
        }


def emit_message(
    state: dict[str, Any],
    *,
    source_agent: str,
    target_agent: str,
    message_type: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append one structured message to the shared LangGraph bus."""
    message = AgentMessage(
        source_agent=source_agent,
        target_agent=target_agent,
        message_type=message_type,
        payload=payload,
        execution_id=str(state.get("execution_id", "")),
    ).to_dict()
    bus = list(state.get("inter_agent_bus", []))
    bus.append(message)
    return bus
