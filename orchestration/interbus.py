from __future__ import annotations

import uuid
from typing import Any, Literal, TypedDict

from .state import HalluciGuardState, utc_now

MessageStatus = Literal["created", "delivered", "failed"]


class AgentMessage(TypedDict, total=False):
    message_id: str
    execution_id: str
    source_agent: str
    target_agent: str
    message_type: str
    payload: dict[str, Any]
    timestamp: str
    status: MessageStatus


def publish_message(
    state: HalluciGuardState,
    *,
    source_agent: str,
    target_agent: str,
    message_type: str,
    payload: dict[str, Any],
    status: MessageStatus = "delivered",
) -> list[AgentMessage]:
    messages = list(state.get("inter_agent_bus", []))
    messages.append(
        {
            "message_id": str(uuid.uuid4()),
            "execution_id": state.get("execution_id", state.get("request_id", "")),
            "source_agent": source_agent,
            "target_agent": target_agent,
            "message_type": message_type,
            "payload": payload,
            "timestamp": utc_now(),
            "status": status,
        }
    )
    return messages
