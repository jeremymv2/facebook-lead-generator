"""Structured application logging without third-party runtime dependencies."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import ClassVar

from lead_agent.config import Settings


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line using an intentionally small safe field set."""

    contextual_fields: ClassVar[tuple[str, ...]] = (
        "action",
        "lead_id",
        "post_id",
        "group",
        "group_id",
        "attempt",
        "error_code",
        "retry_in_seconds",
        "result",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "severity": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        for name in self.contextual_fields:
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def configure_logging(settings: Settings) -> None:
    """Configure the root logger once for the command being run."""
    handler = logging.StreamHandler()
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)
