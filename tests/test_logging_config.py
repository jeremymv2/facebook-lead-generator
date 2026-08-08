import json
import logging
from pathlib import Path

import pytest

from lead_agent.config import Settings
from lead_agent.logging_config import JsonFormatter, configure_logging


def test_json_formatter_emits_context_fields() -> None:
    record = logging.LogRecord(
        name="lead_agent.scanner",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Post discovered",
        args=(),
        exc_info=None,
    )
    record.action = "post.discovered"
    record.post_id = 42
    record.group = "louisville-homeowners"
    record.result = "new"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["component"] == "lead_agent.scanner"
    assert payload["action"] == "post.discovered"
    assert payload["post_id"] == 42
    assert payload["severity"] == "INFO"


def test_configure_logging_uses_requested_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, log_level="warning", log_json=False)

    configure_logging(settings)

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1
