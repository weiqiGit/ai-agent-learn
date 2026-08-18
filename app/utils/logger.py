import json
import os
from datetime import datetime, timezone
from typing import Any, Literal

LOG_DIR = os.getenv("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
JSON_LOG_FILE = os.path.join(LOG_DIR, "structured.log")


LogType = Literal["request", "tool", "memory", "error", "info"]


class StructuredLogger:
    def log(self, log_type: LogType, data: dict[str, Any]):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            log_type: log_type,
            **data,
        }
        with open(JSON_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


logger = StructuredLogger()
