"""Versioned evidence types. Numerical inputs are checked before persistence."""
from dataclasses import dataclass, field, asdict
from typing import Any
import json
import math
from datetime import datetime

SCHEMA_VERSION = 1
KINDS = {"occurrence", "quotation", "recall", "prediction", "imagination", "reflection"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def finite(value, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected finite number")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError("number outside allowed range")
    return float(value)


@dataclass(frozen=True)
class Event:
    id: str
    source: str
    actor: str
    target: str
    task: str
    timestamp: str
    kind: str = "occurrence"
    confidence: float = 1.0
    data: dict[str, Any] = field(default_factory=dict)
    causal_links: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        for key in ("id", "source", "actor", "target", "task", "timestamp"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"{key} must be a nonempty string")
        try:
            timestamp=datetime.fromisoformat(self.timestamp.replace("Z","+00:00"))
            if timestamp.tzinfo is None: raise ValueError("timezone required")
        except (ValueError,TypeError) as exc:
            raise ValueError("timestamp must be ISO 8601 with timezone") from exc
        if self.kind not in KINDS or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported event kind/schema")
        finite(self.confidence, 0, 1)
        if not isinstance(self.data, dict) or not isinstance(self.causal_links, list) or not all(isinstance(x,str) for x in self.causal_links):
            raise ValueError("invalid data/causal links")
        for key in ("quality", "approval", "control", "uncertainty", "responsibility"):
            if key in self.data:
                finite(self.data[key], -1 if key in {"quality", "approval"} else 0, 1)
        if not isinstance(self.data.get("goal_delta", {}),dict):
            raise ValueError("goal_delta must be a mapping")
        for v in self.data.get("goal_delta", {}).values():
            finite(v, -1, 1)
        canonical(asdict(self))

    def to_dict(self):
        return asdict(self)
