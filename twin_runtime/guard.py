import json
from typing import Literal

import plyer

THRESH = 0.65


def risk_gate(msg: dict, risk: float) -> Literal["allow", "ask"]:
    """Return 'allow' or 'ask'."""
    if risk >= THRESH:
        plyer.notification.notify(
            title="Twin needs approval",
            message=json.dumps(msg)[:120] + "...",
            timeout=5,
        )
        return "ask"
    return "allow"
