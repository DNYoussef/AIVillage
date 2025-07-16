# PURPOSE: Central chat engine that wraps Twin inference and applies optional
#          conformal calibration to the model's raw confidence score.
#
# IMPLEMENTATION NOTES
# • Feature-flag via env var CALIBRATION_ENABLED=1
# • Falls back gracefully if calibrator fails >3× in a row
# • Twin client discovered via TWIN_URL (HTTP JSON API)
# • Exposes ChatEngine.process_chat(message, conversation_id)
#   ↳ returns dict {"response": str,
#                   "conversation_id": str,
#                   "raw_prob": float,
#                   "calibrated_prob": float|None,
#                   "timestamp": datetime}

from __future__ import annotations

from datetime import datetime
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TWIN_URL = os.getenv("TWIN_URL", "http://twin:8001/v1/chat")
CALIB_ENABLED = os.getenv("CALIBRATION_ENABLED", "0") == "1"

if CALIB_ENABLED:
    try:
        from calibration.conformal import ConformalCalibrator

        _calibrator = ConformalCalibrator.load_default()
        logger.info("Calibration enabled (loaded default model)")
    except Exception:  # pragma: no cover - fallback is tested separately
        logger.exception("Failed to init calibrator – continuing without")
        CALIB_ENABLED = False
        _calibrator = None
else:  # pragma: no cover - branch for disabled feature
    _calibrator = None


class ChatEngine:
    """Thin wrapper around the Twin micro-service plus optional calibration."""

    def __init__(self) -> None:
        self._calib_enabled = CALIB_ENABLED
        self._consecutive_calib_errors = 0

    # ------------------------------------------------------------------
    def process_chat(self, message: str, conversation_id: str) -> dict[str, Any]:
        started = time.time()
        twin_payload = {"prompt": message, "conversation_id": conversation_id}
        try:
            twin_resp = requests.post(TWIN_URL, json=twin_payload, timeout=15)
            twin_resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - network failure
            logger.exception("Twin request failed")
            raise exc

        data = twin_resp.json()
        resp_text: str = data.get("response", "")
        raw_prob: float = float(data.get("raw_prob", 0.5))
        calibrated: float | None = data.get("calibrated_prob")

        if calibrated is None and self._calib_enabled:
            try:
                calibrated = _calibrator.calibrate(raw_prob)  # type: ignore[attr-defined]
                self._consecutive_calib_errors = 0
            except Exception as exc:  # pragma: no cover - rarely triggered
                logger.warning("Calibration error: %s", exc)
                self._consecutive_calib_errors += 1
                if self._consecutive_calib_errors >= 3:
                    logger.error("Disabling calibration after 3 failures")
                    self._calib_enabled = False

        return {
            "response": resp_text,
            "conversation_id": conversation_id,
            "raw_prob": raw_prob,
            "calibrated_prob": calibrated,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": int((time.time() - started) * 1000),
        }
