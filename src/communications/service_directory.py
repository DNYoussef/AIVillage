import json
import os
from pathlib import Path
from threading import RLock
from typing import Dict

_CACHE_PATH = Path('.cache/agents.json')

class ServiceDirectory:
    """Lightweight in-process service directory with persistence."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._services: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if _CACHE_PATH.exists():
            try:
                data = json.loads(_CACHE_PATH.read_text())
                if isinstance(data, dict):
                    self._services.update({str(k): str(v) for k, v in data.items()})
            except Exception:
                # Corrupt cache; start fresh
                self._services = {}

    def _save(self) -> None:
        _CACHE_PATH.parent.mkdir(exist_ok=True)
        tmp_path = _CACHE_PATH.with_suffix('.tmp')
        tmp_path.write_text(json.dumps(self._services))
        tmp_path.replace(_CACHE_PATH)

    def register(self, agent_id: str, url: str) -> None:
        """Register agent_id with URL."""
        with self._lock:
            self._services[agent_id] = url
            self._save()

    def lookup(self, agent_id: str) -> str | None:
        """Return registered URL or default from environment."""
        with self._lock:
            url = self._services.get(agent_id)
        if url:
            return url
        host = os.getenv('COMM_DEFAULT_HOST', 'localhost')
        port = os.getenv('COMM_DEFAULT_PORT')
        if port:
            return f"ws://{host}:{port}/ws"
        return None

# singleton instance
service_directory = ServiceDirectory()

__all__ = ['ServiceDirectory', 'service_directory']
