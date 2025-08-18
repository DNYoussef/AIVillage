from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from src.digital_twin import guard, runner
from src.digital_twin.security.preference_vault import PreferenceVault


def _write_prefs(tmp_path: Path, data: dict) -> None:
    key = Fernet.generate_key()
    os.environ["TWIN_PREF_KEY"] = key.decode()
    home = tmp_path / "home"
    (home / ".aivillage").mkdir(parents=True, exist_ok=True)
    enc = Fernet(key).encrypt(json.dumps(data).encode())
    (home / ".aivillage" / "prefs.json.enc").write_bytes(enc)
    os.environ["HOME"] = str(home)


def test_risk_gate_block_vs_allow(tmp_path: Path) -> None:
    _write_prefs(tmp_path, {"allow_shell": False})
    assert guard.risk_gate({"content": "rm -rf /"}) == "deny"

    _write_prefs(tmp_path, {"allow_shell": True})
    assert guard.risk_gate({"content": "ls"}) == "allow"
    assert guard.risk_gate({"content": "api_key=123"}) == "deny"


def test_vault_load_failure(tmp_path: Path) -> None:
    # Preferences file present but key missing
    home = tmp_path / "home"
    (home / ".aivillage").mkdir(parents=True)
    os.environ.pop("TWIN_PREF_KEY", None)
    os.environ["HOME"] = str(home)
    (home / ".aivillage" / "prefs.json.enc").write_bytes(b"garbage")

    vault = PreferenceVault()
    with pytest.raises(RuntimeError):
        vault.load()


def test_chat_echo(tmp_path: Path) -> None:
    _write_prefs(tmp_path, {"allow_shell": True})
    tokens = list(runner.chat("hello world"))
    joined = "".join(tokens)
    assert "hello" in joined and "world" in joined
