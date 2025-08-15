"""Placeholder for libp2p mesh module used in tests."""
from dataclasses import dataclass
from enum import Enum


@dataclass
class MeshConfiguration:
    node_id: str = "test"
    listen_port: int = 0
    max_peers: int = 0
    transports: list[str] | None = None


class MeshMessageType(str, Enum):
    DATA = "data"


@dataclass
class MeshMessage:
    sender: str
    recipient: str
    payload: str
    type: MeshMessageType = MeshMessageType.DATA


class LibP2PMeshNetwork:
    def __init__(self, config: MeshConfiguration | None = None) -> None:
        self.config = config or MeshConfiguration()
        self.status = type("S", (), {"value": "inactive"})
        self.node_id = self.config.node_id
