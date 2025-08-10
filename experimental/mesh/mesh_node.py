# SPDX-License-Identifier: Apache-2.0
"""MeshNode: P2P pub/sub (libp2p+Noise) + gRPC/TLS fallback for MCP tool calls."""

import json
import logging

import grpc
from libp2p import new_node
from libp2p.pubsub import FloodSub
from libp2p.security.noise.transport import NoiseSecureTransport
from nacl.public import Box, PrivateKey, PublicKey
import nacl.utils

from communications.message import Message

log = logging.getLogger("communications.mesh_node")


class SecureEnvelope:
    """Encrypt/decrypt payloads via NaCl Box (Noise keypair)."""

    def __init__(self, priv: PrivateKey, peer_pub: PublicKey) -> None:
        self.box = Box(priv, peer_pub)

    def encode(self, plaintext: bytes) -> bytes:
        nonce = nacl.utils.random(Box.NONCE_SIZE)
        return self.box.encrypt(plaintext, nonce)

    def decode(self, ciphertext: bytes) -> bytes:
        return self.box.decrypt(ciphertext)


class MeshNode:
    """- libp2p FloodSub over Noise
    - automatic peer-key registry via handshake messages
    - on_message hook for subclasses
    - fallback mcp_call via gRPC+TLS.
    """

    HANDSHAKE = "__handshake__"

    def __init__(
        self,
        listen_addr: str,
        bootstrap: list[str] | None = None,
        on_message: callable | None = None,
    ) -> None:
        self.listen_addr = listen_addr
        self.bootstrap = bootstrap or []
        self.on_message = on_message

        # generate our Noise keypair
        self.priv = PrivateKey.generate()
        self.pub = self.priv.public_key
        self.peer_keys: dict[str, PublicKey] = {}

        self.node = None
        self.pubsub = None

    async def start(self) -> None:
        # 1) start libp2p host with Noise
        self.node = await new_node(
            transport_opt=[NoiseSecureTransport(self.priv)], muxer_opt=[], sec_opt=[]
        )
        await self.node.get_network().listen(self.listen_addr)

        # 2) dial bootstrap peers
        for addr in self.bootstrap:
            try:
                await self.node.get_network().dial(addr)
            except Exception:
                log.warning("Bootstrap dial failed: %s", addr)

        # 3) setup PubSub
        self.pubsub = FloodSub(self.node)
        # subscribe to our ID topic
        await self.pubsub.subscribe(self.node.get_id().to_base58(), self._on_pubsub)

        # 4) broadcast handshake so others learn our pubkey
        await self._broadcast_handshake()

        log.info(
            "MeshNode listening on %s, id=%s",
            self.listen_addr,
            self.node.get_id().to_base58(),
        )

        await self.pubsub.publish(
            "meta/announce",
            json.dumps(
                {
                    "id": self.node.get_id().to_base58(),
                    "node_type": "twin",
                    "speed_toks": 90,
                }
            ).encode(),
        )

    async def _broadcast_handshake(self) -> None:
        payload = {
            "type": self.HANDSHAKE,
            "peer_id": self.node.get_id().to_base58(),
            "pubkey": self.pub.encode().hex(),
        }
        await self.pubsub.publish("global-handshake", json.dumps(payload).encode())

    async def _on_pubsub(self, peer_id: str, msg) -> None:
        try:
            data = msg.data
            # if handshake topic
            if msg.topic_ids and msg.topic_ids[0] == "global-handshake":
                payload = json.loads(data.decode())
                if payload["type"] == self.HANDSHAKE:
                    pid = payload["peer_id"]
                    pk = bytes.fromhex(payload["pubkey"])
                    self.peer_keys[pid] = PublicKey(pk)
                return

            # direct encrypted message
            pkt = SecureEnvelope(self.priv, self.peer_keys[peer_id]).decode(data)
            message = Message.from_json(pkt.decode())
            if self.on_message:
                await self.on_message(message)
        except Exception as e:
            log.debug("mesh_node drop malformed: %s", e)

    async def send(self, target_id: str, message: Message) -> None:
        """Encrypt & publish to target’s topic. target_id must be in peer_keys."""
        if target_id not in self.peer_keys:
            msg = f"unknown peer {target_id}"
            raise KeyError(msg)
        enc = SecureEnvelope(self.priv, self.peer_keys[target_id]).encode(
            message.to_json().encode()
        )
        await self.pubsub.publish(target_id, enc)
        import os

        from communications.credit_manager import CreditManager

        CreditManager(os.getenv("TWIN_MNEMONIC", "")).mint(
            task_id=message.id,
            macs=len(message.content.get("tensor", [])) * 1_000_000,
        )

    # --------------------------------------------------------------------------
    # Fallback MCP tool call via gRPC+TLS
    # --------------------------------------------------------------------------
    async def mcp_call(
        self, service_addr: str, stub_class, request, timeout: float = 5.0
    ):
        """service_addr: "host:port"; stub_class: generated gRPC Stub; request: Protobuf request."""
        creds = grpc.ssl_channel_credentials()
        async with grpc.aio.secure_channel(service_addr, creds) as ch:
            stub = stub_class(ch)
            return await stub.Call(request, timeout=timeout)
