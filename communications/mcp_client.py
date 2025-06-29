import os
import uuid
import requests
from jose import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")

class MCPClient:
    """JSON-RPC 2.0 over HTTPS with mTLS & JWT."""

    def __init__(self, endpoint: str, cert: str, key: str, ca: str):
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.verify = ca
        self.session.cert = (cert, key)

    def _make_token(self) -> str:
        return jwt.encode({"aud": "mcp"}, JWT_SECRET, algorithm="HS256")

    def call(self, method: str, params: dict) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": str(uuid.uuid4()),
        }
        headers = {"Authorization": f"Bearer {self._make_token()}"}
        resp = self.session.post(self.endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()
