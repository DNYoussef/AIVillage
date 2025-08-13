import json

from cryptography import x509
from jwcrypto import jwe, jwk, jws
import requests


def _load_priv(pem_path: str) -> jwk.JWK:
    return jwk.JWK.from_pem(open(pem_path, "rb").read())


def _load_pub(cert_path: str) -> jwk.JWK:
    cert = x509.load_pem_x509_certificate(open(cert_path, "rb").read())
    return jwk.JWK.from_pyca(cert.public_key())


def sign(payload: dict, key: jwk.JWK) -> str:
    token = jws.JWS(json.dumps(payload).encode())
    token.add_signature(key, None, json.dumps({"alg": "RS256"}))
    return token.serialize()


def encrypt(signed: str, key: jwk.JWK) -> str:
    enc = jwe.JWE(signed.encode(), json.dumps({"alg": "RSA-OAEP", "enc": "A256GCM"}))
    enc.add_recipient(key)
    return enc.serialize()


def decrypt(raw: str, priv: jwk.JWK) -> str:
    token = jwe.JWE()
    token.deserialize(raw, key=priv)
    return token.payload.decode()


def verify(signed: str, pub: jwk.JWK) -> dict:
    token = jws.JWS()
    token.deserialize(signed)
    token.verify(pub)
    return json.loads(token.payload)


def send_a2a(
    url: str,
    message: dict,
    sender_priv: str,
    receiver_pub: str,
    encrypt_msg: bool = True,
) -> requests.Response:
    priv = _load_priv(sender_priv)
    pub = _load_pub(receiver_pub)
    signed = sign(message, priv)
    body = encrypt(signed, pub) if encrypt_msg else signed
    hdr = {"Content-Type": "application/jwe" if encrypt_msg else "application/jws"}
    return requests.post(url, data=body, headers=hdr, timeout=30)


def receive_a2a(raw: str, receiver_priv: str, sender_pub: str, encrypted: bool = True) -> dict:
    priv = _load_priv(receiver_priv)
    data = decrypt(raw, priv) if encrypted else raw
    return verify(data, _load_pub(sender_pub))
