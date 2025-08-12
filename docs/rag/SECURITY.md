# RAG Security

## Qdrant TLS Configuration

- Set `QDRANT_URL` to an `https://` endpoint in production.
- Provide TLS certificate and key paths via the `TLS_CERT_FILE` and `TLS_KEY_FILE` environment variables when hosting your own Qdrant instance.
- Supply a CA bundle with `TLS_CA_FILE` or `REQUESTS_CA_BUNDLE` if using self-signed certificates.
- The system will fail to start in production if `QDRANT_URL` uses plain `http://`.
