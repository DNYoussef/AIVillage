# AIVillage Entry Points Guide

This document describes how to launch the Retrieval-Augmented Generation (RAG)
components of the project. Previous versions of this file referred to a
non‑existent `main.py` wrapper; the actual executables are documented below.

## RAG API server

The production RAG service lives in
`src/production/rag/rag_api_server.py`. It exposes a FastAPI application with
query, indexing and health endpoints.

Run the server directly with Python:

```bash
python -m src.production.rag.rag_api_server
```

or via ``uvicorn``:

```bash
uvicorn src.production.rag.rag_api_server:app --host 0.0.0.0 --port 8082
```

The port defaults to ``8082`` but can be overridden with the ``RAG_API_PORT``
environment variable.

## Command‑line interface

For simple CLI interaction there is a thin wrapper script at
`rag_system/main.py`. It delegates to the production pipeline implementation in
`src.production.rag.rag_system.main`.

Example usage:

```bash
python rag_system/main.py query --question "What is AI?"
```

Available actions mirror those of the production pipeline (`query`, `index`,
`search`, `status`, `config`). The CLI is intended for experimentation and
shares the same code as the production system.
