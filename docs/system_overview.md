# System Overview

This document provides a quick reference for the structure of the repository and the main
technologies used.

## Directory Layout

- `agent_forge/` – Training utilities and experimental pipelines.
- `agents/` – Specialized agent implementations (King, Sage, Magi).
- `rag_system/` – Retrieval‑Augmented Generation pipeline and supporting modules.
- `docs/` – Project documentation including pipeline descriptions.
- `tests/` – Unit tests for core modules.
- `ui/` – Static files for the simple FastAPI dashboard.

Other folders contain configuration files, small utility modules and merged model
artifacts.

## Technologies

- **Language**: Python 3 (≈21k lines across 240+ files).
- **Web Framework**: FastAPI powers the server (`server.py`).
- **ML Libraries**: PyTorch, Transformers, FAISS, Langroid.

The project follows a monolithic layout with modular packages rather than multiple
microservices. Agents share a common `UnifiedBaseAgent` class and interact with the
RAG pipeline through well defined interfaces.
