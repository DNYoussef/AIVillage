# System Overview

This document provides a quick reference for the structure of the repository and the main
technologies used.

> **Note**
> Features such as the SAGE framework, expert vector system and ADAS optimization are conceptual only. The current codebase implements a basic RAG pipeline without these advanced capabilities.

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

 - **Language**: Python 3.10+ (≈25k lines across 325 files).
- **Web Framework**: FastAPI powers the server (`server.py`).
- **ML Libraries**: PyTorch, Transformers, FAISS, Langroid.

The project follows a monolithic layout with modular packages rather than multiple
microservices. Agents share a common `UnifiedBaseAgent` class and interact with the
RAG pipeline through well defined interfaces.

Many advanced features referenced in the repository (such as the SAGE framework,
expert vectors and ADAS optimization) are presently conceptual. The running code
implements only the base RAG pipeline and prompt-baking utilities.

## User Intent Interpretation

The `UserIntentInterpreter` in `agents/sage` detects simple intent types using
keyword patterns. The following phrases are recognised:

- `search for`, `look up`, `find` → **search**
- `summarize`, `summary of` → **summarize**
- `explain` → **explanation**
- `analyze` → **analysis**
- `compare` → **comparison**
- `generate`, `write`, `create` → **generation**

Queries that do not match any pattern are marked as `unknown` with a lower
confidence score.

