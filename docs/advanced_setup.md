## Setup

This project requires **Python 3.10 or higher**. Ensure your virtual
environment uses a compatible interpreter.

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-village.git
   cd ai-village
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   # Optional: compile `llama.cpp` for mobile builds.

The tokenizer requires the `cl100k_base.tiktoken` vocabulary file. If it is
missing, download it from the [tiktoken repository](https://github.com/openai/tiktoken)
and place it in `rag_system/utils/token_data/`.

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`
   - Add an API key for the server: `API_KEY=my_secret_key`

5. Review the default Retrieval-Augmented Generation configuration:
   - The file `configs/rag_config.yaml` contains the default settings used by
     the RAG pipeline. Edit this file if you need to customize the behaviour.
   - Parameters such as `vector_store_type` and `graph_store_type` are
     currently informational only&mdash;`VectorStore` always uses FAISS and
     `GraphStore` always uses NetworkX.
6. Tune decision-making hyperparameters:
   - `configs/decision_making.yaml` defines settings for the MCTS and DPO modules.

## Installing Heavy Dependencies

Some features rely on large libraries such as `numpy`, `torch` and `faiss`. These packages may need to be installed separately, especially when GPU support is desired. Example commands:
```bash
pip install numpy torch faiss-cpu  # or faiss-gpu for CUDA systems
```

The geometry-aware training helpers attempt to use the optional
`torch-twonn` package for intrinsic dimension estimation but will fall
back to the lightweight implementation in `agent_forge/geometry/id_twonn.py`
if it is not installed.

Ensure the file `rag_system/utils/token_data/cl100k_base.tiktoken` exists before running the code. If absent, download it from the [tiktoken repository](https://github.com/openai/tiktoken). After installing the dependencies and placing the file, you can run the test suite with `pytest`:
```bash
pytest
```

## Embedding Model Requirement

The vector store expects an embedding model instance capable of providing
sentence embeddings. By default the code uses `BERTEmbeddingModel` from
`rag_system.utils.embedding`, which loads a small BERT model from
HuggingFace. If you prefer a different embedding source you can inject your
own model when constructing `VectorStore`.

## Testing

Before running the tests ensure that **all** dependencies are installed. The
main `requirements.txt` file includes heavy packages such as `torch`,
`faiss-cpu` and `numpy` that are required by the tests. Run the setup
script to install everything you need:

```bash
bash scripts/setup_env.sh
```

The script installs both runtime and development requirements,
including `PyYAML` which is required by the test configuration.

After installing these dependencies you can execute the full test suite:

```bash
pytest
```

Some unit tests automatically skip themselves if optional packages such
as PyTorch are missing. Installing the full requirements is recommended
for complete coverage.


\n## Docker\nA `Dockerfile` is provided to run tests in a container:\n```bash\ndocker build -t aivillage .\ndocker run aivillage\n```
CI uses [docker/setup-docker](https://github.com/docker/setup-docker) to provision the Docker CLI on restricted runners.

## Development Environment

Run `make dev-up` to build the Docker image and start the server on http://localhost:8000.

## Sprint 3 Monitoring & Soak Testing

To launch the services with Prometheus and Grafana enabled:

```bash
./run-monitoring.sh
```

This starts Prometheus (http://localhost:9090) and Grafana (http://localhost:3000). The default Grafana login is `admin/changeme`.
Grafana loads `ai-village-core.json` with p99 latency and error-rate panels for Gateway and Twin.

For a full 8‑hour load test run:

```bash
./run-soak-test.sh
```

This uses Locust to simulate chat traffic and pushes metrics to the Pushgateway.

See [docs/adr/ADR-S3-01-observability.md](docs/adr/ADR-S3-01-observability.md) for details.
Demo: [video](https://example.com/demo.mp4)

## 🎉 Sprint 4β (v0.5.0)

| Feature | TL;DR |
|---------|-------|
| **Confidence Calibration** | Optional `calibrated_prob` returned from `/v1/chat` (enable with `CALIBRATION_ENABLED=1`). |
| **Qdrant Vector Store** | Production search now served by Qdrant (<150 ms p95); FAISS fallback baked in. |
| **Path Explainer API** | `POST /explain` returns up-to-3-hop reasoning path for transparency. |

Upgrade steps:  
```bash
docker compose pull && \
export RAG_USE_QDRANT=1 CALIBRATION_ENABLED=1 && \
docker compose up -d --build
```
