# AI Village Self-Improving System
[![API Docs](https://img.shields.io/badge/docs-latest-blue)](https://atlantisai.github.io/atlantis) [![Coverage](docs/assets/coverage.svg)](#)

This project implements a multi-agent AI system featuring a self-evolving
architecture for continuous improvement and adaptation.

> **Status**
> The following pieces of the project are usable today:
> - Retrieval-Augmented Generation (RAG) pipeline
> - FastAPI server with a simple web UI
> - Experimental model merging utilities
> 
> All other phases &mdash; including Quiet-STaR thought generation, expert
> vectors and ADAS optimization &mdash; remain future work and are outlined for
> reference only in the documentation.  The `SelfEvolvingSystem` class present
> in the code is a lightweight stub used for demos and tests.
> The SAGE Framework remains a placeholder and is not yet implemented.

Refer to [docs/feature_matrix.md](docs/feature_matrix.md) for a status overview of all major components.

<!--feature-matrix-start-->
| Sub-system | Status |
|------------|--------|
| Twin Runtime | ✅ |
| King / Sage / Magi | ✅ |
| Self‑Evolving System | 🔴 |
| HippoRAG | 🔴 |
| Mesh Credits | ✅ |
| ADAS Optimisation | ✅ |
| ConfidenceEstimator | ✅ |
<!--feature-matrix-end-->
The [messaging protocol decision](docs/adr/0002-messaging-protocol.md) is documented in **ADR-0002**.  gRPC/WebSocket support described there is **not yet implemented**; the Gateway and Twin currently communicate over plain HTTP.

## System Components

1. UnifiedBaseAgent: A comprehensive base agent class that combines features from previous agent implementations.
2. Specialized Agents:
   - KingAgent: Coordinates tasks and manages other agents.
   - SageAgent: Handles research and analysis tasks.
   - MagiAgent: Focuses on development and coding tasks.
3. Self-Evolving System: Implements multi-layer improvement mechanisms:
   - Quality Assurance: Ensures task safety and stability using Uncertainty-enhanced Preference Optimization (UPO).
   - Prompt Baking: Efficiently incorporates new knowledge.
   - Continuous Learning: Rapidly integrates new experiences.
   - SAGE Framework: Enables recursive self-improvement through assistant-checker-reviser cycle *(planned feature, not yet implemented).* 
   - Decision Making: Uses dedicated Monte Carlo Tree Search (MCTS) and Direct Preference Optimization (DPO) modules.
4. IncentiveModel: A sophisticated model for calculating and managing incentives for agents based on their performance and task complexity.

## Self-Evolving System

The self-evolving system is the core of the AI Village's continuous improvement capabilities. It consists of several interconnected layers:

1. Quality Assurance Layer: Uses uncertainty estimation to ensure task safety.
2. Foundational Layer (Prompt Baking): Encodes and integrates new knowledge efficiently into the system's knowledge base.
3. Continuous Learning Layer: Extracts valuable information from tasks and results to update the system's capabilities.
4. Agent Architecture Layer (SAGE Framework): Implements a self-aware generative engine for response generation, evaluation, and revision. *(This layer is currently a placeholder.)*
5. Decision-Making Layer: Utilizes advanced AI techniques for making informed decisions based on tasks and context.

The system periodically evolves by updating agent capabilities, refining decision-making processes, and optimizing its overall architecture.

See [docs/roadmap.md](docs/roadmap.md) for a short overview of completed and
planned milestones.

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

The test suite uses lightweight stubs so that `pytest` can discover the tests
even when heavy optional dependencies like PyTorch are missing.  To actually
execute the ML focused tests install the full requirements first.  The helper
script below installs both runtime and development dependencies including
`torch`, `faiss-cpu` and `scikit-learn`:

```bash
bash scripts/setup_env.sh
```

The script installs both runtime and development requirements,
including `PyYAML` which is required by the test configuration.
If you prefer to manage the heavy ML libraries yourself, simply install the
test extras with:

```bash
pip install -r tests/requirements.txt
```

After installing these dependencies you can execute the full test suite:

```bash
pytest
```

Some unit tests automatically skip themselves if optional packages such
as PyTorch are missing. When these packages are available they will be used
instead of the lightweight stubs, providing complete coverage.



## Running the System

Start the API server:

```
python server.py
```

This will start a FastAPI application that initializes the RAG pipeline and exposes the HTTP endpoints.

### Web UI

After starting the server, open `http://localhost:8000/` in a browser to access a simple dashboard. The UI allows you to:

1. Submit queries to the `/query` endpoint.
2. Upload text files that update the vector store.
3. View the current BayesRAG graph snapshot and system status.
4. Inspect recent retrieval logs.

### Stand-alone RAG Example

You can also exercise the pipeline without the web server:

```bash
python rag_system/main.py
```

This script issues a sample query and performs a short knowledge graph
exploration using the built-in reasoning engine.

## Extending the System

To add new capabilities or agents:

1. Create a new agent class inheriting from `UnifiedBaseAgent`
2. Implement the `execute_task` method for the new agent
3. Add the new agent to the `SelfEvolvingSystem` in `orchestration.py`

## Communication Message Types

Agents exchange `Message` objects categorized by `MessageType`. The core types are
`TASK`, `QUERY`, `RESPONSE`, and `NOTIFICATION`. Collaborative features also use
`COLLABORATION_REQUEST`, `KNOWLEDGE_SHARE`, `TASK_RESULT`, and
`JOINT_REASONING_RESULT`. Additional system messages include `UPDATE`, `COMMAND`,
`BULK_UPDATE`, `PROJECT_UPDATE`, `SYSTEM_STATUS_UPDATE`, `CONFIG_UPDATE`, and
`TOOL_CALL` as defined in `communications/message.py`.


## Model Compression

This repository includes a two-stage compression framework in `agent_forge/compression`. It converts linear weights to 1.58-bit BitNet form, encodes them with SeedLM, applies VPTQ quantization, and optionally hyperfunction encoding. Use `stream_compress_model` to compress a PyTorch model. See `docs/ultimate_llm_compression_framework.md` for details.

## Recent Updates

The agent system has recently undergone significant updates to improve modularity, reduce redundancy, and incorporate a self-evolving system. Key changes include:

1. Introduction of the `UnifiedBaseAgent` class, which serves as the foundation for all specialized agents.
2. Implementation of the `SelfEvolvingSystem` class, which manages the continuous improvement of agents.
3. Initial plumbing for the self-evolving system. Full integration across all agents is still planned.
4. Removal of the `langroid` folder, with its functionality now integrated into the main agent structure.
5. Updates to the `orchestration.py` file to use the new agent classes and self-evolving system.
6. Addition of a mesh-sharding subsystem in `communications/` providing peer-to-peer networking, federated learning, credit management, and sharding utilities.
7. `VectorStore` persistence now uses JSON instead of pickle. Convert existing `.pkl` stores or regenerate them.

For details see [docs/migration_notes.md](docs/migration_notes.md).

These changes have made the code more modular and easier to maintain. The self-evolving system remains a stub used for demos and tests, so additional work is required before it can manage real agent evolution.

## Populating the RAG System with Academic Papers

To provide your AI Village with a starting base of information, you can manually feed academic papers into the RAG system. Follow these steps to add several dozen papers:

1. Prepare your academic papers:
   - Ensure your papers are in a readable format (PDF, TXT, or DOCX).
   - Organize the papers in a single directory for easy access.

2. Start the AI Village server if it's not already running:
   ```
   python server.py
   ```

3. Use the `/upload` endpoint to add each paper to the knowledge base:
   - For each paper, send a POST request to `http://localhost:8000/upload`
   - Use a tool like curl, Postman, or a custom script to automate this process

   Example using curl:
   ```
   curl -X POST -H "Content-Type: multipart/form-data" -F "file=@path/to/paper.pdf" http://localhost:8000/upload
   ```

   Example using Python with requests library:
   ```python
   import requests
   import os

   papers_directory = "/path/to/papers_directory"
   
   for filename in os.listdir(papers_directory):
       file_path = os.path.join(papers_directory, filename)
       with open(file_path, 'rb') as file:
           files = {'file': (filename, file)}
           response = requests.post('http://localhost:8000/upload', files=files)
       print(f"Uploaded {filename}: {response.status_code}")
   ```

4. Monitor the upload process:
   - Check the server logs for any errors or warnings during the upload process.
   - Ensure that each paper is successfully processed and added to the knowledge base.

5. Verify the integration:
   - After uploading the papers, you can test the system by sending queries related to the content of the uploaded papers.
   - Use the `/query` endpoint to ask questions and verify that the system can retrieve and use the information from the uploaded papers.

   Example query:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"query": "Summarize the key findings from the papers on AI ethics"}' http://localhost:8000/query
   ```

6. Fine-tune and optimize:
   - Based on the query results, you may need to adjust the RAG system parameters or preprocessing steps to improve information retrieval and synthesis from the uploaded papers.
   - Consider updating the `configs/rag_config.yaml` file to optimize the RAG system for academic paper processing.

By following these steps, you can manually feed several dozen academic papers into your RAG system, providing a rich starting base of information for your AI Village. This will enhance the system's ability to answer queries and perform tasks related to the content of these papers.

### Prerequisites

Make sure you have **Python 3.10+** installed before continuing with the installation steps below.

## Installation

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

4. Set up environment variables:
   - Create a `.env` file in the root directory and add the following variables:
     ```
     OPENAI_API_KEY=your_openai_api_key
     NEO4J_URI=your_neo4j_uri
     NEO4J_USER=your_neo4j_username
     NEO4J_PASSWORD=your_neo4j_password
     ```

## Usage

1. Start the AI Village server:
   ```
   python server.py
   ```

2. The server will start running on `http://localhost:8000`. You can now use the following endpoints:

  - POST `/query`: Send a query to the AI Village
  - POST `/upload`: Upload a file to populate the knowledge base

3. Use a tool like curl or Postman to interact with the API, or integrate it into your application.

## Documentation

For a quick repository structure overview see `docs/system_overview.md`. Detailed pipeline documentation is provided in `docs/complete_agent_forge_pipeline.md` and other files in the `docs/` directory.

## Model Mergers

AI Village includes functionality for merging language models using the EvoMerge system. This allows you to combine different models to create a new, potentially more powerful model.

### Available Mergers

1. **Parameter Space (PS) Techniques**: 
   - Linear Merge
   - SLERP (Spherical Linear Interpolation)
   - TIES (Task-Informed Parameter Ensembling)
   - DARE (Density-Aware Representation Ensembling)

2. **Deep Fusion Space (DFS) Techniques**:
   - Frankenmerge
   - DFS (Deep Fusion Space)

3. **Weight Masking**: Apply weight masking to the merged model for potential performance improvements.

### Usage

To use the model mergers, follow these steps:

1. Navigate to the `agent_forge/evomerge` directory:
   ```
   cd agent_forge/evomerge
   ```

2. Run the merger script with the desired configuration:
   ```
   python cli.py --download-and-merge --model1 <model1_path> --model2 <model2_path> [--model3 <model3_path>] [options]
   ```

3. The merged model will be saved in the directory specified by the `--custom-dir` option. By default, this is the `merged_models/` folder in your working directory. You can modify this location by editing `merge_config.yaml` or passing `--custom-dir` on the command line.

### Configuration Options

- `--model1`, `--model2`, `--model3`: Paths or Hugging Face model IDs for the models to merge (at least two required)
- `--ps-technique1`: First parameter space merging technique (default: linear)
- `--ps-technique2`: Second parameter space merging technique (default: ties)
- `--dfs-technique`: Deep fusion space merging technique (default: frankenmerge)
- `--weight-mask-rate`: Weight mask rate, between 0.0 and 1.0 (default: 0.0)
- `--use-weight-rescale`: Use weight rescaling (flag)
- `--mask-strategy`: Mask strategy, either "random" or "magnitude" (default: random)
- `--use-disk-based-merge`: Perform merging on disk instead of RAM (flag)
- `--chunk-size`: Chunk size for disk-based operations (default: 1000000)
- `--use-cli`: Use Hugging Face CLI to download models (flag)
- `--verbose`: Enable verbose output (flag)

### Example Command

```bash
python cli.py --download-and-merge --model1 gpt2 --model2 distilgpt2 --ps-technique1 slerp --ps-technique2 dare --dfs-technique frankenmerge --weight-mask-rate 0.1 --use-weight-rescale --mask-strategy magnitude --verbose
```

This command will merge the GPT-2 and DistilGPT-2 models using SLERP and DARE for parameter space merging, Frankenmerge for deep fusion space merging, and apply weight masking with a rate of 0.1 using the magnitude-based strategy.

For more detailed information on model merging and advanced configurations, please refer to the `docs/model_merging.md` file.

## King Agent

The King Agent is a sophisticated AI system designed to coordinate and manage multiple AI agents in the AI Village project. It uses advanced decision-making processes, task routing, and management techniques to efficiently handle complex tasks and workflows.

### Key Features

1. **Monte Carlo Tree Search (MCTS) for Workflow Optimization**: The King Agent currently includes a simplified stub for MCTS to generate and optimize workflows. Advanced algorithms are planned in future releases.

2. **Incentive-Based Agent Management**: An incentive model is implemented to motivate and manage multiple agents effectively.

3. **Intelligent Task Routing**: The King Agent uses a preference-based approach to route tasks to the most appropriate agents.

4. **Hierarchical Sub-Goal Generation**: The system can break down complex tasks into manageable sub-goals, improving overall task completion efficiency.

5. **Continuous Learning and Adaptation**: The King Agent implements a feedback loop to continuously improve its decision-making and task allocation processes.

6. **RAG System Integration**: The King Agent is now integrated with an Enhanced RAG (Retrieval-Augmented Generation) system for improved information retrieval and decision-making.

7. **Robust Error Handling and Logging**: Comprehensive error handling and logging have been implemented throughout the King Agent system for better debugging and monitoring.

8. **Enhanced Incentive-Based Management**: The King Agent now utilizes an improved IncentiveModel for more effective agent motivation and task allocation.

### Components

- **KingAgent**: The main class that integrates all other components and serves as the primary interface for the King Agent system.
- **KingCoordinator**: The central component that manages interactions between different parts of the system.
- **UnifiedTaskManager**: Responsible for creating, assigning, and managing tasks across different agents.
- **DecisionMaker**: Relies on MCTS and DPO utilities from `agents.utils` and integrates RAG-enhanced analysis.
- **ProblemAnalyzer**: Analyzes tasks and generates comprehensive problem analyses by collaborating with other agents.
- **AgentRouter**: Efficiently routes tasks to the most appropriate agents based on their capabilities and past performance.

### Usage

To use the King Agent in your project:

1. Initialize the KingAgent with the necessary dependencies (communication protocol, RAG system).
2. Set up your agents and register them with the KingCoordinator.
3. Send task messages to the KingAgent for processing.

Example:

```python
from agents.king.king_agent import KingAgent, KingAgentConfig
from your_communication_protocol import CommunicationProtocol
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.core.config import load_from_yaml
from rag_system.retrieval.vector_store import VectorStore
from rag_system.retrieval.graph_store import GraphStore
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker

# Initialize dependencies
comm_protocol = CommunicationProtocol()
rag_config = load_from_yaml("configs/rag_config.yaml")
vector_store = VectorStore(rag_config)
tracker = UnifiedKnowledgeTracker(vector_store, GraphStore())
rag_system = EnhancedRAGPipeline(rag_config, tracker)

# Create KingAgent
config = KingAgentConfig(name="KingAgent", description="Main coordinator for AI Village", model="gpt-4")
king_agent = KingAgent(config, comm_protocol, vector_store, tracker)

# Register agents
await king_agent.coordinator.add_agent("sage", SageAgent(comm_protocol))
await king_agent.coordinator.add_agent("magi", MagiAgent(comm_protocol))

# Send a task message
task_message = Message(content={"description": "Analyze this dataset"})
result = await king_agent.execute_task(task_message)
```

For more detailed information about the King Agent, its components, and usage, please refer to the `agents/king/README.md` file and the `agents/king/demo.py` script for a working example.

- **IncentiveModel**: A sophisticated model for calculating and adjusting incentives based on task complexity, agent performance, and other factors.

[... Keep the rest of the existing content in this section ...]

## Recent Updates

The agent system has recently undergone significant updates to improve modularity, reduce redundancy, and incorporate a self-evolving system. Key changes include:

[... Keep the existing updates ...]

6. Enhancement of the IncentiveModel with improved functionality for calculating incentives, updating based on task results, and adapting to changing agent performance.
7. Improved integration of incentive-based management in the UnifiedTaskManager for more effective task allocation and agent motivation.
8. `VectorStore` persistence now uses JSON instead of pickle. Convert existing `.pkl` stores or regenerate them.

For details see [docs/migration_notes.md](docs/migration_notes.md).

These changes have further improved the system's ability to adapt and optimize its performance over time.

## Future Improvements

1. Implement a concrete VectorStore for efficient knowledge storage and retrieval.
2. Develop more sophisticated task generation and result processing mechanisms.
3. Implement proper error handling and logging throughout the system.
4. Enhance the evolution mechanisms to include more advanced techniques like neural architecture search.
5. Further refine the IncentiveModel to incorporate more complex factors and long-term performance trends.
6. Develop advanced analytics and visualization tools for monitoring agent performance and system efficiency.
# RAG System

This repository contains an implementation of a Retrieval-Augmented Generation (RAG) system with advanced features and a modular architecture.

## System Architecture

The RAG system consists of the following main components:

1. **UnifiedConfig**: A centralized configuration class that manages settings for all components of the system.
2. **SageAgent**: An advanced agent capable of processing queries, executing tasks, and managing the RAG pipeline.
3. **EnhancedRAGPipeline**: The core RAG system that handles retrieval and generation tasks and reports retrieved documents to the tracker.
4. **UnifiedKnowledgeTracker**: Tracks knowledge changes and logs all retrievals for later analysis.
5. **ErrorController**: A centralized error handling system for managing and recovering from errors.

## Key Features

- Modular and extensible architecture
- Unified configuration management
- Advanced error handling and recovery mechanisms
- Continuous learning and self-evolution capabilities
- Integration of multiple research capabilities (web search, data analysis, information synthesis, etc.)
- Streamlined query processing pipeline

## Getting Started

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   # Optional: compile `llama.cpp` for mobile builds.
3. Run the main script:
   ```
   python rag_system/main.py
   ```

## Running Tests

To run the unit tests and integration tests (with real ML libraries installed):

```bash
pytest -vv
```
Install the additional testing dependencies with:
```bash
pip install -r tests/requirements.txt
```

## Configuration

The system can be configured by modifying the `UnifiedConfig` class in `rag_system/core/config.py`. This class manages settings for all components of the RAG system.

## Agent Forge Pipeline

See [docs/complete_agent_forge_pipeline.md](docs/complete_agent_forge_pipeline.md) for a detailed description of the planned training pipeline. Advanced phases such as Quiet-STaR, expert vectors and ADAS are noted there as **future work**. A shorter summary remains available in [docs/agent_forge_pipeline_overview.md](docs/agent_forge_pipeline_overview.md).

Additional utilities for this pipeline include:
- `agent_forge/training/self_modeling.py` – temperature‑based self‑modeling.
- `agent_forge/training/prompt_baking.py` – wrapper for deep prompt baking cycles.
- `agent_forge/training/expert_vectors.py` – tools for expert vector creation and composition *(planned feature)*.
- `agent_forge/training/identity.py` – identity formation and moral framework utilities.
- `agent_forge/training/geometry_pipeline.py` – minimal geometry-aware training loop using SVF and Edge-of-Chaos PID.

- For geometry-aware modules integrating Two-NN ID estimation, Grokfast optimization and SleepNet/DreamNet cycles see [docs/geometry_aware_training.md](docs/geometry_aware_training.md).
## Contributing

Please refer to the CONTRIBUTING.md file for guidelines on how to contribute to this project.

## License

This project is licensed under the [MIT License](LICENSE).

\n## Docker\nA `Dockerfile` is provided to run tests in a container:\n```bash\ndocker build -t aivillage .\ndocker run aivillage\n```
CI uses [docker/setup-docker](https://github.com/docker/setup-docker) to provision the Docker CLI on restricted runners.

## Development Environment

Run `make dev-up` to build the Docker image and start the server on http://localhost:8000.

## Sprint 3 Monitoring & Soak Testing

To launch the services with Prometheus and Grafana enabled:

```bash
./scripts/manage.sh start
```

This starts Prometheus (`$PROMETHEUS_URL`, default `http://localhost:9090`) and Grafana (`$GRAFANA_URL`, default `http://localhost:3000`).
Grafana loads `ai-village-core.json` with p99 latency and error‑rate panels for Gateway and Twin.

For a full 8‑hour load test run:

```bash
./scripts/manage.sh soak-test
```

This uses Locust to simulate chat traffic and pushes metrics to the Pushgateway (`$PUSHGATEWAY_URL`).

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
