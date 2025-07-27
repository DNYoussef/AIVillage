# Sage Agent

The Sage Agent performs research and analysis tasks within the AI Village system. It builds on the `UnifiedBaseAgent` and integrates a collection of advanced modules for query processing, learning and collaboration.

## Main Modules

- **SageAgent** – central class orchestrating the agent's behaviour and RAG interactions.
- **FoundationalLayer** – initializes core resources and embeddings.
- **ContinuousLearningLayer** – tracks performance and adapts behaviour over time.
- **QueryProcessor** – extracts entities and relations from user queries and prepares them for the RAG pipeline.
- **TaskExecutor** – runs structured tasks and delegates work to sub‑components.
- **CollaborationManager** – handles knowledge sharing and task delegation between agents.
- **ResearchCapabilities** – manages optional capabilities such as web search and data analysis.
- **UserIntentInterpreter** – lightweight intent detection used when handling free‑form queries.
- **ResponseGenerator** – formats final answers based on RAG results and detected intent.

## Usage

```python
from agents.sage import SageAgent, SageAgentConfig
from communications.protocol import StandardCommunicationProtocol
from rag_system.retrieval.vector_store import VectorStore

config = SageAgentConfig(name="sage")
comm = StandardCommunicationProtocol()
vector_store = VectorStore()

sage = SageAgent(config, comm, vector_store)
result = await sage.execute_task(Task(sage, "Find recent AI papers", "", 1))
```

The agent may also be run via the message protocol by sending `Message` objects to its `handle_message` method.
