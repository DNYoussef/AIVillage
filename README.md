# AI Village Self-Improving System

This project implements a multi-agent AI system using Langroid, featuring a self-evolving architecture for continuous improvement and adaptation.

## System Components

1. Agent: A comprehensive agent class that combines features from the previous Agent and BaseAgent classes, built on Langroid's ChatAgent.
2. Specialized Agents:
   - King Agent: Coordinates tasks and manages other agents.
   - Sage Agent: Handles research and analysis tasks.
   - Magi Agent: Focuses on development and coding tasks.
3. Self-Evolving System: Implements multi-layer improvement mechanisms:
   - Quality Assurance: Ensures task safety and stability using Uncertainty-enhanced Preference Optimization (UPO).
   - Prompt Baking: Efficiently incorporates new knowledge using Low-Rank Adaptation (LoRA) techniques.
   - Continuous Learning: Rapidly integrates new experiences using Self-Educated Learning for Function PARaMeterization (SELF-PARAM).
   - SAGE Framework: Enables recursive self-improvement through assistant-checker-reviser cycle.
   - Decision Making: Utilizes advanced algorithms (MCTS and DPO) for effective choices.

## Self-Evolving System

The self-evolving system is the core of the AI Village's continuous improvement capabilities. It consists of several interconnected layers:

1. Quality Assurance Layer: Uses Monte Carlo dropout for uncertainty estimation to ensure task safety.
2. Foundational Layer (Prompt Baking): Encodes and integrates new knowledge efficiently into the system's knowledge base.
3. Continuous Learning Layer: Extracts valuable information from tasks and results to update the system's capabilities.
4. Agent Architecture Layer (SAGE Framework): Implements a self-aware generative engine for response generation, evaluation, and revision.
5. Decision-Making Layer: Utilizes advanced AI techniques for making informed decisions based on tasks and context.

The system periodically evolves by updating agent capabilities, refining decision-making processes, and optimizing its overall architecture.


## Setup

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

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Running the System

Execute the main script:

```
python agents/orchestration.py
```

This will start the AI Village system, initializing all agents and the self-evolving components.

## Extending the System

To add new capabilities or agents:

1. Create a new agent class inheriting from `Agent`
2. Implement the `execute_task` method for the new agent
3. Add the new agent to the `SelfEvolvingSystem` in `orchestration.py`

## Recent Refactoring

The agent system has recently undergone a refactoring process to improve modularity and reduce redundancy. Key changes include:

1. Consolidation of the `Agent` and `BaseAgent` classes into a single, more comprehensive `Agent` class.
2. Updates to the `orchestration.py` file to use the new `Agent` class and improve its structure.
3. Refactoring of the `self_evolving_system.py` file to work with the new `Agent` class and improve overall structure and readability.

These changes have made the code more modular, easier to maintain, and more consistent across the system. The self-evolving system now works with a unified `Agent` class, which should make it easier to add new agent types or modify existing ones in the future.

## Future Improvements

1. Implement a concrete VectorStore for efficient knowledge storage and retrieval.
2. Develop more sophisticated task generation and result processing mechanisms.
3. Implement proper error handling and logging throughout the system.
4. Enhance the evolution mechanisms to include more advanced techniques like neural architecture search.
## Populating the RAG System with Academic Papers

To provide your AI Village with a starting base of information, you can manually feed academic papers into the RAG system. Follow these steps to add several dozen papers:

1. Prepare your academic papers:
   - Ensure your papers are in a readable format (PDF, TXT, or DOCX).
   - Organize the papers in a single directory for easy access.

2. Start the AI Village server if it's not already running:
   ```
   python main.py
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
   python main.py
   ```

2. The server will start running on `http://localhost:8000`. You can now use the following endpoints:

   - POST `/query`: Send a query to the AI Village
   - POST `/upload`: Upload a file to populate the knowledge base
   - POST `/import_open_researcher`: Import data from the Open Researcher project

3. Use a tool like curl or Postman to interact with the API, or integrate it into your application.

## Documentation

For more detailed information about the AI Village architecture, usage, and API reference, please refer to the documents in the `docs/` directory.

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

3. The merged model will be saved in the directory specified by the `--custom-dir` option (default is `./merged_models`).

### Configuration Options

- `--model1`, `--model2`, `--model3`: Paths or Hugging Face model IDs for the models to merge (at least two required)
- `--ps-technique1`: First parameter space merging technique (default: linear)
- `--ps-technique2`: Second parameter space merging technique (default: ties)
- `--dfs-technique`: Deep fusion space merging technique (default: frankenmerge)
- `--weight-mask-rate`: Weight mask rate, between 0.0 and 1.0 (default: 0.0)
- `--use-weight-rescale`: Use weight rescaling (flag)
- `--mask-strategy`: Mask strategy, either "random" or "magnitude" (default: random)
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

1. **Monte Carlo Tree Search (MCTS) for Workflow Optimization**: The King Agent uses MCTS to generate and optimize workflows for complex tasks.

2. **Incentive-Based Agent Management**: An incentive model is implemented to motivate and manage multiple agents effectively.

3. **Intelligent Task Routing**: The King Agent uses a preference-based approach to route tasks to the most appropriate agents.

4. **Hierarchical Sub-Goal Generation**: The system can break down complex tasks into manageable sub-goals, improving overall task completion efficiency.

5. **Continuous Learning and Adaptation**: The King Agent implements a feedback loop to continuously improve its decision-making and task allocation processes.

6. **RAG System Integration**: The King Agent is now integrated with an Enhanced RAG (Retrieval-Augmented Generation) system for improved information retrieval and decision-making.

7. **Robust Error Handling and Logging**: Comprehensive error handling and logging have been implemented throughout the King Agent system for better debugging and monitoring.

### Components

- **KingAgent**: The main class that integrates all other components and serves as the primary interface for the King Agent system.
- **KingCoordinator**: The central component that manages interactions between different parts of the system.
- **UnifiedTaskManager**: Responsible for creating, assigning, and managing tasks across different agents.
- **DecisionMaker**: Makes complex decisions using various AI techniques, including MCTS and RAG-enhanced analysis.
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

# Initialize dependencies
comm_protocol = CommunicationProtocol()
rag_system = EnhancedRAGPipeline()

# Create KingAgent
config = KingAgentConfig(name="KingAgent", description="Main coordinator for AI Village", model="gpt-4")
king_agent = KingAgent(config, comm_protocol, rag_system)

# Register agents
await king_agent.coordinator.add_agent("sage", SageAgent(comm_protocol))
await king_agent.coordinator.add_agent("magi", MagiAgent(comm_protocol))

# Send a task message
task_message = Message(content={"description": "Analyze this dataset"})
result = await king_agent.execute_task(task_message)
```

For more detailed information about the King Agent, its components, and usage, please refer to the `agents/king/README.md` file and the `agents/king/demo.py` script for a working example.

[... Keep the rest of the existing content ...]
