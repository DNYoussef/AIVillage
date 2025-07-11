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

