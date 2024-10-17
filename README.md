# AI Village

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
