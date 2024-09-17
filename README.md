# AI Village

AI Village is an advanced Retrieval-Augmented Generation (RAG) system that combines vector and graph-based storage with active learning and planning capabilities. It's designed to provide intelligent responses to queries by leveraging a comprehensive knowledge base.

## Features

- Hybrid RAG system with vector and graph storage
- Active learning for query refinement
- Planning-aware retrieval for optimized search strategies
- Community-aware search within the knowledge graph
- Integration with multiple AI agents (Archive, King, Sage, Magi)
- Flexible pipeline for query processing and knowledge management
- Built on top of the Langroid framework for enhanced AI capabilities

## Installation

1. Clone the repository:
git clone https://github.com/your-username/ai-village.git
cd ai-village

2. Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

3. Install the required packages:
pip install -r requirements.txt

4. Set up environment variables:
- Create a `.env` file in the root directory and add the following variables:
- `OPENAI_API_KEY`={{your_openai_api_key}}
- `NEO4J_URI`={{your_neo4j_uri}}
- `NEO4J_USER`={{your_neo4j_username}}
- `NEO4J_PASSWORD`={{your_neo4j_password}}

## Usage

1. Start the AI Village server:
`python main.py`

2. The server will start running on `http://localhost:8000`. You can now use the following endpoints:

- POST `/query`: Send a query to the AI Village
- POST `/upload`: Upload a file to populate the knowledge base
- POST `/import_open_researcher`: Import data from the Open Researcher project

3. Use a tool like curl or Postman to interact with the API, or integrate it into your application.

## Documentation

For more detailed information about the AI Village architecture, usage, and API reference, please refer to the documents in the `docs/` directory.

## Testing

Run the test suite using pytest:
pytest tests/

## Contributing

Contributions to AI Village are welcome! Please refer to CONTRIBUTING.md for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the LICENSE file for detail



## Model Mergers

AI Village includes functionality for merging language models using the MergeKit library. This allows you to combine different models to create a new, potentially more powerful model.

### Available Mergers

1. **Simple Merger**: Combines two or more models using a simple averaging technique.
2. **Weighted Merger**: Merges models with custom weights assigned to each input model.
3. **Task-Specific Merger**: Optimizes the merged model for specific tasks by adjusting layer contributions.

### Usage

To use the model mergers, follow these steps:

1. Navigate to the `agent_forge` directory:
   ```
   cd agent_forge
   ```

2. Run the merger script with the desired configuration:
   ```
   python main.py --config configs/merge_config.yaml
   ```

3. The merged model will be saved in the `./merged_model` directory by default.

### Configuration

Merger configurations are specified in YAML files located in the `configs/` directory. You can create custom configurations by modifying existing files or creating new ones.

Example configuration (`configs/merge_config.yaml`):

```yaml
base_model: ollama:llama2
models_to_merge:
  - name: ollama:llama2
    weight: 0.7
  - name: ollama:codellama
    weight: 0.3
merge_method: weighted
output_path: ./merged_model
```

### Commands

- To list available models:
  ```
  python main.py --list-models
  ```

- To merge models using a specific configuration:
  ```
  python main.py --config configs/your_config.yaml
  ```

- To specify a custom output path:
  ```
  python main.py --config configs/your_config.yaml --output ./custom_output
  ```

For more detailed information on model merging and advanced configurations, please refer to the `docs/model_merging.md` file.


