markdownCopy# AI Village

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
Copy
2. Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
Copy
3. Install the required packages:
pip install -r requirements.txt
Copy
4. Set up environment variables:
Create a `.env` file in the root directory and add the following variables:
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=your_neo4j_uri
NEO4J_USER=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
Copy
## Usage

1. Start the AI Village server:
python main.py
Copy
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
Copy
## Contributing

Contributions to AI Village are welcome! Please refer to CONTRIBUTING.md for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the LICENSE file for detail