# Usage Examples

These examples illustrate how to interact with the system once the server is running.

## Running the Server

Start the API server with:

```bash
python server.py
```

Open `http://localhost:8000/` to access the simple dashboard. You can submit queries, upload text files and inspect retrieval logs.

## Stand-alone RAG Example

Run the pipeline without the web server:

```bash
python rag_system/main.py
```

This issues a sample query and performs a short knowledge graph exploration.

## Populating the RAG System with Academic Papers

1. Place your papers in a directory.
2. Start the server:
   ```bash
   python server.py
   ```
3. Upload each paper via `http://localhost:8000/upload`. Example curl command:
   ```bash
   curl -X POST -H "Content-Type: multipart/form-data" -F "file=@path/to/paper.pdf" http://localhost:8000/upload
   ```
4. Test retrieval with the `/query` endpoint:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"query": "Summarize the key findings"}' http://localhost:8000/query
   ```

Adjust the RAG configuration in `configs/rag_config.yaml` to optimise for paper processing.
