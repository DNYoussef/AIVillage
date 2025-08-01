# Advanced Setup

This guide covers environment variables, heavy dependencies and testing helpers for the AI Village system.

## Environment Variables

Create a `.env` file in the project root and define at least:

```
OPENAI_API_KEY=your_openai_api_key
API_KEY=my_secret_key
```

Additional variables such as `NEO4J_URI` and `NEO4J_USER` may be required when enabling optional services.

## Heavy Dependencies

Some features rely on large libraries like `numpy`, `torch` and `faiss`. Install them separately when GPU support is needed:

```bash
pip install numpy torch faiss-cpu  # or faiss-gpu for CUDA systems
```

The geometry-aware helpers attempt to use `torch-twonn` for intrinsic dimension estimation but will fall back to the lightweight implementation if not installed.

Ensure the file `rag_system/utils/token_data/cl100k_base.tiktoken` exists before running the code. Download it from the [tiktoken repository](https://github.com/openai/tiktoken) if missing.

## Testing

After installing heavy dependencies you can run the full test suite:

```bash
pytest
```

The helper script `scripts/setup_env.sh` installs both runtime and development requirements including `scikit-learn`.
