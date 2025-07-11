# Migration Notes

## Vector Store Format Change

The persisted `VectorStore` now uses a JSON file (`vector_store.json`) instead of the previous pickle format (`vector_store.pkl`).

Existing pickled stores must be converted or regenerated before use with the updated code.
