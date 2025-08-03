# Unified Compression Pipeline

This repository previously contained several disjoint compression modules
spread across `src/compression`, `src/core/compression` and
`src/production/compression`.  The new `src/compression/pipeline.py`
provides a single entry point that wraps the `UnifiedCompressor` from the
core package.  It automatically selects between a lightweight quantizer and a
multi-stage advanced compressor.

## Usage

```python
from src.compression import pipeline

model = ...  # any ``torch.nn.Module``
result = pipeline.compress(model)
restored = pipeline.decompress(result)
```

Additional parameters such as `target_compression` can be passed to
`pipeline.compress` to force the advanced pipeline.
