"""Embedding utilities with graceful fallbacks."""

from typing import List
import hashlib
import numpy as np

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - transformers optional
    AutoModel = None
    AutoTokenizer = None
    torch = None
    _TRANSFORMERS_AVAILABLE = False

DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_DIM = 768


class BERTEmbeddingModel:
    """Simple wrapper around a small transformers model with a stub fallback."""

    def __init__(self, model_name: str = DEFAULT_MODEL, dimension: int = DEFAULT_DIM):
        self.dimension = dimension
        if _TRANSFORMERS_AVAILABLE:
            try:  # pragma: no cover - heavy model loading not tested
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
            except Exception:  # pragma: no cover - any loading failure results in fallback
                self.tokenizer = None
                self.model = None
                self._use_stub = True
            else:
                self._use_stub = False
        else:
            self.tokenizer = None
            self.model = None
            self._use_stub = True

    def encode(self, text: str) -> List[float]:
        """Return an embedding for ``text``.

        If transformers are unavailable or model loading failed, a deterministic
        stub embedding is returned based on a hash of the input text.
        """
        if not self._use_stub and self.tokenizer is not None and self.model is not None:
            with torch.no_grad():  # pragma: no cover - not executed in tests
                tokens = self.tokenizer(text, return_tensors="pt")
                output = self.model(**tokens)
                vec = output.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                return vec.tolist()

        # Lightweight deterministic fallback
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        arr = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        if arr.size < self.dimension:
            arr = np.tile(arr, int(np.ceil(self.dimension / arr.size)))
        arr = arr[: self.dimension]
        return (arr / 255.0).tolist()
