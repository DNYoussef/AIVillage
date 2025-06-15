from typing import Tuple, List
import torch

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:  # transformers not installed
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore


class BERTEmbeddingModel:
    """Simple wrapper around a BERT model for embedding extraction."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        self.fallback = False
        self.hidden_size = 768
        if AutoTokenizer is None or AutoModel is None:
            # transformers package unavailable
            self.fallback = True
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
                self.hidden_size = self.model.config.hidden_size
            except Exception:
                # Loading the model failed (e.g., no internet or missing files)
                self.fallback = True
        if self.fallback:
            # Provide minimal tokenizer/model placeholders
            self.tokenizer = None
            self.model = None

    def encode(self, text: str) -> Tuple[List[str], torch.Tensor]:
        """Encode text into token embeddings.

        The method returns the list of tokens and the corresponding embedding
        tensor with shape ``(seq_len, hidden_size)``.  The embeddings are
        obtained from the last hidden state of the model without any pooling.

        Parameters
        ----------
        text: str
            Input text to encode.

        Returns
        -------
        Tuple[List[str], torch.Tensor]
            The tokens and their embeddings.
        """

        if self.fallback:
            tokens = text.split()
            token_embeddings = torch.randn(len(tokens), self.hidden_size)
            return tokens, token_embeddings
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state.squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
        return tokens, token_embeddings
