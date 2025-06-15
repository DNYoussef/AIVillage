from typing import Tuple, List
import torch
from transformers import AutoTokenizer, AutoModel


class BERTEmbeddingModel:
    """Simple wrapper around a BERT model for embedding extraction."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

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

        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state.squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
        return tokens, token_embeddings
