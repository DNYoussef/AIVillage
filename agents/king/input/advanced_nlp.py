import logging
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertForSequenceClassification, BertModel, BertTokenizer

logger = logging.getLogger(__name__)


class AdvancedNLP:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            revision="main",  # Pin to main branch for security
            trust_remote_code=False  # Disable remote code execution
        )
        self.model = BertModel.from_pretrained(
            "bert-base-uncased",
            revision="main",
            trust_remote_code=False
        )
        self.classifier = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            num_labels=2,
            revision="main",
            trust_remote_code=False
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.classifier.to(self.device)

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate BERT embeddings for a list of texts."""
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Use the CLS token embedding as the sentence embedding
        sentence_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        return sentence_embeddings

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using BERT embeddings."""
        embeddings = self.get_embeddings([text1, text2])
        similarity = cosine_similarity(embeddings)[0][1]
        return similarity

    def classify_sentiment(self, text: str) -> dict[str, float]:
        """Classify the sentiment of a given text using BERT."""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.classifier(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        positive_prob = probabilities[0][1].item()
        negative_prob = probabilities[0][0].item()

        return {"positive": positive_prob, "negative": negative_prob}

    def extract_keywords(self, text: str, top_k: int = 5) -> list[str]:
        """Extract top-k keywords from the given text using BERT embeddings."""
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)

        # Get embeddings for each token
        encoded_input = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            outputs = self.model(**encoded_input)

        token_embeddings = outputs.last_hidden_state[0].cpu().numpy()

        # Calculate the importance of each token based on its similarity to the overall sentence embedding
        sentence_embedding = token_embeddings.mean(axis=0)
        token_importance = cosine_similarity(token_embeddings, [sentence_embedding])

        # Get the top-k important tokens
        top_indices = token_importance.argsort()[0][-top_k:]
        keywords = [
            tokens[i]
            for i in top_indices
            if tokens[i] not in self.tokenizer.all_special_tokens
        ]

        return keywords

    def generate_summary(self, text: str, max_length: int = 100) -> str:
        """Generate a summary of the given text using BERT."""
        inputs = self.tokenizer(
            [text], max_length=512, return_tensors="pt", truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use the CLS token embedding as the document embedding
        document_embedding = outputs.last_hidden_state[:, 0, :]

        # Split the text into sentences
        sentences = text.split(".")
        sentence_embeddings = self.get_embeddings(sentences)

        # Calculate similarity between each sentence and the document
        similarities = cosine_similarity(sentence_embeddings, document_embedding)

        # Sort sentences by similarity and select top sentences
        sorted_indices = similarities.argsort(axis=0)[::-1]
        selected_sentences = []
        current_length = 0

        for idx in sorted_indices.flatten():
            sentence = sentences[idx].strip()
            if current_length + len(sentence) <= max_length:
                selected_sentences.append(sentence)
                current_length += len(sentence)
            else:
                break

        # Join selected sentences to form the summary
        summary = ". ".join(selected_sentences)
        return summary

    async def process_text(self, text: str) -> dict[str, Any]:
        """Process the given text using various NLP techniques."""
        try:
            embeddings = self.get_embeddings([text])[0]
            sentiment = self.classify_sentiment(text)
            keywords = self.extract_keywords(text)
            summary = self.generate_summary(text)

            return {
                "embeddings": embeddings.tolist(),
                "sentiment": sentiment,
                "keywords": keywords,
                "summary": summary,
            }
        except Exception as e:
            logger.error(f"Error processing text: {e!s}")
            raise

    async def compare_texts(self, text1: str, text2: str) -> dict[str, Any]:
        """Compare two texts using various NLP techniques."""
        try:
            similarity = self.semantic_similarity(text1, text2)
            embeddings1 = self.get_embeddings([text1])[0]
            embeddings2 = self.get_embeddings([text2])[0]
            sentiment1 = self.classify_sentiment(text1)
            sentiment2 = self.classify_sentiment(text2)
            keywords1 = self.extract_keywords(text1)
            keywords2 = self.extract_keywords(text2)

            return {
                "similarity": similarity,
                "embeddings1": embeddings1.tolist(),
                "embeddings2": embeddings2.tolist(),
                "sentiment1": sentiment1,
                "sentiment2": sentiment2,
                "keywords1": keywords1,
                "keywords2": keywords2,
            }
        except Exception as e:
            logger.error(f"Error comparing texts: {e!s}")
            raise
