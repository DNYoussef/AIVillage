from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertForQuestionAnswering, BertForSequenceClassification, BertModel, BertTokenizer


class AdvancedNLP:
    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_classifier = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        self.bert_qa = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate BERT embeddings for a list of texts."""
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.bert_model(**encoded_input)
        return model_output.last_hidden_state[:, 0, :].numpy()

    def classify_text(self, texts: list[str], labels: list[str]) -> list[str]:
        """Classify texts into predefined labels using BERT."""
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.bert_classifier(**encoded_input).logits
        predictions = torch.argmax(logits, dim=1)
        return [labels[pred] for pred in predictions]

    def answer_question(self, context: str, question: str) -> str:
        """Answer a question based on the given context using BERT."""
        inputs = self.tokenizer(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.bert_qa(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )
        return answer

    def semantic_search(self, query: str, documents: list[str], top_k: int = 5) -> list[dict[str, Any]]:
        """Perform semantic search on a list of documents using BERT embeddings."""
        query_embedding = self.get_embeddings([query])
        doc_embeddings = self.get_embeddings(documents)

        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({"document": documents[idx], "similarity": similarities[idx]})

        return results

    def extract_keywords(self, text: str, top_k: int = 5) -> list[str]:
        """Extract keywords from the given text using BERT."""
        tokens = self.tokenizer.tokenize(text)
        token_embeddings = self.get_embeddings(tokens)

        # Calculate the centroid of all token embeddings
        centroid = np.mean(token_embeddings, axis=0)

        # Calculate cosine similarity between each token and the centroid
        similarities = cosine_similarity(token_embeddings, [centroid])[:, 0]

        # Get the top-k tokens with highest similarity to the centroid
        top_indices = similarities.argsort()[-top_k:][::-1]
        keywords = [tokens[idx] for idx in top_indices]

        return keywords

    def generate_summary(self, text: str, max_length: int = 100) -> str:
        """Generate a summary of the given text using BERT."""
        inputs = self.tokenizer([text], max_length=512, return_tensors="pt", truncation=True)
        summary_ids = self.bert_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def analyze_sentiment(self, text: str) -> dict[str, float]:
        """Analyze the sentiment of the given text using BERT."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_classifier(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment_scores = {
            "positive": probabilities[0][1].item(),
            "negative": probabilities[0][0].item(),
        }
        return sentiment_scores
