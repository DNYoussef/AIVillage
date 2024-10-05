from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from ..core.config import RAGConfig
from ..core.structures import IdeaUnit
import numpy as np

class DocumentProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
        self.model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(config.SUMMARIZER_MODEL)
        self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(config.SUMMARIZER_MODEL)
        self.context_window = config.CONTEXT_WINDOW_SIZE

    def process_document(self, document_text: str) -> List[IdeaUnit]:
        # Step 1: Read and summarize chunks
        chunk_summaries = self._chunk_and_summarize(document_text)

        # Step 2: Integrate summaries into a holistic summary
        holistic_summary = self._integrate_summaries(chunk_summaries)

        # Step 3: Identify idea units using vector difference
        idea_units = self._identify_idea_units(document_text, holistic_summary)

        return idea_units

    def _chunk_and_summarize(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        current_position = 0
        chunk_summaries = []

        while current_position < total_tokens:
            end_position = min(current_position + self.context_window, total_tokens)
            chunk_tokens = tokens[current_position:end_position]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Summarize the chunk
            chunk_summary = self._summarize_text(chunk_text)
            chunk_summaries.append(chunk_summary)

            current_position = end_position

        return chunk_summaries

    def _summarize_text(self, text: str) -> str:
        inputs = self.summarizer_tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=self.config.SUMMARIZER_MAX_LENGTH, truncation=True)
        summary_ids = self.summarizer_model.generate(inputs, max_length=self.config.SUMMARIZER_SUMMARY_LENGTH, min_length=25, length_penalty=5.0, num_beams=2)
        summary = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def _integrate_summaries(self, summaries: List[str]) -> str:
        combined_text = ' '.join(summaries)
        holistic_summary = self._summarize_text(combined_text)
        return holistic_summary

    def _identify_idea_units(self, text: str, holistic_summary: str) -> List[IdeaUnit]:
        sentences = self._split_into_sentences(text)
        idea_units = []
        vectors = []
        model = self.model

        # Generate vectors for sliding windows of 3 sentences
        for i in range(len(sentences) - 2):
            chunk = ' '.join(sentences[i:i+3])
            vector = model.encode(chunk)
            vectors.append((i, vector))

        # Identify topic shifts
        previous_vector = None
        current_unit_sentences = []
        for idx, (i, vector) in enumerate(vectors):
            if previous_vector is not None:
                similarity = self._cosine_similarity(vector, previous_vector)
                if similarity < self.config.TOPIC_SHIFT_THRESHOLD:
                    # Topic shift detected, save current idea unit
                    idea_unit_text = ' '.join(current_unit_sentences)
                    context_note = self._generate_context_note(idea_unit_text, holistic_summary)
                    idea_unit = IdeaUnit(text=idea_unit_text, context_note=context_note)
                    idea_units.append(idea_unit)
                    current_unit_sentences = []
            previous_vector = vector
            current_unit_sentences.append(sentences[i+2])  # Add the new sentence

        # Add the last idea unit
        if current_unit_sentences:
            idea_unit_text = ' '.join(current_unit_sentences)
            context_note = self._generate_context_note(idea_unit_text, holistic_summary)
            idea_unit = IdeaUnit(text=idea_unit_text, context_note=context_note)
            idea_units.append(idea_unit)

        return idea_units

    def _split_into_sentences(self, text: str) -> List[str]:
        import nltk
        nltk.download('punkt', quiet=True)
        sentences = nltk.tokenize.sent_tokenize(text)
        return sentences

    def _cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _generate_context_note(self, idea_unit_text: str, holistic_summary: str) -> str:
        # Generate a context note by comparing idea unit with holistic summary
        # Placeholder implementation; can be enhanced with more sophisticated methods
        context_note = f"This idea unit relates to the overall document theme: {holistic_summary}"
        return context_note