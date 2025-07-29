"""Simple named entity recognition utilities."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class NamedEntityRecognizer:
    """A lightweight named entity recognizer.

    The implementation attempts to use ``spaCy`` with the ``en_core_web_sm``
    model when available.  If spaCy or the model cannot be loaded, a
    simplistic regex based approach is used as a fall back.  The ``recognize``
    method returns a list of dictionaries where each dictionary has the keys
    ``"text"`` and ``"label"``.
    """

    def __init__(self) -> None:
        try:
            import spacy  # type: ignore

            self._nlp = spacy.load("en_core_web_sm")
            logger.debug("spaCy model loaded for NER")
        except Exception:  # pragma: no cover - best effort to load spaCy
            self._nlp = None
            logger.warning("spaCy model not available; falling back to regex based NER")

        # Regex matches sequences of capitalised words as a naive entity
        self._regex = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
        self._stop_words = {"The", "A", "An"}

    def recognize(self, text: str) -> list[dict[str, str]]:
        """Extract entities from ``text``.

        Parameters
        ----------
        text: str
            The input text from which to extract entities.

        Returns:
        -------
        List[Dict[str, str]]
            A list of recognised entities.  Each entity is represented as a
            dictionary with ``text`` and ``label`` keys.
        """
        if not text:
            return []

        # Use spaCy when available for more robust entity extraction
        if self._nlp is not None:
            doc = self._nlp(text)
            return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        # Fallback regex based entity recognition
        entities = []
        for match in self._regex.finditer(text):
            ent_text = match.group(0)
            if ent_text in self._stop_words:
                continue
            entities.append({"text": ent_text, "label": "ENTITY"})
        return entities
