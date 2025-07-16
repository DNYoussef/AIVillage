from pathlib import Path
import sys
import unittest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag_system.utils.named_entity_recognition import NamedEntityRecognizer


class TestNamedEntityRecognizer(unittest.TestCase):
    def test_simple_regex_entities(self):
        text = "John Doe visited New York City yesterday."
        ner = NamedEntityRecognizer()
        entities = ner.recognize(text)
        texts = [e["text"] for e in entities]
        assert "John Doe" in texts
        assert "New York City" in texts


if __name__ == "__main__":
    unittest.main()
