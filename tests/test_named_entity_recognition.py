import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag_system.utils.named_entity_recognition import NamedEntityRecognizer


class TestNamedEntityRecognizer(unittest.TestCase):
    def test_simple_regex_entities(self):
        text = "John Doe visited New York City yesterday."
        ner = NamedEntityRecognizer()
        entities = ner.recognize(text)
        texts = [e["text"] for e in entities]
        self.assertIn("John Doe", texts)
        self.assertIn("New York City", texts)


if __name__ == "__main__":
    unittest.main()
