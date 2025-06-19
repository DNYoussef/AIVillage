# Coherence and Relevance Metrics

The `measure_coherence` and `measure_relevance` utilities operate on batches of embedding vectors produced during training. Both metrics rely on cosine similarity so that the scores are bounded between ``-1`` and ``1`` before averaging. In practice the values fall in ``[0, 1]`` after averaging due to the mostly positive similarities of related embeddings. The functions expect three‑dimensional ``(batch, seq, dim)`` arrays and return ``1.0`` when a sequence contains only a single thought.

* **Coherence** – evaluates how well consecutive thought vectors align. For each pair of neighbouring thought embeddings in a sequence we compute the cosine similarity and then average over the sequence and batch. Perfectly identical consecutive thoughts yield a score of ``1`` while unrelated thoughts approach ``0``.
* **Relevance** – measures the relationship between the thoughts and target embeddings. Thought and target sequences are averaged to single vectors and compared using cosine similarity. A score near ``1`` indicates the thoughts closely match the targets.

These metrics provide lightweight estimates of logical flow and task focus without requiring textual generation. They can be extended with more advanced linguistic analysis if needed.
