import importlib


def test_knowledge_constructor_imports() -> None:
    """Ensure modules using relative imports load correctly."""
    importlib.import_module(
        "src.production.rag.rag_system.processing.knowledge_constructor"
    )
