import importlib


def test_smoke_imports():
    """Ensure key modules import successfully via the AIVillage namespace."""
    importlib.import_module("AIVillage.ingestion.connectors.amazon_orders")
