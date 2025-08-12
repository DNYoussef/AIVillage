import importlib


def test_smoke_imports():
    importlib.import_module("AIVillage")
    modules = [
        "mcp_servers.hyperag.guardian.gate",
        "production.distributed_inference.adaptive_resharding",
    ]
    for mod in modules:
        importlib.import_module(mod)
