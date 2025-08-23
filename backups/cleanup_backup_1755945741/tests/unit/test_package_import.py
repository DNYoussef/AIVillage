import importlib


def test_aivillage_package_redirect():
    importlib.import_module("AIVillage")
    runner = importlib.import_module("AIVillage.twin_runtime.runner")
    legacy = importlib.import_module("AIVillage.src.twin_runtime.runner")
    assert hasattr(runner, "chat")
    assert callable(runner.chat)
    assert legacy.chat("ping") == runner.chat("ping")
