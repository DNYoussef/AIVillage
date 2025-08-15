import importlib


def test_lookup_fallback(tmp_path, monkeypatch):
    module = importlib.import_module('src.communications.service_directory')
    monkeypatch.setattr(module, '_CACHE_PATH', tmp_path / 'agents.json')
    sd = module.ServiceDirectory()
    monkeypatch.setenv('COMM_DEFAULT_HOST', 'localhost')
    monkeypatch.setenv('COMM_DEFAULT_PORT', '9999')
    assert sd.lookup('missing') == 'ws://localhost:9999/ws'
    sd.register('agent', 'ws://host:1234/ws')
    assert sd.lookup('agent') == 'ws://host:1234/ws'
