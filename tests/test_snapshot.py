import importlib
import unittest

if importlib.util.find_spec("torch") is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

import builtins

import torch

from agent_forge.geometry.snapshot import snapshot


class TestSnapshot(unittest.TestCase):
    def test_snapshot_keys(self):
        h = torch.randn(2, 3, 8)
        g = snapshot(h)
        assert "ID_nl" in g
        assert "ID_lin" in g
        assert g["ID_lin"] >= 1

    def test_twonn_fallback(self):
        import importlib as il

        import agent_forge.geometry.snapshot as snap
        orig_import = builtins.__import__
        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "twonn":
                raise ImportError
            return orig_import(name, globals, locals, fromlist, level)
        with unittest.mock.patch("builtins.__import__", side_effect=fake_import):
            snap = il.reload(snap)
            assert "id_twonn" in snap.twonn.__module__

if __name__ == "__main__":
    unittest.main()
