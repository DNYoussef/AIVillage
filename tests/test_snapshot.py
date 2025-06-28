import importlib
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch not installed")

import torch
import builtins
from agent_forge.geometry.snapshot import snapshot

class TestSnapshot(unittest.TestCase):
    def test_snapshot_keys(self):
        h = torch.randn(2, 3, 8)
        g = snapshot(h)
        self.assertIn("ID_nl", g)
        self.assertIn("ID_lin", g)
        self.assertGreaterEqual(g["ID_lin"], 1)

    def test_twonn_fallback(self):
        import importlib as il
        import agent_forge.geometry.snapshot as snap
        orig_import = builtins.__import__
        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'twonn':
                raise ImportError
            return orig_import(name, globals, locals, fromlist, level)
        with unittest.mock.patch('builtins.__import__', side_effect=fake_import):
            snap = il.reload(snap)
            self.assertIn('id_twonn', snap.twonn.__module__)

if __name__ == "__main__":
    unittest.main()
