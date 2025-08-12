import importlib
import subprocess
import sys


def test_package_installs_and_imports(tmp_path):
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        ".",
        "--no-deps",
        "--target",
        str(tmp_path),
    ])
    sys.path.insert(0, str(tmp_path))
    module = importlib.import_module(
        "AIVillage.ingestion.connectors.amazon_orders"
    )
    assert hasattr(module, "AmazonOrdersConnector")
