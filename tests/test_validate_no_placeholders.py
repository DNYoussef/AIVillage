import subprocess
from pathlib import Path

def test_stub_reference_is_allowed(tmp_path):
    prod_dir = tmp_path / "prod"
    prod_dir.mkdir()
    sample = prod_dir / "example.py"
    sample.write_text(
        "def stub_function():\n"
        "    return 'stub implementation'\n"
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_no_placeholders.sh"
    result = subprocess.run([
        "bash",
        str(script),
        "-d",
        str(prod_dir),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
