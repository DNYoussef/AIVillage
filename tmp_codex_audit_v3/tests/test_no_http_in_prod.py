from pathlib import Path

TARGET_DIRS = [
    Path("src/communications"),
    Path("src/production"),
    Path("src/digital_twin"),
]


def test_no_http_in_prod():
    for base in TARGET_DIRS:
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "http://" not in text, f"http:// found in {path}"
