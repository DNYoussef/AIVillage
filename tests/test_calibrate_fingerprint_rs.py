import subprocess
from pathlib import Path


def test_calibration_average(tmp_path):
    """Compile and run the calibrate_fingerprint Rust tool."""
    src = (
        Path(__file__).resolve().parent.parent
        / "infrastructure"
        / "shared"
        / "tools"
        / "betanet"
        / "calibrate_fingerprint.rs"
    )
    binary = tmp_path / "calibrate_fingerprint"
    subprocess.run(["rustc", src, "-O", "-o", binary], check=True)
    result = subprocess.run([binary, "10", "20", "30"], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "Calibrated fingerprint: 20.00"

