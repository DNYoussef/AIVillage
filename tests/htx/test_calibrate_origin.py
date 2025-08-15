import importlib.util
import pathlib
import pytest

# Import HTXTransport directly to avoid heavy package initialization
spec = importlib.util.spec_from_file_location(
    "betanet_htx_transport",
    pathlib.Path(__file__).resolve().parents[2]
    / "src/core/p2p/betanet_htx_transport.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
HTXTransport = mod.HTXTransport


@pytest.mark.asyncio
async def test_calibrate_origin_example_com():
    transport = HTXTransport()
    calibration = await transport.calibrate_origin("example.com")
    assert calibration.origin_host == "example.com"
    assert len(calibration.ja3_fingerprint) == 32
    assert len(calibration.ja4_fingerprint) == 16
    assert calibration.cipher_suites
    assert calibration.extensions_order
