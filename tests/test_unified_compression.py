from torch import nn

from core.compression.unified_compressor import UnifiedCompressor


class TestUnifiedCompression:
    """Test the unified compression system"""

    def test_small_model_uses_simple(self):
        model = nn.Linear(100, 100)
        compressor = UnifiedCompressor()
        result = compressor.compress(model)
        assert result["method"] == "simple"
        assert "data" in result
        original_size = 100 * 100 * 4
        compressed_size = len(result["data"])
        ratio = original_size / compressed_size
        assert ratio >= 3.5

    def test_large_model_uses_advanced(self):
        model = nn.Sequential(
            nn.Linear(10000, 5000),
            nn.Linear(5000, 5000),
            nn.Linear(5000, 5000),
        )
        compressor = UnifiedCompressor()
        result = compressor.compress(model)
        assert result["method"] == "advanced"
        assert result["stages"] == ["bitnet", "seedlm", "vptq", "hyper"]

    def test_fallback_on_advanced_failure(self):
        model = nn.Linear(1000, 1000)
        compressor = UnifiedCompressor(memory_limit_mb=1, target_compression=1000.0)

        def failing_compress(*args, **kwargs):  # pragma: no cover - mocked failure
            raise RuntimeError("Advanced compression failed")

        compressor.advanced.compress_model = failing_compress  # type: ignore
        result = compressor.compress(model)
        assert result["method"] == "simple"

    def test_backwards_compatibility(self):
        from core.compression.simple_quantizer import SimpleQuantizer

        model = nn.Linear(100, 100)
        simple = SimpleQuantizer()
        old = simple.quantize_model(model)
        unified = UnifiedCompressor()
        wrapped = {"method": "simple", "data": old}
        decompressed = unified.decompress(wrapped)
        assert decompressed is not None
