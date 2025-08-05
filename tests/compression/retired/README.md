# Retired Compression Test Files

This directory contains compression test files that have been consolidated into the unified compression system.

## What Was Retired

These files were fragmented test implementations that created redundancy and maintenance overhead:

### Demo Files (28 files retired)
- `compression_demo.py` - Basic compression demonstration
- `compression_improvements_demo.py` - Enhanced compression demo
- `quick_compression_demo.py` - Fast compression test
- `prove_compression*.py` - Compression validation scripts
- `validate_*compression*.py` - Various validation scripts

### Test Files (20+ files retired)
- `test_*compression*.py` - Multiple compression test implementations
- `test_*stages*.py` - Stage-specific compression tests  
- `test_*model*.py` - Model-specific compression tests
- `test_*1_5b*.py` - 1.5B parameter model tests
- `test_real_*.py` - Real model compression tests
- `test_synthetic_*.py` - Synthetic model tests

## Why They Were Retired

1. **Redundancy**: Multiple files testing the same functionality
2. **Fragmentation**: Tests scattered across 28+ files instead of organized suites
3. **Inconsistent Standards**: Mix of pytest and script-style tests
4. **Unverified Claims**: Some tests produced questionable compression ratios (458x)
5. **Maintenance Burden**: Too many files to maintain and update

## What Replaced Them

All functionality has been consolidated into:

- **`tests/compression/test_unified_compression.py`** - Comprehensive test suite
- **`src/production/compression/unified_compressor.py`** - Single compression interface
- **`src/production/compression/README.md`** - Complete documentation

## Migration Guide

If you need functionality from these retired files:

### For Basic Compression Testing
```python
from src.production.compression import compress_simple
result = await compress_simple(model)
```

### For Advanced Compression Testing  
```python
from src.production.compression import UnifiedCompressor, CompressionStrategy
compressor = UnifiedCompressor(strategy=CompressionStrategy.ADVANCED)
result = await compressor.compress_model(model)
```

### For Running Tests
```bash
# Run unified compression tests
pytest tests/compression/test_unified_compression.py -v

# Run specific test categories
pytest tests/compression/test_unified_compression.py::TestUnifiedCompressor -v
```

## Historical Note

These files represented the evolution of the AIVillage compression system from:
- Simple 4x quantization (working)
- Claims of 20.8x compression (plausible)  
- Claims of 458x compression (questionable)

The unified system focuses on verified, realistic compression ratios with proper fallback mechanisms and comprehensive testing.

## File Archive

The retired files are preserved here for reference but should not be used in new development. They may be permanently deleted in a future cleanup.
