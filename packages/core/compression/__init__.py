"""Core compression utilities."""

class SimpleQuantizer:
    """Basic quantizer for testing purposes."""

    def __init__(self, bits=8):
        self.bits = bits

    def quantize(self, data):
        """Quantize input data using basic bit reduction."""
        import numpy as np
        if hasattr(data, 'numpy'):
            data = data.numpy()
        if isinstance(data, np.ndarray):
            # Simple quantization: scale and round to available bit levels
            scale = (2 ** self.bits - 1)
            data_min, data_max = data.min(), data.max()
            if data_max > data_min:
                normalized = (data - data_min) / (data_max - data_min)
                quantized = np.round(normalized * scale) / scale
                return quantized * (data_max - data_min) + data_min
        return data

    def dequantize(self, data):
        """Dequantize input data (identity operation for basic quantizer)."""
        # For this simple quantizer, dequantization is identity since
        # quantization already returns data in original range
        return data

    def quantize_model(self, model):
        """Apply basic quantization to model parameters."""
        # For basic quantization, iterate through model parameters
        if hasattr(model, 'parameters'):
            for param in model.parameters():
                if param.requires_grad:
                    param.data = self.quantize(param.data)
        return model

__all__ = ["SimpleQuantizer"]
