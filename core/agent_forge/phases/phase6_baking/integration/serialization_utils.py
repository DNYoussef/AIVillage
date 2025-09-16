"""
Robust Serialization Utilities for Phase 6 Integration Pipeline

This module provides comprehensive serialization support for PyTorch tensors,
NumPy arrays, datetime objects, and other complex types that cause JSON
serialization errors in the baking pipeline.
"""

import json
import pickle
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, date, time
from dataclasses import dataclass, asdict, is_dataclass
from decimal import Decimal
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SerializationConfig:
    """Configuration for serialization operations"""
    use_compression: bool = True
    include_metadata: bool = True
    fail_on_unsupported: bool = False
    max_tensor_size_mb: float = 100.0
    datetime_format: str = "%Y-%m-%d %H:%M:%S.%f"

class EnhancedJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder that handles PyTorch tensors, NumPy arrays, and datetime objects"""

    def __init__(self, config: Optional[SerializationConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or SerializationConfig()

    def default(self, obj):
        """Convert non-JSON serializable objects to JSON serializable format"""
        try:
            # Handle datetime objects
            if isinstance(obj, datetime):
                return {
                    '__type__': 'datetime',
                    '__value__': obj.strftime(self.config.datetime_format)
                }

            if isinstance(obj, date):
                return {
                    '__type__': 'date',
                    '__value__': obj.isoformat()
                }

            if isinstance(obj, time):
                return {
                    '__type__': 'time',
                    '__value__': obj.isoformat()
                }

            # Handle NumPy arrays
            if isinstance(obj, np.ndarray):
                return self._serialize_numpy_array(obj)

            # Handle NumPy scalars
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()

            # Handle PyTorch tensors
            if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
                return self._serialize_torch_tensor(obj)

            # Handle Decimal
            if isinstance(obj, Decimal):
                return {
                    '__type__': 'decimal',
                    '__value__': str(obj)
                }

            # Handle dataclasses
            if is_dataclass(obj):
                return {
                    '__type__': 'dataclass',
                    '__class__': obj.__class__.__name__,
                    '__module__': obj.__class__.__module__,
                    '__value__': asdict(obj)
                }

            # Handle Path objects
            if isinstance(obj, Path):
                return {
                    '__type__': 'path',
                    '__value__': str(obj)
                }

            # Handle sets
            if isinstance(obj, set):
                return {
                    '__type__': 'set',
                    '__value__': list(obj)
                }

            # Handle complex numbers
            if isinstance(obj, complex):
                return {
                    '__type__': 'complex',
                    '__value__': [obj.real, obj.imag]
                }

            # Handle bytes
            if isinstance(obj, bytes):
                return {
                    '__type__': 'bytes',
                    '__value__': base64.b64encode(obj).decode('utf-8')
                }

            # Try to handle custom objects with __dict__
            if hasattr(obj, '__dict__'):
                return {
                    '__type__': 'object',
                    '__class__': obj.__class__.__name__,
                    '__module__': obj.__class__.__module__,
                    '__value__': obj.__dict__
                }

            # If fail_on_unsupported is False, convert to string
            if not self.config.fail_on_unsupported:
                return {
                    '__type__': 'string_repr',
                    '__value__': str(obj)
                }

            # Default behavior - raise TypeError
            return super().default(obj)

        except Exception as e:
            logger.error(f"Serialization error for {type(obj)}: {e}")
            if self.config.fail_on_unsupported:
                raise
            return {
                '__type__': 'serialization_error',
                '__value__': f"Failed to serialize {type(obj).__name__}: {str(e)}"
            }

    def _serialize_numpy_array(self, array: np.ndarray) -> Dict[str, Any]:
        """Serialize NumPy array to JSON-compatible format"""
        # Check size constraints
        size_mb = array.nbytes / (1024 * 1024)
        if size_mb > self.config.max_tensor_size_mb:
            return {
                '__type__': 'numpy_array_large',
                '__shape__': array.shape,
                '__dtype__': str(array.dtype),
                '__size_mb__': size_mb,
                '__value__': 'Array too large for JSON serialization'
            }

        serialized = {
            '__type__': 'numpy_array',
            '__shape__': array.shape,
            '__dtype__': str(array.dtype)
        }

        # For small arrays, include the data
        if array.size < 1000:  # Small arrays
            serialized['__value__'] = array.tolist()
        else:
            # For larger arrays, use base64 encoding
            if self.config.use_compression:
                import gzip
                compressed_data = gzip.compress(array.tobytes())
                serialized['__value__'] = base64.b64encode(compressed_data).decode('utf-8')
                serialized['__compressed__'] = True
            else:
                serialized['__value__'] = base64.b64encode(array.tobytes()).decode('utf-8')
                serialized['__compressed__'] = False

        if self.config.include_metadata:
            serialized['__metadata__'] = {
                'size': array.size,
                'nbytes': array.nbytes,
                'ndim': array.ndim
            }

        return serialized

    def _serialize_torch_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Serialize PyTorch tensor to JSON-compatible format"""
        # Convert to NumPy first (detach if needed)
        try:
            if tensor.requires_grad:
                numpy_array = tensor.detach().cpu().numpy()
            else:
                numpy_array = tensor.cpu().numpy()
        except Exception as e:
            return {
                '__type__': 'torch_tensor_error',
                '__value__': f"Failed to convert tensor to numpy: {str(e)}",
                '__shape__': list(tensor.shape) if hasattr(tensor, 'shape') else None,
                '__device__': str(tensor.device) if hasattr(tensor, 'device') else None
            }

        # Check size constraints
        size_mb = numpy_array.nbytes / (1024 * 1024)
        if size_mb > self.config.max_tensor_size_mb:
            return {
                '__type__': 'torch_tensor_large',
                '__shape__': list(tensor.shape),
                '__dtype__': str(tensor.dtype),
                '__device__': str(tensor.device),
                '__size_mb__': size_mb,
                '__value__': 'Tensor too large for JSON serialization'
            }

        serialized = {
            '__type__': 'torch_tensor',
            '__shape__': list(tensor.shape),
            '__dtype__': str(tensor.dtype),
            '__device__': str(tensor.device),
            '__requires_grad__': tensor.requires_grad
        }

        # For small tensors, include the data
        if tensor.numel() < 1000:  # Small tensors
            serialized['__value__'] = numpy_array.tolist()
        else:
            # For larger tensors, use base64 encoding
            if self.config.use_compression:
                import gzip
                compressed_data = gzip.compress(numpy_array.tobytes())
                serialized['__value__'] = base64.b64encode(compressed_data).decode('utf-8')
                serialized['__compressed__'] = True
            else:
                serialized['__value__'] = base64.b64encode(numpy_array.tobytes()).decode('utf-8')
                serialized['__compressed__'] = False

        if self.config.include_metadata:
            serialized['__metadata__'] = {
                'numel': tensor.numel(),
                'nbytes': numpy_array.nbytes,
                'ndim': tensor.ndim
            }

        return serialized

class EnhancedJSONDecoder:
    """Enhanced JSON decoder that reconstructs PyTorch tensors, NumPy arrays, and datetime objects"""

    def __init__(self, config: Optional[SerializationConfig] = None):
        self.config = config or SerializationConfig()

    def decode(self, json_str: str) -> Any:
        """Decode JSON string with enhanced object reconstruction"""
        return json.loads(json_str, object_hook=self._object_hook)

    def _object_hook(self, obj):
        """Convert specially encoded objects back to their original types"""
        if not isinstance(obj, dict) or '__type__' not in obj:
            return obj

        obj_type = obj['__type__']

        try:
            if obj_type == 'datetime':
                return datetime.strptime(obj['__value__'], self.config.datetime_format)

            elif obj_type == 'date':
                return datetime.fromisoformat(obj['__value__']).date()

            elif obj_type == 'time':
                return datetime.fromisoformat(f"1970-01-01T{obj['__value__']}").time()

            elif obj_type == 'numpy_array':
                return self._deserialize_numpy_array(obj)

            elif obj_type == 'torch_tensor':
                return self._deserialize_torch_tensor(obj)

            elif obj_type == 'decimal':
                return Decimal(obj['__value__'])

            elif obj_type == 'path':
                return Path(obj['__value__'])

            elif obj_type == 'set':
                return set(obj['__value__'])

            elif obj_type == 'complex':
                real, imag = obj['__value__']
                return complex(real, imag)

            elif obj_type == 'bytes':
                return base64.b64decode(obj['__value__'].encode('utf-8'))

            elif obj_type in ['numpy_array_large', 'torch_tensor_large']:
                # Return metadata for large arrays/tensors
                return {
                    'type': obj_type,
                    'shape': obj.get('__shape__'),
                    'dtype': obj.get('__dtype__'),
                    'size_mb': obj.get('__size_mb__'),
                    'message': obj.get('__value__')
                }

            elif obj_type in ['torch_tensor_error', 'serialization_error', 'string_repr']:
                # Return error/string representations as-is
                return obj['__value__']

            else:
                logger.warning(f"Unknown object type for deserialization: {obj_type}")
                return obj

        except Exception as e:
            logger.error(f"Deserialization error for {obj_type}: {e}")
            return obj  # Return original object if deserialization fails

    def _deserialize_numpy_array(self, obj: Dict[str, Any]) -> np.ndarray:
        """Deserialize NumPy array from JSON-compatible format"""
        shape = tuple(obj['__shape__'])
        dtype = np.dtype(obj['__dtype__'])

        if isinstance(obj['__value__'], list):
            # Small array stored as list
            return np.array(obj['__value__'], dtype=dtype).reshape(shape)
        else:
            # Large array stored as base64
            if obj.get('__compressed__', False):
                import gzip
                compressed_data = base64.b64decode(obj['__value__'].encode('utf-8'))
                array_bytes = gzip.decompress(compressed_data)
            else:
                array_bytes = base64.b64decode(obj['__value__'].encode('utf-8'))

            return np.frombuffer(array_bytes, dtype=dtype).reshape(shape)

    def _deserialize_torch_tensor(self, obj: Dict[str, Any]) -> torch.Tensor:
        """Deserialize PyTorch tensor from JSON-compatible format"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for tensor deserialization")

        shape = tuple(obj['__shape__'])
        dtype_str = obj['__dtype__']
        device_str = obj['__device__']
        requires_grad = obj.get('__requires_grad__', False)

        # Parse dtype
        if hasattr(torch, dtype_str.split('.')[-1]):
            dtype = getattr(torch, dtype_str.split('.')[-1])
        else:
            dtype = torch.float32  # fallback

        if isinstance(obj['__value__'], list):
            # Small tensor stored as list
            tensor_data = np.array(obj['__value__']).reshape(shape)
            tensor = torch.from_numpy(tensor_data).to(dtype)
        else:
            # Large tensor stored as base64
            if obj.get('__compressed__', False):
                import gzip
                compressed_data = base64.b64decode(obj['__value__'].encode('utf-8'))
                array_bytes = gzip.decompress(compressed_data)
            else:
                array_bytes = base64.b64decode(obj['__value__'].encode('utf-8'))

            # Infer numpy dtype from torch dtype
            if dtype == torch.float32:
                np_dtype = np.float32
            elif dtype == torch.float64:
                np_dtype = np.float64
            elif dtype == torch.int32:
                np_dtype = np.int32
            elif dtype == torch.int64:
                np_dtype = np.int64
            else:
                np_dtype = np.float32  # fallback

            tensor_data = np.frombuffer(array_bytes, dtype=np_dtype).reshape(shape)
            tensor = torch.from_numpy(tensor_data).to(dtype)

        # Set device
        if device_str != 'cpu' and torch.cuda.is_available():
            try:
                tensor = tensor.to(device_str)
            except Exception:
                logger.warning(f"Could not move tensor to {device_str}, keeping on CPU")

        # Set requires_grad
        if requires_grad:
            tensor.requires_grad_(True)

        return tensor

class SafeJSONSerializer:
    """Safe JSON serializer that handles all common Phase 6 data types"""

    def __init__(self, config: Optional[SerializationConfig] = None):
        self.config = config or SerializationConfig()
        self.encoder = EnhancedJSONEncoder(self.config)
        self.decoder = EnhancedJSONDecoder(self.config)

    def serialize(self, obj: Any) -> str:
        """Serialize object to JSON string"""
        try:
            return json.dumps(obj, cls=lambda *args, **kwargs: self.encoder, indent=2)
        except Exception as e:
            logger.error(f"JSON serialization failed: {e}")
            raise SerializationError(f"Failed to serialize object: {e}")

    def deserialize(self, json_str: str) -> Any:
        """Deserialize JSON string to object"""
        try:
            return self.decoder.decode(json_str)
        except Exception as e:
            logger.error(f"JSON deserialization failed: {e}")
            raise SerializationError(f"Failed to deserialize JSON: {e}")

    def serialize_to_file(self, obj: Any, file_path: Union[str, Path]) -> bool:
        """Serialize object to JSON file"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(obj, f, cls=lambda *args, **kwargs: self.encoder, indent=2)

            logger.info(f"Successfully serialized object to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to serialize to file {file_path}: {e}")
            return False

    def deserialize_from_file(self, file_path: Union[str, Path]) -> Any:
        """Deserialize object from JSON file"""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f, object_hook=self.decoder._object_hook)

        except Exception as e:
            logger.error(f"Failed to deserialize from file {file_path}: {e}")
            raise SerializationError(f"Failed to deserialize from file: {e}")

class PickleSerializer:
    """Fallback pickle serializer for objects that can't be JSON serialized"""

    def __init__(self, compression: bool = True):
        self.compression = compression

    def serialize(self, obj: Any) -> bytes:
        """Serialize object using pickle"""
        try:
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

            if self.compression:
                import gzip
                data = gzip.compress(data)

            return data

        except Exception as e:
            logger.error(f"Pickle serialization failed: {e}")
            raise SerializationError(f"Failed to pickle object: {e}")

    def deserialize(self, data: bytes) -> Any:
        """Deserialize object using pickle"""
        try:
            if self.compression:
                import gzip
                data = gzip.decompress(data)

            return pickle.loads(data)

        except Exception as e:
            logger.error(f"Pickle deserialization failed: {e}")
            raise SerializationError(f"Failed to unpickle object: {e}")

    def serialize_to_file(self, obj: Any, file_path: Union[str, Path]) -> bool:
        """Serialize object to pickle file"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            data = self.serialize(obj)

            with open(file_path, 'wb') as f:
                f.write(data)

            logger.info(f"Successfully pickled object to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to pickle to file {file_path}: {e}")
            return False

    def deserialize_from_file(self, file_path: Union[str, Path]) -> Any:
        """Deserialize object from pickle file"""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, 'rb') as f:
                data = f.read()

            return self.deserialize(data)

        except Exception as e:
            logger.error(f"Failed to unpickle from file {file_path}: {e}")
            raise SerializationError(f"Failed to unpickle from file: {e}")

class HybridSerializer:
    """Hybrid serializer that tries JSON first, falls back to pickle"""

    def __init__(self, config: Optional[SerializationConfig] = None):
        self.config = config or SerializationConfig()
        self.json_serializer = SafeJSONSerializer(self.config)
        self.pickle_serializer = PickleSerializer(self.config.use_compression)

    def serialize(self, obj: Any, prefer_json: bool = True) -> Tuple[bytes, str]:
        """
        Serialize object, returning (data, format)
        format is either 'json' or 'pickle'
        """
        if prefer_json:
            try:
                json_str = self.json_serializer.serialize(obj)
                return json_str.encode('utf-8'), 'json'
            except Exception as e:
                logger.warning(f"JSON serialization failed, falling back to pickle: {e}")

        # Fallback to pickle
        pickle_data = self.pickle_serializer.serialize(obj)
        return pickle_data, 'pickle'

    def deserialize(self, data: bytes, format_hint: str) -> Any:
        """Deserialize data based on format hint"""
        if format_hint == 'json':
            json_str = data.decode('utf-8')
            return self.json_serializer.deserialize(json_str)
        elif format_hint == 'pickle':
            return self.pickle_serializer.deserialize(data)
        else:
            raise ValueError(f"Unknown format: {format_hint}")

    def serialize_to_file(self, obj: Any, file_path: Union[str, Path],
                         prefer_json: bool = True) -> Tuple[bool, str]:
        """
        Serialize to file, returning (success, format_used)
        """
        file_path = Path(file_path)

        if prefer_json:
            # Try JSON with .json extension
            json_path = file_path.with_suffix('.json')
            if self.json_serializer.serialize_to_file(obj, json_path):
                return True, 'json'

        # Fallback to pickle with .pkl extension
        pickle_path = file_path.with_suffix('.pkl')
        success = self.pickle_serializer.serialize_to_file(obj, pickle_path)
        return success, 'pickle' if success else 'failed'

    def deserialize_from_file(self, file_path: Union[str, Path]) -> Any:
        """Deserialize from file, auto-detecting format"""
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.json':
            return self.json_serializer.deserialize_from_file(file_path)
        elif file_path.suffix.lower() in ['.pkl', '.pickle']:
            return self.pickle_serializer.deserialize_from_file(file_path)
        else:
            # Try to auto-detect
            try:
                return self.json_serializer.deserialize_from_file(file_path)
            except Exception:
                return self.pickle_serializer.deserialize_from_file(file_path)

class SerializationError(Exception):
    """Custom exception for serialization errors"""
    pass

# Convenience functions
def safe_json_dumps(obj: Any, config: Optional[SerializationConfig] = None) -> str:
    """Safely serialize object to JSON string"""
    serializer = SafeJSONSerializer(config)
    return serializer.serialize(obj)

def safe_json_loads(json_str: str, config: Optional[SerializationConfig] = None) -> Any:
    """Safely deserialize JSON string to object"""
    serializer = SafeJSONSerializer(config)
    return serializer.deserialize(json_str)

def create_serializer(serialization_type: str = 'hybrid',
                     config: Optional[SerializationConfig] = None):
    """Factory function to create serializers"""
    if serialization_type == 'json':
        return SafeJSONSerializer(config)
    elif serialization_type == 'pickle':
        return PickleSerializer(config.use_compression if config else True)
    elif serialization_type == 'hybrid':
        return HybridSerializer(config)
    else:
        raise ValueError(f"Unknown serialization type: {serialization_type}")

# Testing and validation functions
def test_serialization():
    """Test serialization functionality with various data types"""
    import sys

    # Test data
    test_data = {
        'datetime': datetime.now(),
        'date': date.today(),
        'numpy_array': np.random.randn(10, 10),
        'numpy_scalar': np.float32(3.14),
        'path': Path('/test/path'),
        'set': {1, 2, 3},
        'complex': complex(1, 2),
        'decimal': Decimal('3.14159'),
        'bytes': b'test bytes'
    }

    if TORCH_AVAILABLE:
        test_data['torch_tensor'] = torch.randn(5, 5)
        test_data['torch_tensor_cuda'] = torch.randn(3, 3)

    # Test JSON serialization
    print("Testing JSON serialization...")
    json_serializer = SafeJSONSerializer()

    try:
        json_str = json_serializer.serialize(test_data)
        reconstructed = json_serializer.deserialize(json_str)
        print("✓ JSON serialization successful")
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")

    # Test hybrid serialization
    print("Testing hybrid serialization...")
    hybrid_serializer = HybridSerializer()

    try:
        data, format_used = hybrid_serializer.serialize(test_data)
        reconstructed = hybrid_serializer.deserialize(data, format_used)
        print(f"✓ Hybrid serialization successful (format: {format_used})")
    except Exception as e:
        print(f"✗ Hybrid serialization failed: {e}")

    print("Serialization testing completed!")

if __name__ == "__main__":
    test_serialization()