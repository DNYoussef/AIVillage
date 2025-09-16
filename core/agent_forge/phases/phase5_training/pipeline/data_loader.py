"""
Agent Forge Phase 5 - Advanced Data Loading System
Efficient multi-format data loading with streaming support
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from typing import Dict, List, Any, Optional, Iterator, Union
import json
import pickle
import h5py
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import mmap
import os
from dataclasses import dataclass
from enum import Enum

class DataFormat(Enum):
    TEXT = "text"
    TOKENIZED = "tokenized"
    PREPROCESSED = "preprocessed"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    JSON = "json"
    BINARY = "binary"

@dataclass
class DataConfig:
    """Data loading configuration"""
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    max_sequence_length: int = 2048
    cache_size: int = 1000
    streaming: bool = False
    validation_split: float = 0.1
    quality_threshold: float = 0.8

class QualityValidator:
    """Data quality validation and filtering"""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'filtered_samples': 0
        }

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate individual data sample"""
        try:
            # Check for required fields
            if 'input' not in sample:
                return False

            # Check data quality metrics
            input_data = sample['input']

            # Length validation
            if isinstance(input_data, str) and len(input_data) < 10:
                return False

            if isinstance(input_data, (list, torch.Tensor)) and len(input_data) < 5:
                return False

            # Content quality checks
            if isinstance(input_data, str):
                # Check for minimum word count
                words = input_data.split()
                if len(words) < 3:
                    return False

                # Check for non-ASCII ratio
                ascii_chars = sum(1 for c in input_data if ord(c) < 128)
                if ascii_chars / len(input_data) < 0.8:
                    return False

            # Token sequence validation
            if isinstance(input_data, (list, torch.Tensor)):
                # Check for valid token range
                if torch.is_tensor(input_data):
                    if input_data.min() < 0 or input_data.max() > 50000:
                        return False
                elif isinstance(input_data, list):
                    if any(token < 0 or token > 50000 for token in input_data):
                        return False

            self.stats['valid_samples'] += 1
            return True

        except Exception as e:
            logging.warning(f"Sample validation error: {e}")
            return False
        finally:
            self.stats['total_samples'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self.stats['total_samples']
        if total > 0:
            self.stats['quality_ratio'] = self.stats['valid_samples'] / total
            self.stats['filter_ratio'] = self.stats['filtered_samples'] / total
        return self.stats.copy()

class CachedDataset(Dataset):
    """High-performance cached dataset with LRU eviction"""

    def __init__(self, data_path: str, config: DataConfig):
        self.data_path = Path(data_path)
        self.config = config
        self.cache = {}
        self.access_order = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.validator = QualityValidator(config.quality_threshold)

        # Load metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load dataset metadata"""
        metadata_path = self.data_path.with_suffix('.metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = self._create_metadata()

        self.length = self.metadata.get('length', 0)

    def _create_metadata(self) -> Dict[str, Any]:
        """Create metadata for dataset"""
        metadata = {
            'format': self._detect_format(),
            'length': self._calculate_length(),
            'sample_shape': None,
            'dtype': None
        }

        # Save metadata
        metadata_path = self.data_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _detect_format(self) -> DataFormat:
        """Auto-detect data format"""
        suffix = self.data_path.suffix.lower()
        if suffix == '.json':
            return DataFormat.JSON
        elif suffix == '.pkl':
            return DataFormat.PICKLE
        elif suffix in ['.h5', '.hdf5']:
            return DataFormat.HDF5
        elif suffix == '.txt':
            return DataFormat.TEXT
        else:
            return DataFormat.BINARY

    def _calculate_length(self) -> int:
        """Calculate dataset length"""
        format_type = self._detect_format()

        if format_type == DataFormat.JSON:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                return len(data) if isinstance(data, list) else 1
        elif format_type == DataFormat.HDF5:
            with h5py.File(self.data_path, 'r') as f:
                # Assume first key contains the data
                first_key = list(f.keys())[0]
                return len(f[first_key])
        elif format_type == DataFormat.PICKLE:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
                return len(data) if hasattr(data, '__len__') else 1
        else:
            # Estimate based on file size and average sample size
            file_size = self.data_path.stat().st_size
            return max(1, file_size // 1024)  # Rough estimate

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """Load individual sample from storage"""
        format_type = self.metadata['format']

        try:
            if format_type == DataFormat.JSON:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data[idx]
                    else:
                        return data

            elif format_type == DataFormat.HDF5:
                with h5py.File(self.data_path, 'r') as f:
                    first_key = list(f.keys())[0]
                    sample_data = f[first_key][idx]
                    return {'input': sample_data}

            elif format_type == DataFormat.PICKLE:
                with open(self.data_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        return data[idx]
                    else:
                        return data

            else:
                # Handle binary/text formats
                return self._load_binary_sample(idx)

        except Exception as e:
            logging.error(f"Error loading sample {idx}: {e}")
            return {'input': torch.zeros(self.config.max_sequence_length)}

    def _load_binary_sample(self, idx: int) -> Dict[str, Any]:
        """Load sample from binary format"""
        # Implement binary loading logic based on your format
        return {'input': torch.randn(self.config.max_sequence_length)}

    def _manage_cache(self):
        """Manage LRU cache"""
        if len(self.cache) >= self.config.cache_size:
            # Remove oldest accessed item
            oldest_idx = self.access_order.pop(0)
            del self.cache[oldest_idx]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Check cache
        if idx in self.cache:
            self.cache_hits += 1
            # Update access order
            self.access_order.remove(idx)
            self.access_order.append(idx)
            sample = self.cache[idx]
        else:
            self.cache_misses += 1
            # Load from storage
            sample = self._load_sample(idx)

            # Validate sample quality
            if not self.validator.validate_sample(sample):
                # Return a default/skip sample
                sample = {'input': torch.zeros(self.config.max_sequence_length)}

            # Cache management
            self._manage_cache()
            self.cache[idx] = sample
            self.access_order.append(idx)

        return sample

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0

        return {
            'hit_rate': hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self.cache),
            'max_cache_size': self.config.cache_size
        }

class StreamingDataset(IterableDataset):
    """Memory-efficient streaming dataset for large data"""

    def __init__(self, data_path: str, config: DataConfig):
        self.data_path = Path(data_path)
        self.config = config
        self.validator = QualityValidator(config.quality_threshold)

    def _stream_json(self) -> Iterator[Dict[str, Any]]:
        """Stream JSON data line by line"""
        with open(self.data_path, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    if self.validator.validate_sample(sample):
                        yield sample
                except json.JSONDecodeError:
                    continue

    def _stream_hdf5(self) -> Iterator[Dict[str, Any]]:
        """Stream HDF5 data"""
        with h5py.File(self.data_path, 'r') as f:
            first_key = list(f.keys())[0]
            dataset = f[first_key]

            for i in range(len(dataset)):
                sample = {'input': dataset[i]}
                if self.validator.validate_sample(sample):
                    yield sample

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over streaming data"""
        format_type = self._detect_format()

        if format_type == DataFormat.JSON:
            yield from self._stream_json()
        elif format_type == DataFormat.HDF5:
            yield from self._stream_hdf5()
        else:
            # Fallback to chunked reading
            yield from self._stream_binary()

    def _detect_format(self) -> DataFormat:
        """Auto-detect data format"""
        suffix = self.data_path.suffix.lower()
        if suffix == '.jsonl':
            return DataFormat.JSON
        elif suffix in ['.h5', '.hdf5']:
            return DataFormat.HDF5
        else:
            return DataFormat.BINARY

    def _stream_binary(self) -> Iterator[Dict[str, Any]]:
        """Stream binary data in chunks"""
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(self.data_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                # Process chunk into samples
                # This is format-specific implementation
                sample = {'input': torch.from_numpy(np.frombuffer(chunk, dtype=np.uint8))}
                if self.validator.validate_sample(sample):
                    yield sample

class DataLoaderFactory:
    """Factory for creating optimized data loaders"""

    @staticmethod
    def create_loader(
        data_path: str,
        config: DataConfig,
        dataset_type: str = "cached"
    ) -> DataLoader:
        """Create optimized data loader"""

        # Choose dataset type
        if dataset_type == "streaming" or config.streaming:
            dataset = StreamingDataset(data_path, config)
        else:
            dataset = CachedDataset(data_path, config)

        # Custom collate function for variable-length sequences
        def collate_fn(batch):
            """Custom collate function with padding"""
            if not batch:
                return {}

            # Handle different input types
            inputs = [item['input'] for item in batch]

            if isinstance(inputs[0], str):
                # Text data - return as list
                return {'input': inputs}
            elif isinstance(inputs[0], torch.Tensor):
                # Tensor data - pad to max length
                max_len = min(max(len(seq) for seq in inputs), config.max_sequence_length)
                padded = torch.zeros(len(inputs), max_len)
                for i, seq in enumerate(inputs):
                    length = min(len(seq), max_len)
                    padded[i, :length] = seq[:length]
                return {'input': padded}
            else:
                # List data - convert to tensor and pad
                max_len = min(max(len(seq) for seq in inputs), config.max_sequence_length)
                padded = torch.zeros(len(inputs), max_len, dtype=torch.long)
                for i, seq in enumerate(inputs):
                    length = min(len(seq), max_len)
                    padded[i, :length] = torch.tensor(seq[:length])
                return {'input': padded}

        # Create data loader with optimizations
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle and not config.streaming,
            num_workers=config.num_workers,
            drop_last=config.drop_last,
            prefetch_factor=config.prefetch_factor,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers and config.num_workers > 0,
            collate_fn=collate_fn
        )

        return loader

    @staticmethod
    def create_validation_split(
        data_path: str,
        config: DataConfig
    ) -> tuple[DataLoader, DataLoader]:
        """Create train/validation split"""

        # Load full dataset
        dataset = CachedDataset(data_path, config)

        # Calculate split sizes
        total_size = len(dataset)
        val_size = int(total_size * config.validation_split)
        train_size = total_size - val_size

        # Create random split
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create loaders
        train_config = DataConfig(**{
            **config.__dict__,
            'shuffle': True,
            'drop_last': True
        })

        val_config = DataConfig(**{
            **config.__dict__,
            'shuffle': False,
            'drop_last': False
        })

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=train_config.shuffle,
            num_workers=train_config.num_workers,
            drop_last=train_config.drop_last,
            pin_memory=train_config.pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_config.batch_size,
            shuffle=val_config.shuffle,
            num_workers=val_config.num_workers,
            drop_last=val_config.drop_last,
            pin_memory=val_config.pin_memory
        )

        return train_loader, val_loader

class DataLoadingProfiler:
    """Profiler for data loading performance"""

    def __init__(self):
        self.stats = {
            'batches_loaded': 0,
            'total_time': 0.0,
            'avg_batch_time': 0.0,
            'samples_per_second': 0.0,
            'cache_stats': {},
            'quality_stats': {}
        }
        self.start_time = None

    def start_profiling(self):
        """Start profiling session"""
        import time
        self.start_time = time.time()
        self.stats = {key: 0.0 if isinstance(val, (int, float)) else {}
                     for key, val in self.stats.items()}

    def record_batch(self, batch_size: int, load_time: float):
        """Record batch loading statistics"""
        self.stats['batches_loaded'] += 1
        self.stats['total_time'] += load_time
        self.stats['avg_batch_time'] = self.stats['total_time'] / self.stats['batches_loaded']
        self.stats['samples_per_second'] = batch_size / load_time if load_time > 0 else 0

    def update_cache_stats(self, cache_stats: Dict[str, Any]):
        """Update cache statistics"""
        self.stats['cache_stats'] = cache_stats

    def update_quality_stats(self, quality_stats: Dict[str, Any]):
        """Update data quality statistics"""
        self.stats['quality_stats'] = quality_stats

    def get_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        import time
        if self.start_time:
            total_elapsed = time.time() - self.start_time
            self.stats['total_elapsed'] = total_elapsed

            if self.stats['batches_loaded'] > 0:
                self.stats['batches_per_second'] = self.stats['batches_loaded'] / total_elapsed

        return self.stats.copy()

def create_optimized_loader(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    streaming: bool = False,
    **kwargs
) -> DataLoader:
    """Convenience function to create optimized data loader"""

    config = DataConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        streaming=streaming,
        **kwargs
    )

    return DataLoaderFactory.create_loader(data_path, config)

if __name__ == "__main__":
    # Example usage and testing
    import tempfile
    import json

    # Create test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = [
            {'input': [1, 2, 3, 4, 5] * 100},
            {'input': [6, 7, 8, 9, 10] * 100},
            {'input': [11, 12, 13, 14, 15] * 100}
        ] * 100
        json.dump(test_data, f)
        test_file = f.name

    # Test data loader
    config = DataConfig(
        batch_size=16,
        num_workers=2,
        cache_size=50,
        validation_split=0.2
    )

    train_loader, val_loader = DataLoaderFactory.create_validation_split(test_file, config)

    # Profile performance
    profiler = DataLoadingProfiler()
    profiler.start_profiling()

    import time
    for i, batch in enumerate(train_loader):
        start_time = time.time()
        # Simulate processing time
        time.sleep(0.01)
        load_time = time.time() - start_time

        profiler.record_batch(len(batch['input']), load_time)

        if i >= 10:  # Test first 10 batches
            break

    # Print results
    report = profiler.get_report()
    print("Data Loading Performance Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    # Cleanup
    os.unlink(test_file)