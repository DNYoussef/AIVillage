import torch
import torch.nn as nn
import torch.jit
import onnx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import os
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
import time
import platform

@dataclass
class DeploymentConfig:
    """Configuration for deployment packaging."""
    target_platforms: List[str] = field(default_factory=lambda: ['cpu', 'cuda'])
    export_formats: List[str] = field(default_factory=lambda: ['pytorch', 'onnx', 'torchscript'])
    optimization_level: str = 'O2'  # O0, O1, O2, O3
    include_metadata: bool = True
    include_validation: bool = True
    include_benchmarks: bool = True
    compression_artifacts: bool = True
    package_format: str = 'zip'  # 'zip', 'tar', 'directory'
    
@dataclass
class DeploymentPackage:
    """Deployment package information."""
    package_path: str
    model_formats: List[str]
    metadata: Dict[str, Any]
    validation_results: Optional[Dict[str, Any]]
    benchmark_results: Optional[Dict[str, Any]]
    deployment_guides: List[str]
    package_size_mb: float
    
class ModelExporter:
    """Handle model export to different formats."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def export_pytorch(self, model: nn.Module, output_path: str, 
                      sample_input: torch.Tensor, metadata: Dict[str, Any]) -> str:
        """Export model in PyTorch format."""
        try:
            # Save model state dict
            model_path = os.path.join(output_path, 'model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_architecture': str(model),
                'metadata': metadata
            }, model_path)
            
            # Save complete model
            complete_model_path = os.path.join(output_path, 'model_complete.pth')
            torch.save(model, complete_model_path)
            
            self.logger.info(f"PyTorch model exported to {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"PyTorch export failed: {e}")
            raise
            
    def export_torchscript(self, model: nn.Module, output_path: str, 
                          sample_input: torch.Tensor, metadata: Dict[str, Any]) -> str:
        """Export model as TorchScript."""
        try:
            model.eval()
            
            # Try tracing first
            try:
                traced_model = torch.jit.trace(model, sample_input)
                script_path = os.path.join(output_path, 'model_traced.pt')
                traced_model.save(script_path)
                self.logger.info(f"TorchScript traced model exported to {script_path}")
                
            except Exception as trace_error:
                self.logger.warning(f"Tracing failed: {trace_error}, trying scripting")
                
                # Fall back to scripting
                scripted_model = torch.jit.script(model)
                script_path = os.path.join(output_path, 'model_scripted.pt')
                scripted_model.save(script_path)
                self.logger.info(f"TorchScript scripted model exported to {script_path}")
                
            # Save metadata
            metadata_path = os.path.join(output_path, 'torchscript_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return script_path
            
        except Exception as e:
            self.logger.error(f"TorchScript export failed: {e}")
            raise
            
    def export_onnx(self, model: nn.Module, output_path: str, 
                   sample_input: torch.Tensor, metadata: Dict[str, Any]) -> str:
        """Export model to ONNX format."""
        try:
            model.eval()
            
            onnx_path = os.path.join(output_path, 'model.onnx')
            
            # Dynamic axes for batch size
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
            
            # Export to ONNX
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            
            # Verify ONNX model
            try:
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                self.logger.info(f"ONNX model verified and exported to {onnx_path}")
            except Exception as verify_error:
                self.logger.warning(f"ONNX verification failed: {verify_error}")
                
            # Save metadata
            metadata_path = os.path.join(output_path, 'onnx_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return onnx_path
            
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            raise
            
    def export_tflite(self, model: nn.Module, output_path: str, 
                     sample_input: torch.Tensor, metadata: Dict[str, Any]) -> str:
        """Export model to TensorFlow Lite format (via ONNX)."""
        try:
            # First export to ONNX
            onnx_path = self.export_onnx(model, output_path, sample_input, metadata)
            
            # Convert ONNX to TensorFlow Lite (requires additional dependencies)
            try:
                import onnx2tf
                import tensorflow as tf
                
                # Convert ONNX to TensorFlow
                tf_model_path = os.path.join(output_path, 'tf_model')
                onnx2tf.convert(
                    input_onnx_file_path=onnx_path,
                    output_folder_path=tf_model_path
                )
                
                # Convert TensorFlow to TensorFlow Lite
                converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                
                tflite_path = os.path.join(output_path, 'model.tflite')
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                    
                self.logger.info(f"TensorFlow Lite model exported to {tflite_path}")
                return tflite_path
                
            except ImportError:
                self.logger.warning("TensorFlow Lite conversion requires onnx2tf and tensorflow")
                return onnx_path
                
        except Exception as e:
            self.logger.error(f"TensorFlow Lite export failed: {e}")
            raise
            
class BenchmarkGenerator:
    """Generate deployment benchmarks."""
    
    def __init__(self, device: torch.device, logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
    def generate_benchmarks(self, models: Dict[str, nn.Module], 
                          sample_input: torch.Tensor,
                          num_runs: int = 100) -> Dict[str, Dict[str, float]]:
        """Generate comprehensive benchmarks for different model formats."""
        try:
            benchmarks = {}
            
            for model_name, model in models.items():
                self.logger.info(f"Benchmarking {model_name}")
                
                model_benchmarks = {
                    'inference_time_ms': self._benchmark_inference_time(model, sample_input, num_runs),
                    'throughput_samples_per_sec': self._benchmark_throughput(model, sample_input),
                    'memory_usage_mb': self._benchmark_memory_usage(model, sample_input),
                    'model_size_mb': self._get_model_size(model),
                    'cpu_utilization': self._benchmark_cpu_utilization(model, sample_input),
                    'gpu_utilization': self._benchmark_gpu_utilization(model, sample_input) if self.device.type == 'cuda' else 0.0
                }
                
                benchmarks[model_name] = model_benchmarks
                
            return benchmarks
            
        except Exception as e:
            self.logger.error(f"Benchmark generation failed: {e}")
            raise
            
    def _benchmark_inference_time(self, model: nn.Module, 
                                 sample_input: torch.Tensor, 
                                 num_runs: int) -> float:
        """Benchmark inference time."""
        model.eval()
        model.to(self.device)
        sample_input = sample_input.to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                model(sample_input)
                
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                model(sample_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
                
        return float(np.mean(times))
        
    def _benchmark_throughput(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """Benchmark throughput."""
        model.eval()
        model.to(self.device)
        sample_input = sample_input.to(self.device)
        
        # Measure throughput for 5 seconds
        total_samples = 0
        start_time = time.time()
        
        with torch.no_grad():
            while time.time() - start_time < 5.0:
                model(sample_input)
                total_samples += sample_input.size(0)
                
        elapsed_time = time.time() - start_time
        return total_samples / elapsed_time
        
    def _benchmark_memory_usage(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """Benchmark memory usage."""
        model.to(self.device)
        sample_input = sample_input.to(self.device)
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                model(sample_input)
                
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            return float(peak_memory)
        else:
            # Estimate CPU memory usage
            model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            activation_memory = sample_input.numel() * sample_input.element_size() / (1024 * 1024)
            return model_memory + activation_memory * 2  # Rough estimate
            
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size / (1024 * 1024)
        
    def _benchmark_cpu_utilization(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """Benchmark CPU utilization."""
        try:
            import psutil
            
            model.eval()
            model.to(torch.device('cpu'))
            sample_input = sample_input.to(torch.device('cpu'))
            
            # Monitor CPU during inference
            cpu_percentages = []
            
            def monitor_cpu():
                for _ in range(10):
                    cpu_percentages.append(psutil.cpu_percent(interval=0.1))
                    
            import threading
            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()
            
            # Run inference
            with torch.no_grad():
                for _ in range(10):
                    model(sample_input)
                    
            monitor_thread.join()
            
            return float(np.mean(cpu_percentages))
            
        except ImportError:
            self.logger.warning("psutil not available for CPU monitoring")
            return 0.0
            
    def _benchmark_gpu_utilization(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """Benchmark GPU utilization."""
        if self.device.type != 'cuda':
            return 0.0
            
        try:
            import pynvml
            
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            model.eval()
            model.to(self.device)
            sample_input = sample_input.to(self.device)
            
            # Monitor GPU during inference
            gpu_percentages = []
            
            def monitor_gpu():
                for _ in range(10):
                    info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percentages.append(info.gpu)
                    time.sleep(0.1)
                    
            import threading
            monitor_thread = threading.Thread(target=monitor_gpu)
            monitor_thread.start()
            
            # Run inference
            with torch.no_grad():
                for _ in range(10):
                    model(sample_input)
                    
            monitor_thread.join()
            
            return float(np.mean(gpu_percentages))
            
        except ImportError:
            self.logger.warning("pynvml not available for GPU monitoring")
            return 0.0
            
class DocumentationGenerator:
    """Generate deployment documentation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def generate_deployment_guide(self, output_path: str, 
                                metadata: Dict[str, Any],
                                formats: List[str]) -> str:
        """Generate deployment guide."""
        try:
            guide_content = self._create_deployment_guide_content(metadata, formats)
            
            guide_path = os.path.join(output_path, 'DEPLOYMENT_GUIDE.md')
            with open(guide_path, 'w') as f:
                f.write(guide_content)
                
            self.logger.info(f"Deployment guide generated: {guide_path}")
            return guide_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate deployment guide: {e}")
            raise
            
    def _create_deployment_guide_content(self, metadata: Dict[str, Any], formats: List[str]) -> str:
        """Create deployment guide content."""
        content = f"""# Model Deployment Guide

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
Platform: {platform.system()} {platform.release()}

## Model Information

- **Model Name**: {metadata.get('model_name', 'Unknown')}
- **Model Type**: {metadata.get('model_type', 'Unknown')}
- **Input Shape**: {metadata.get('input_shape', 'Unknown')}
- **Output Shape**: {metadata.get('output_shape', 'Unknown')}
- **Parameters**: {metadata.get('total_parameters', 'Unknown'):,}
- **Model Size**: {metadata.get('model_size_mb', 'Unknown'):.2f} MB

## Available Formats

"""
        
        for fmt in formats:
            content += f"- **{fmt.upper()}**: `model.{self._get_file_extension(fmt)}`\n"
            
        content += f"""

## Quick Start

### PyTorch

```python
import torch

# Load the model
model = torch.load('model_complete.pth')
model.eval()

# Or load state dict
model = YourModelClass()
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

### TorchScript

```python
import torch

# Load traced/scripted model
model = torch.jit.load('model_traced.pt')
# or model = torch.jit.load('model_scripted.pt')

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

### ONNX

```python
import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession('model.onnx')

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Inference
result = session.run([output_name], {{input_name: input_array}})
```

## Performance Considerations

1. **Batch Size**: Optimal batch size is {metadata.get('optimal_batch_size', 'TBD')}
2. **Memory Requirements**: ~{metadata.get('memory_requirements_mb', 'TBD')} MB
3. **Expected Latency**: ~{metadata.get('expected_latency_ms', 'TBD')} ms per sample

## Deployment Platforms

### CPU Deployment
- Use PyTorch or ONNX Runtime for CPU inference
- Consider Intel MKL-DNN for optimized CPU performance
- Thread count optimization may be needed

### GPU Deployment
- Use PyTorch with CUDA or TensorRT for GPU inference
- Ensure CUDA compatibility: {metadata.get('cuda_version', 'Check requirements')}
- Consider mixed precision for better performance

### Mobile Deployment
- Use TorchScript for iOS deployment
- Use ONNX or TensorFlow Lite for Android deployment
- Model quantization recommended for mobile

### Edge Deployment
- TensorFlow Lite for edge devices
- ONNX Runtime for IoT devices
- Consider model pruning for resource-constrained devices

## Troubleshooting

### Common Issues

1. **Input Shape Mismatch**
   - Ensure input tensor shape matches: {metadata.get('input_shape', 'Check model specs')}
   - Check batch dimension handling

2. **Device Compatibility**
   - Verify CUDA availability for GPU inference
   - Check device memory requirements

3. **Performance Issues**
   - Use appropriate batch sizes
   - Enable optimizations (JIT, quantization)
   - Consider model compilation (TorchScript)

### Support

For additional support or questions, refer to the model documentation or contact the development team.
"""
        
        return content
        
    def _get_file_extension(self, format_name: str) -> str:
        """Get file extension for format."""
        extensions = {
            'pytorch': 'pth',
            'torchscript': 'pt',
            'onnx': 'onnx',
            'tflite': 'tflite'
        }
        return extensions.get(format_name.lower(), 'bin')
        
    def generate_api_documentation(self, output_path: str, 
                                 metadata: Dict[str, Any]) -> str:
        """Generate API documentation."""
        try:
            api_content = self._create_api_documentation_content(metadata)
            
            api_path = os.path.join(output_path, 'API_REFERENCE.md')
            with open(api_path, 'w') as f:
                f.write(api_content)
                
            return api_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate API documentation: {e}")
            raise
            
    def _create_api_documentation_content(self, metadata: Dict[str, Any]) -> str:
        """Create API documentation content."""
        return f"""# API Reference

## Model Interface

### Input Specification

- **Shape**: {metadata.get('input_shape', 'Unknown')}
- **Data Type**: {metadata.get('input_dtype', 'float32')}
- **Range**: {metadata.get('input_range', 'Normalized [0, 1] or [-1, 1]')}
- **Preprocessing**: {metadata.get('preprocessing', 'See preprocessing section')}

### Output Specification

- **Shape**: {metadata.get('output_shape', 'Unknown')}
- **Data Type**: {metadata.get('output_dtype', 'float32')}
- **Interpretation**: {metadata.get('output_interpretation', 'Raw logits or probabilities')}
- **Postprocessing**: {metadata.get('postprocessing', 'Apply softmax for probabilities')}

### Model Methods

#### `forward(input_tensor)`

Performs forward inference on the input tensor.

**Parameters:**
- `input_tensor` (Tensor): Input tensor with shape {metadata.get('input_shape', 'Unknown')}

**Returns:**
- `output_tensor` (Tensor): Output tensor with shape {metadata.get('output_shape', 'Unknown')}

**Example:**
```python
output = model(input_tensor)
```

## Preprocessing

{metadata.get('preprocessing_details', 'Define preprocessing steps here')}

## Postprocessing

{metadata.get('postprocessing_details', 'Define postprocessing steps here')}

## Performance Characteristics

- **Inference Time**: ~{metadata.get('inference_time_ms', 'TBD')} ms
- **Memory Usage**: ~{metadata.get('memory_usage_mb', 'TBD')} MB
- **Throughput**: ~{metadata.get('throughput_samples_per_sec', 'TBD')} samples/sec
"""
        
class DeploymentPackager:
    """Main deployment packaging agent."""
    
    def __init__(self, device: torch.device = torch.device('cpu'), logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.exporter = ModelExporter(logger)
        self.benchmark_generator = BenchmarkGenerator(device, logger)
        self.doc_generator = DocumentationGenerator(logger)
        
    def create_deployment_package(self, model: nn.Module,
                                sample_input: torch.Tensor,
                                output_dir: str,
                                config: DeploymentConfig,
                                metadata: Optional[Dict[str, Any]] = None) -> DeploymentPackage:
        """Create comprehensive deployment package."""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare metadata
            if metadata is None:
                metadata = self._generate_default_metadata(model, sample_input)
                
            # Export model in different formats
            exported_formats = []
            model_files = {}
            
            for export_format in config.export_formats:
                try:
                    if export_format == 'pytorch':
                        path = self.exporter.export_pytorch(model, output_dir, sample_input, metadata)
                        exported_formats.append('pytorch')
                        model_files['pytorch'] = path
                        
                    elif export_format == 'torchscript':
                        path = self.exporter.export_torchscript(model, output_dir, sample_input, metadata)
                        exported_formats.append('torchscript')
                        model_files['torchscript'] = path
                        
                    elif export_format == 'onnx':
                        path = self.exporter.export_onnx(model, output_dir, sample_input, metadata)
                        exported_formats.append('onnx')
                        model_files['onnx'] = path
                        
                    elif export_format == 'tflite':
                        path = self.exporter.export_tflite(model, output_dir, sample_input, metadata)
                        exported_formats.append('tflite')
                        model_files['tflite'] = path
                        
                except Exception as e:
                    self.logger.warning(f"Failed to export {export_format}: {e}")
                    
            # Generate benchmarks
            benchmark_results = None
            if config.include_benchmarks:
                try:
                    models_to_benchmark = {'original': model}
                    benchmark_results = self.benchmark_generator.generate_benchmarks(
                        models_to_benchmark, sample_input
                    )
                    
                    # Save benchmarks
                    benchmark_path = os.path.join(output_dir, 'benchmarks.json')
                    with open(benchmark_path, 'w') as f:
                        json.dump(benchmark_results, f, indent=2)
                        
                except Exception as e:
                    self.logger.warning(f"Benchmark generation failed: {e}")
                    
            # Generate documentation
            deployment_guides = []
            if config.include_metadata:
                try:
                    guide_path = self.doc_generator.generate_deployment_guide(
                        output_dir, metadata, exported_formats
                    )
                    deployment_guides.append(guide_path)
                    
                    api_path = self.doc_generator.generate_api_documentation(
                        output_dir, metadata
                    )
                    deployment_guides.append(api_path)
                    
                except Exception as e:
                    self.logger.warning(f"Documentation generation failed: {e}")
                    
            # Save comprehensive metadata
            if config.include_metadata:
                metadata_path = os.path.join(output_dir, 'metadata.json')
                comprehensive_metadata = {
                    'model_metadata': metadata,
                    'deployment_info': {
                        'formats': exported_formats,
                        'platforms': config.target_platforms,
                        'optimization_level': config.optimization_level,
                        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'python_version': platform.python_version(),
                        'pytorch_version': torch.__version__
                    },
                    'file_manifest': model_files
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(comprehensive_metadata, f, indent=2)
                    
            # Create package
            package_path = output_dir
            if config.package_format == 'zip':
                package_path = self._create_zip_package(output_dir)
            elif config.package_format == 'tar':
                package_path = self._create_tar_package(output_dir)
                
            # Calculate package size
            package_size = self._calculate_package_size(package_path)
            
            # Validation results placeholder
            validation_results = None
            if config.include_validation:
                validation_results = {'status': 'Package created successfully'}
                
            package = DeploymentPackage(
                package_path=package_path,
                model_formats=exported_formats,
                metadata=metadata,
                validation_results=validation_results,
                benchmark_results=benchmark_results,
                deployment_guides=deployment_guides,
                package_size_mb=package_size
            )
            
            self.logger.info(f"Deployment package created: {package_path}")
            self.logger.info(f"Package size: {package_size:.2f} MB")
            self.logger.info(f"Included formats: {', '.join(exported_formats)}")
            
            return package
            
        except Exception as e:
            self.logger.error(f"Deployment packaging failed: {e}")
            raise
            
    def _generate_default_metadata(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Generate default metadata for the model."""
        model.eval()
        
        with torch.no_grad():
            sample_output = model(sample_input)
            
        return {
            'model_name': model.__class__.__name__,
            'model_type': 'Neural Network',
            'input_shape': list(sample_input.shape),
            'output_shape': list(sample_output.shape) if hasattr(sample_output, 'shape') else 'Variable',
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
            'input_dtype': str(sample_input.dtype),
            'output_dtype': str(sample_output.dtype) if hasattr(sample_output, 'dtype') else 'unknown',
            'pytorch_version': torch.__version__,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    def _create_zip_package(self, output_dir: str) -> str:
        """Create ZIP package."""
        zip_path = f"{output_dir}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
                    
        return zip_path
        
    def _create_tar_package(self, output_dir: str) -> str:
        """Create TAR package."""
        import tarfile
        
        tar_path = f"{output_dir}.tar.gz"
        
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(output_dir, arcname=os.path.basename(output_dir))
            
        return tar_path
        
    def _calculate_package_size(self, package_path: str) -> float:
        """Calculate package size in MB."""
        if os.path.isfile(package_path):
            return os.path.getsize(package_path) / (1024 * 1024)
        elif os.path.isdir(package_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(package_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)
        else:
            return 0.0
            
    def create_config(self, **kwargs) -> DeploymentConfig:
        """Create deployment configuration."""
        return DeploymentConfig(**kwargs)
        
    def validate_package(self, package: DeploymentPackage) -> Dict[str, bool]:
        """Validate deployment package."""
        validation_results = {
            'package_exists': os.path.exists(package.package_path),
            'has_models': len(package.model_formats) > 0,
            'has_metadata': package.metadata is not None,
            'has_documentation': len(package.deployment_guides) > 0,
            'reasonable_size': package.package_size_mb < 1000  # Less than 1GB
        }
        
        validation_results['overall_valid'] = all(validation_results.values())
        
        return validation_results
        
    def extract_package(self, package_path: str, extract_dir: str) -> str:
        """Extract deployment package."""
        try:
            os.makedirs(extract_dir, exist_ok=True)
            
            if package_path.endswith('.zip'):
                with zipfile.ZipFile(package_path, 'r') as zipf:
                    zipf.extractall(extract_dir)
            elif package_path.endswith('.tar.gz') or package_path.endswith('.tar'):
                import tarfile
                with tarfile.open(package_path, 'r:*') as tar:
                    tar.extractall(extract_dir)
            else:
                # Assume directory
                shutil.copytree(package_path, extract_dir)
                
            self.logger.info(f"Package extracted to {extract_dir}")
            return extract_dir
            
        except Exception as e:
            self.logger.error(f"Package extraction failed: {e}")
            raise
            
    def load_model_from_package(self, package_dir: str, 
                              format_preference: List[str] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load model from deployment package."""
        try:
            # Load metadata
            metadata_path = os.path.join(package_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Determine format to load
            if format_preference is None:
                format_preference = ['pytorch', 'torchscript', 'onnx']
                
            model = None
            for fmt in format_preference:
                if fmt == 'pytorch':
                    model_path = os.path.join(package_dir, 'model_complete.pth')
                    if os.path.exists(model_path):
                        model = torch.load(model_path)
                        break
                elif fmt == 'torchscript':
                    for script_file in ['model_traced.pt', 'model_scripted.pt']:
                        model_path = os.path.join(package_dir, script_file)
                        if os.path.exists(model_path):
                            model = torch.jit.load(model_path)
                            break
                    if model is not None:
                        break
                        
            if model is None:
                raise ValueError("No compatible model format found in package")
                
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load model from package: {e}")
            raise
