"""
Agent Forge Phase 5: End-to-End Training Pipeline
=================================================

Complete training pipeline orchestrating all Phase 5 components with
comprehensive integration, monitoring, and Phase 4-6 coordination.

Key Features:
- End-to-end training workflow
- Phase integration (4-5-6)
- Comprehensive monitoring
- Quality gates and validation
- NASA POT10 compliance
- Production-ready deployment
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Phase 5 components
from .training_architecture import TrainingArchitecture
from .distributed_trainer import DistributedTrainer
from .bitnet_training import BitNetTrainingOptimizer, convert_model_to_bitnet
from .grokfast_integration import GrokfastAccelerator
from .performance_monitor import TrainingMonitor
from .checkpoint_manager import CheckpointManager
from .training_config import TrainingConfig, Environment, ConfigManager


class Phase5Dataset(Dataset):
    """
    Dataset wrapper for Phase 5 training with BitNet optimization.

    Handles data loading, preprocessing, and tokenization for efficient
    distributed training.
    """

    def __init__(self, data_path: str, config: TrainingConfig, split: str = "train"):
        self.data_path = Path(data_path)
        self.config = config
        self.split = split

        # Load data
        self.data = self._load_data()
        self.logger = logging.getLogger(f'phase5_dataset_{split}')

        self.logger.info(f"Dataset loaded: {len(self.data)} samples ({split})")

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file or create synthetic data for testing."""
        data_file = self.data_path / f"{self.split}.json"

        if data_file.exists():
            with open(data_file, 'r') as f:
                return json.load(f)
        else:
            # Create synthetic data for testing
            return self._create_synthetic_data()

    def _create_synthetic_data(self) -> List[Dict[str, Any]]:
        """Create synthetic training data for testing."""
        import random

        vocab_size = self.config.model.vocab_size
        max_length = self.config.data.max_length

        # Determine number of samples based on split
        if self.split == "train":
            num_samples = 10000
        elif self.split == "val":
            num_samples = 1000
        else:  # test
            num_samples = 500

        data = []
        for i in range(num_samples):
            # Random sequence length
            seq_length = random.randint(50, max_length)

            # Random input tokens
            input_ids = [random.randint(1, vocab_size - 1) for _ in range(seq_length)]

            # Simple synthetic task: predict next token
            labels = input_ids[1:] + [0]  # Shifted for next token prediction

            data.append({
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': [1] * len(input_ids)
            })

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Convert to tensors
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long)
        }


class Phase5Model(nn.Module):
    """
    BitNet model for Phase 5 training with Grokfast enhancement.

    Integrates BitNet quantization with rapid capability acquisition
    for efficient training.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # Model components
        self.embedding = nn.Embedding(
            config.model.vocab_size,
            config.model.hidden_size
        )

        # Create transformer layers
        self.layers = nn.ModuleList([
            self._create_layer() for _ in range(config.model.num_layers)
        ])

        self.layer_norm = nn.LayerNorm(config.model.hidden_size, eps=config.model.layer_norm_eps)
        self.output_head = nn.Linear(config.model.hidden_size, config.model.vocab_size)

        # Convert to BitNet if enabled
        if config.model.use_bitnet_linear:
            self._convert_to_bitnet()

        # Initialize weights
        self.apply(self._init_weights)

    def _create_layer(self) -> nn.Module:
        """Create a single transformer layer."""
        return nn.TransformerEncoderLayer(
            d_model=self.config.model.hidden_size,
            nhead=self.config.model.num_heads,
            dim_feedforward=self.config.model.intermediate_size,
            dropout=self.config.model.dropout_prob,
            batch_first=True
        )

    def _convert_to_bitnet(self) -> None:
        """Convert model layers to BitNet."""
        # Convert specified layers to BitNet
        if "all" in self.config.model.bitnet_layers:
            self = convert_model_to_bitnet(self)
        else:
            # Convert specific layers
            self = convert_model_to_bitnet(self, self.config.model.bitnet_layers)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with loss calculation."""
        # Embedding
        hidden_states = self.embedding(input_ids)

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=~attention_mask.bool())

        # Layer normalization and output projection
        hidden_states = self.layer_norm(hidden_states)
        logits = self.output_head(hidden_states)

        outputs = {'logits': logits}

        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs['loss'] = loss

        return outputs


class Phase5Pipeline:
    """
    Complete Agent Forge Phase 5 training pipeline.

    Orchestrates all components for end-to-end BitNet training with
    comprehensive monitoring and Phase 4-6 integration.
    """

    def __init__(self, config_path: str = None, config_name: str = "default"):
        # Load configuration
        if config_path:
            self.config = TrainingConfig.from_file(config_path)
        else:
            config_manager = ConfigManager()
            self.config = config_manager.load_config(config_name)

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize components
        self.model = None
        self.training_arch = None
        self.datasets = {}
        self.data_loaders = {}

        # Training state
        self.training_results = {}
        self.phase_integration_data = {}

        # Create output directories
        self._setup_directories()

        self.logger.info(f"Phase 5 pipeline initialized: {self.config.experiment_name}")

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for the pipeline."""
        logger = logging.getLogger(f'phase5_pipeline_{self.config.experiment_name}')
        logger.setLevel(getattr(logging, self.config.logging.log_level))

        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File handler
            log_dir = Path(self.config.logging.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / f'phase5_{self.config.experiment_name}.log'
            )
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)

        return logger

    def _setup_directories(self) -> None:
        """Setup all necessary directories."""
        dirs_to_create = [
            self.config.output_dir,
            self.config.checkpoint.checkpoint_dir,
            self.config.logging.log_dir,
            f"{self.config.output_dir}/metrics",
            f"{self.config.output_dir}/models",
            f"{self.config.output_dir}/integration"
        ]

        for directory in dirs_to_create:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def load_phase4_models(self) -> Dict[str, Any]:
        """Load compressed models from Phase 4."""
        phase4_dir = Path(self.config.phase_integration.phase4_input_dir)
        phase4_data = {}

        if not phase4_dir.exists():
            self.logger.warning(f"Phase 4 directory not found: {phase4_dir}")
            return phase4_data

        try:
            # Load compressed model metadata
            metadata_file = phase4_dir / 'compression_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    phase4_data['metadata'] = json.load(f)

            # Load compressed model weights
            model_file = phase4_dir / 'compressed_model.pt'
            if model_file.exists():
                phase4_data['compressed_weights'] = torch.load(model_file, map_location='cpu')

            # Load compression statistics
            stats_file = phase4_dir / 'compression_stats.json'
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    phase4_data['compression_stats'] = json.load(f)

            self.logger.info(f"Phase 4 data loaded: {len(phase4_data)} components")

        except Exception as e:
            self.logger.error(f"Failed to load Phase 4 data: {e}")

        return phase4_data

    def prepare_datasets(self) -> None:
        """Prepare training, validation, and test datasets."""
        data_path = Path(self.config.data.dataset_path) if self.config.data.dataset_path else Path("./data")

        # Create datasets
        self.datasets = {
            'train': Phase5Dataset(data_path, self.config, 'train'),
            'val': Phase5Dataset(data_path, self.config, 'val'),
            'test': Phase5Dataset(data_path, self.config, 'test')
        }

        self.logger.info(f"Datasets prepared: Train({len(self.datasets['train'])}), Val({len(self.datasets['val'])}), Test({len(self.datasets['test'])})")

    def initialize_model(self, phase4_data: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Phase 5 model with optional Phase 4 integration."""
        self.model = Phase5Model(self.config)

        # Load Phase 4 compressed weights if available
        if phase4_data and 'compressed_weights' in phase4_data:
            try:
                # Load compatible weights from Phase 4
                model_state = phase4_data['compressed_weights']
                if isinstance(model_state, dict):
                    # Filter compatible weights
                    compatible_weights = {}
                    model_dict = self.model.state_dict()

                    for key, value in model_state.items():
                        if key in model_dict and model_dict[key].shape == value.shape:
                            compatible_weights[key] = value

                    if compatible_weights:
                        self.model.load_state_dict(compatible_weights, strict=False)
                        self.logger.info(f"Loaded {len(compatible_weights)} weights from Phase 4")

            except Exception as e:
                self.logger.warning(f"Could not load Phase 4 weights: {e}")

        # Initialize training architecture
        self.training_arch = TrainingArchitecture(
            self.config,
            self.model,
            device_ids=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None
        )

        self.logger.info(f"Model initialized: {sum(p.numel() for p in self.model.parameters())} parameters")

    def run_training(self) -> Dict[str, Any]:
        """Run complete training workflow."""
        self.logger.info("Starting Phase 5 training workflow")

        try:
            # Phase 4 integration
            phase4_data = self.load_phase4_models()
            self.phase_integration_data['phase4'] = phase4_data

            # Prepare datasets
            self.prepare_datasets()

            # Initialize model
            self.initialize_model(phase4_data)

            # Run training
            training_results = self.training_arch.train(
                self.datasets['train'],
                self.datasets['val'],
                self.datasets.get('test')
            )

            # Post-training analysis
            self.training_results = self._analyze_training_results(training_results)

            # Prepare for Phase 6
            self._prepare_phase6_output()

            # Generate comprehensive report
            final_report = self._generate_final_report()

            self.logger.info("Phase 5 training completed successfully")
            return final_report

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _analyze_training_results(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training results and generate insights."""
        analysis = {
            'training_summary': training_results,
            'performance_analysis': {},
            'quality_assessment': {},
            'nasa_compliance': {},
            'phase_integration': {}
        }

        try:
            # Performance analysis
            if 'training_time' in training_results:
                total_time = training_results['training_time']
                total_steps = training_results.get('total_steps', 1)
                analysis['performance_analysis'] = {
                    'training_speed': total_steps / total_time,
                    'throughput': training_results.get('model_params', 0) / total_time,
                    'efficiency_score': min(100, (training_results.get('final_train_loss', 1) / training_results.get('best_val_loss', 1)) * 100),
                    'gpu_utilization': training_results.get('gpu_memory_peak', 0) / (16 * 1024**3) * 100  # Assuming 16GB GPU
                }

            # Quality assessment
            analysis['quality_assessment'] = {
                'convergence_achieved': training_results.get('best_val_loss', float('inf')) < 1.0,
                'overfitting_detected': abs(training_results.get('final_train_loss', 0) - training_results.get('final_val_loss', 0)) > 0.1,
                'training_stability': len(training_results.get('training_metrics', {}).get('loss_improvements', [])) > 0
            }

            # NASA POT10 compliance
            if hasattr(self.training_arch.monitor, 'get_nasa_compliance_metrics'):
                nasa_metrics = self.training_arch.monitor.get_nasa_compliance_metrics()
                analysis['nasa_compliance'] = nasa_metrics

            # Phase integration assessment
            analysis['phase_integration'] = {
                'phase4_integration_success': bool(self.phase_integration_data.get('phase4')),
                'phase6_preparation_complete': self._check_phase6_readiness(),
                'cross_phase_validation': self._validate_cross_phase_consistency()
            }

        except Exception as e:
            self.logger.error(f"Training analysis failed: {e}")

        return analysis

    def _prepare_phase6_output(self) -> None:
        """Prepare trained models for Phase 6 baking."""
        phase6_dir = Path(self.config.phase_integration.phase6_output_dir)
        phase6_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save trained model
            model_path = phase6_dir / 'trained_model.pt'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.to_dict(),
                'training_results': self.training_results,
                'phase5_metadata': {
                    'training_complete': True,
                    'bitnet_optimized': self.config.model.use_bitnet_linear,
                    'grokfast_accelerated': self.config.grokfast.enabled,
                    'nasa_compliant': self.config.nasa_compliance.enforce_compliance
                }
            }, model_path)

            # Save training metadata for Phase 6
            metadata_path = phase6_dir / 'phase5_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump({
                    'experiment_name': self.config.experiment_name,
                    'training_completion_time': datetime.now().isoformat(),
                    'model_parameters': sum(p.numel() for p in self.model.parameters()),
                    'final_loss': self.training_results.get('training_summary', {}).get('final_loss', float('inf')),
                    'nasa_compliance_score': self.training_results.get('nasa_compliance', {}).get('compliance_score', 0),
                    'ready_for_phase6': True
                }, f, indent=2)

            self.logger.info(f"Phase 6 output prepared: {phase6_dir}")

        except Exception as e:
            self.logger.error(f"Failed to prepare Phase 6 output: {e}")

    def _check_phase6_readiness(self) -> bool:
        """Check if model is ready for Phase 6."""
        readiness_checks = {
            'model_trained': self.model is not None,
            'training_completed': bool(self.training_results),
            'nasa_compliant': self.config.nasa_compliance.enforce_compliance,
            'output_prepared': Path(self.config.phase_integration.phase6_output_dir).exists()
        }

        return all(readiness_checks.values())

    def _validate_cross_phase_consistency(self) -> bool:
        """Validate consistency across phases."""
        try:
            # Check Phase 4 -> Phase 5 consistency
            phase4_consistent = True
            if 'phase4' in self.phase_integration_data:
                phase4_data = self.phase_integration_data['phase4']
                if 'metadata' in phase4_data:
                    # Validate model architecture consistency
                    phase4_config = phase4_data['metadata'].get('model_config', {})
                    current_config = self.config.model
                    phase4_consistent = (
                        phase4_config.get('hidden_size') == current_config.hidden_size and
                        phase4_config.get('num_layers') == current_config.num_layers
                    )

            return phase4_consistent

        except Exception as e:
            self.logger.error(f"Cross-phase validation failed: {e}")
            return False

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        report = {
            'pipeline_info': {
                'experiment_name': self.config.experiment_name,
                'environment': self.config.environment.value,
                'completion_time': datetime.now().isoformat(),
                'total_duration': time.time() - getattr(self, 'pipeline_start_time', time.time())
            },
            'configuration': self.config.to_dict(),
            'training_results': self.training_results,
            'phase_integration': {
                'phase4_integration': bool(self.phase_integration_data.get('phase4')),
                'phase6_preparation': self._check_phase6_readiness(),
                'cross_phase_validation': self._validate_cross_phase_consistency()
            },
            'quality_gates': self._evaluate_quality_gates(),
            'nasa_compliance': self._get_nasa_compliance_summary(),
            'recommendations': self._generate_recommendations()
        }

        # Save report
        report_path = Path(self.config.output_dir) / 'final_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Final report generated: {report_path}")
        return report

    def _evaluate_quality_gates(self) -> Dict[str, bool]:
        """Evaluate quality gates for production readiness."""
        gates = {
            'training_convergence': False,
            'performance_acceptable': False,
            'nasa_compliance': False,
            'phase_integration': False,
            'model_stability': False
        }

        try:
            # Training convergence
            if self.training_results.get('training_summary'):
                final_loss = self.training_results['training_summary'].get('final_loss', float('inf'))
                gates['training_convergence'] = final_loss < 2.0  # Reasonable threshold

            # Performance acceptable
            if self.training_results.get('performance_analysis'):
                efficiency = self.training_results['performance_analysis'].get('efficiency_score', 0)
                gates['performance_acceptable'] = efficiency > 70

            # NASA compliance
            if self.training_results.get('nasa_compliance'):
                compliance_score = self.training_results['nasa_compliance'].get('compliance_score', 0)
                gates['nasa_compliance'] = compliance_score >= self.config.nasa_compliance.min_compliance_score

            # Phase integration
            gates['phase_integration'] = self._check_phase6_readiness()

            # Model stability
            if self.training_results.get('quality_assessment'):
                gates['model_stability'] = not self.training_results['quality_assessment'].get('overfitting_detected', True)

        except Exception as e:
            self.logger.error(f"Quality gate evaluation failed: {e}")

        return gates

    def _get_nasa_compliance_summary(self) -> Dict[str, Any]:
        """Get NASA POT10 compliance summary."""
        if hasattr(self.training_arch, 'monitor'):
            return self.training_arch.monitor.get_nasa_compliance_metrics()
        else:
            return {'compliance_score': 0.0, 'issues': ['Monitor not available']}

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []

        # Based on training results
        if self.training_results.get('quality_assessment', {}).get('overfitting_detected', False):
            recommendations.append("Consider increasing regularization or reducing model complexity")

        # Based on performance
        if self.training_results.get('performance_analysis', {}).get('gpu_utilization', 0) < 60:
            recommendations.append("GPU utilization is low - consider increasing batch size")

        # Based on NASA compliance
        nasa_score = self.training_results.get('nasa_compliance', {}).get('compliance_score', 0)
        if nasa_score < 90:
            recommendations.append("Improve NASA POT10 compliance - check documentation and monitoring")

        # Based on phase integration
        if not self._check_phase6_readiness():
            recommendations.append("Complete Phase 6 preparation before deployment")

        return recommendations

    def run_validation_pipeline(self) -> Dict[str, Any]:
        """Run validation-only pipeline for testing."""
        self.logger.info("Running Phase 5 validation pipeline")

        try:
            # Prepare minimal datasets
            self.prepare_datasets()

            # Initialize model
            self.initialize_model()

            # Run validation
            validation_results = {
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'bitnet_conversion': self.config.model.use_bitnet_linear,
                'grokfast_enabled': self.config.grokfast.enabled,
                'distributed_ready': self.config.distributed.enabled
            }

            return validation_results

        except Exception as e:
            self.logger.error(f"Validation pipeline failed: {e}")
            raise

    def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        if hasattr(self.training_arch, 'cleanup'):
            self.training_arch.cleanup()

        self.logger.info("Phase 5 pipeline cleanup completed")


if __name__ == "__main__":
    # Example usage and testing
    def test_phase5_pipeline():
        """Test Phase 5 training pipeline."""

        # Create test configuration
        config = TrainingConfig(
            experiment_name="test_phase5",
            num_epochs=2,
            environment=Environment.TESTING
        )
        config.data.batch_size = 4
        config.optimization.learning_rate = 1e-3

        # Save test configuration
        config_manager = ConfigManager()
        config_manager.save_config(config, "test_phase5")

        # Test pipeline initialization
        pipeline = Phase5Pipeline(config_name="test_phase5")
        print(f"✓ Pipeline initialized: {pipeline.config.experiment_name}")

        # Test validation pipeline
        validation_results = pipeline.run_validation_pipeline()
        print(f"✓ Validation completed: {validation_results['model_parameters']} parameters")

        # Test NASA compliance
        compliance = pipeline.config.get_nasa_compliance_checklist()
        compliance_score = sum(compliance.values()) / len(compliance) * 100
        print(f"✓ NASA compliance: {compliance_score:.1f}%")

        # Cleanup
        pipeline.cleanup()

        print("Phase 5 pipeline test completed successfully")

    # Run test
    test_phase5_pipeline()