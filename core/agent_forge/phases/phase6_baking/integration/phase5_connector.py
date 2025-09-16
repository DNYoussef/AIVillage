"""
Phase 5 Training Model Connector for Phase 6 Integration

This module provides seamless integration between Phase 5 trained models
and Phase 6 model baking processes, ensuring training metadata preservation
and performance metrics transfer.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import tensorflow as tf

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetadata:
    """Training metadata from Phase 5"""
    model_architecture: str
    training_epochs: int
    final_accuracy: float
    validation_loss: float
    optimizer_config: Dict[str, Any]
    training_time: float
    dataset_size: int
    hyperparameters: Dict[str, Any]
    checkpoint_path: str

@dataclass
class ModelTransferResult:
    """Result of model transfer from Phase 5"""
    success: bool
    model_path: str
    metadata: TrainingMetadata
    performance_metrics: Dict[str, float]
    compatibility_score: float
    validation_results: Dict[str, Any]

class Phase5Connector:
    """Connector for integrating Phase 5 trained models into Phase 6 baking"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.phase5_model_dir = Path(config.get('phase5_model_dir', 'models/phase5'))
        self.metadata_cache = {}
        self.performance_cache = {}

    def discover_trained_models(self) -> List[Dict[str, Any]]:
        """Discover all available trained models from Phase 5"""
        models = []

        if not self.phase5_model_dir.exists():
            logger.warning(f"Phase 5 model directory not found: {self.phase5_model_dir}")
            return models

        for model_dir in self.phase5_model_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / 'training_metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                        models.append({
                            'model_id': model_dir.name,
                            'path': str(model_dir),
                            'metadata': metadata,
                            'last_modified': model_dir.stat().st_mtime
                        })
                    except Exception as e:
                        logger.error(f"Error loading metadata for {model_dir}: {e}")

        return sorted(models, key=lambda x: x['last_modified'], reverse=True)

    def load_training_metadata(self, model_path: str) -> TrainingMetadata:
        """Load training metadata from Phase 5 model"""
        if model_path in self.metadata_cache:
            return self.metadata_cache[model_path]

        metadata_file = Path(model_path) / 'training_metadata.json'

        if not metadata_file.exists():
            raise FileNotFoundError(f"Training metadata not found: {metadata_file}")

        with open(metadata_file, 'r') as f:
            data = json.load(f)

        metadata = TrainingMetadata(
            model_architecture=data.get('architecture', 'unknown'),
            training_epochs=data.get('epochs', 0),
            final_accuracy=data.get('final_accuracy', 0.0),
            validation_loss=data.get('validation_loss', float('inf')),
            optimizer_config=data.get('optimizer_config', {}),
            training_time=data.get('training_time', 0.0),
            dataset_size=data.get('dataset_size', 0),
            hyperparameters=data.get('hyperparameters', {}),
            checkpoint_path=data.get('checkpoint_path', '')
        )

        self.metadata_cache[model_path] = metadata
        return metadata

    def validate_model_compatibility(self, model_path: str) -> Tuple[bool, float, Dict[str, Any]]:
        """Validate model compatibility with Phase 6 baking requirements"""
        try:
            metadata = self.load_training_metadata(model_path)
            compatibility_issues = []
            compatibility_score = 1.0

            # Check architecture compatibility
            supported_architectures = self.config.get('supported_architectures', [])
            if supported_architectures and metadata.model_architecture not in supported_architectures:
                compatibility_issues.append(f"Unsupported architecture: {metadata.model_architecture}")
                compatibility_score -= 0.3

            # Check performance thresholds
            min_accuracy = self.config.get('min_accuracy', 0.8)
            if metadata.final_accuracy < min_accuracy:
                compatibility_issues.append(f"Accuracy below threshold: {metadata.final_accuracy} < {min_accuracy}")
                compatibility_score -= 0.2

            # Check model file existence
            model_file = Path(model_path) / 'model.pth'
            if not model_file.exists():
                model_file = Path(model_path) / 'model.h5'
                if not model_file.exists():
                    compatibility_issues.append("Model file not found")
                    compatibility_score -= 0.4

            # Check dataset size requirements
            min_dataset_size = self.config.get('min_dataset_size', 1000)
            if metadata.dataset_size < min_dataset_size:
                compatibility_issues.append(f"Dataset too small: {metadata.dataset_size} < {min_dataset_size}")
                compatibility_score -= 0.1

            validation_results = {
                'compatible': len(compatibility_issues) == 0,
                'issues': compatibility_issues,
                'score': max(0.0, compatibility_score),
                'metadata': metadata.__dict__
            }

            return len(compatibility_issues) == 0, max(0.0, compatibility_score), validation_results

        except Exception as e:
            logger.error(f"Error validating model compatibility: {e}")
            return False, 0.0, {'error': str(e)}

    def transfer_model(self, model_path: str, target_path: str) -> ModelTransferResult:
        """Transfer model from Phase 5 to Phase 6 with metadata preservation"""
        try:
            # Load and validate metadata
            metadata = self.load_training_metadata(model_path)
            compatible, compatibility_score, validation_results = self.validate_model_compatibility(model_path)

            if not compatible:
                return ModelTransferResult(
                    success=False,
                    model_path='',
                    metadata=metadata,
                    performance_metrics={},
                    compatibility_score=compatibility_score,
                    validation_results=validation_results
                )

            # Create target directory
            target_dir = Path(target_path)
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy model files
            source_dir = Path(model_path)
            for file_pattern in ['*.pth', '*.h5', '*.pkl', '*.json']:
                for file_path in source_dir.glob(file_pattern):
                    target_file = target_dir / file_path.name
                    with open(file_path, 'rb') as src, open(target_file, 'wb') as dst:
                        dst.write(src.read())

            # Extract performance metrics
            performance_metrics = self._extract_performance_metrics(model_path, metadata)

            # Create Phase 6 integration metadata
            integration_metadata = {
                'phase5_metadata': metadata.__dict__,
                'transfer_timestamp': np.datetime64('now').item(),
                'compatibility_score': compatibility_score,
                'performance_metrics': performance_metrics,
                'phase6_ready': True
            }

            with open(target_dir / 'phase6_integration.json', 'w') as f:
                json.dump(integration_metadata, f, indent=2, default=str)

            logger.info(f"Successfully transferred model from {model_path} to {target_path}")

            return ModelTransferResult(
                success=True,
                model_path=str(target_dir),
                metadata=metadata,
                performance_metrics=performance_metrics,
                compatibility_score=compatibility_score,
                validation_results=validation_results
            )

        except Exception as e:
            logger.error(f"Error transferring model: {e}")
            return ModelTransferResult(
                success=False,
                model_path='',
                metadata=TrainingMetadata('', 0, 0.0, float('inf'), {}, 0.0, 0, {}, ''),
                performance_metrics={},
                compatibility_score=0.0,
                validation_results={'error': str(e)}
            )

    def _extract_performance_metrics(self, model_path: str, metadata: TrainingMetadata) -> Dict[str, float]:
        """Extract comprehensive performance metrics from Phase 5 model"""
        metrics = {
            'accuracy': metadata.final_accuracy,
            'loss': metadata.validation_loss,
            'training_time': metadata.training_time,
            'epochs': float(metadata.training_epochs),
            'dataset_size': float(metadata.dataset_size)
        }

        # Load additional metrics if available
        metrics_file = Path(model_path) / 'performance_metrics.json'
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    additional_metrics = json.load(f)
                metrics.update(additional_metrics)
            except Exception as e:
                logger.warning(f"Could not load additional metrics: {e}")

        return metrics

    def get_best_model(self, criteria: str = 'accuracy') -> Optional[Dict[str, Any]]:
        """Get the best model based on specified criteria"""
        models = self.discover_trained_models()

        if not models:
            return None

        if criteria == 'accuracy':
            return max(models, key=lambda x: x['metadata'].get('final_accuracy', 0))
        elif criteria == 'loss':
            return min(models, key=lambda x: x['metadata'].get('validation_loss', float('inf')))
        elif criteria == 'recent':
            return models[0]  # Already sorted by last_modified
        else:
            logger.warning(f"Unknown criteria: {criteria}, using accuracy")
            return max(models, key=lambda x: x['metadata'].get('final_accuracy', 0))

    def validate_integration_pipeline(self) -> Dict[str, Any]:
        """Validate the complete Phase 5 to Phase 6 integration pipeline"""
        validation_results = {
            'phase5_models_found': 0,
            'compatible_models': 0,
            'transfer_success_rate': 0.0,
            'average_compatibility_score': 0.0,
            'issues': [],
            'recommendations': []
        }

        try:
            models = self.discover_trained_models()
            validation_results['phase5_models_found'] = len(models)

            if not models:
                validation_results['issues'].append("No Phase 5 models found")
                validation_results['recommendations'].append("Run Phase 5 training first")
                return validation_results

            compatibility_scores = []
            successful_transfers = 0

            for model in models[:5]:  # Test up to 5 models
                try:
                    compatible, score, _ = self.validate_model_compatibility(model['path'])
                    compatibility_scores.append(score)

                    if compatible:
                        validation_results['compatible_models'] += 1
                        successful_transfers += 1

                except Exception as e:
                    validation_results['issues'].append(f"Error validating {model['model_id']}: {e}")

            if compatibility_scores:
                validation_results['average_compatibility_score'] = np.mean(compatibility_scores)
                validation_results['transfer_success_rate'] = successful_transfers / len(models)

            # Generate recommendations
            if validation_results['compatible_models'] == 0:
                validation_results['recommendations'].append("No compatible models found - check training quality")
            elif validation_results['average_compatibility_score'] < 0.8:
                validation_results['recommendations'].append("Low compatibility scores - review training parameters")
            else:
                validation_results['recommendations'].append("Integration pipeline ready for Phase 6")

        except Exception as e:
            validation_results['issues'].append(f"Pipeline validation error: {e}")

        return validation_results

def create_phase5_connector(config: Dict[str, Any]) -> Phase5Connector:
    """Factory function to create Phase 5 connector"""
    return Phase5Connector(config)

# Integration testing utilities
def test_phase5_integration():
    """Test Phase 5 integration functionality"""
    config = {
        'phase5_model_dir': 'models/phase5',
        'supported_architectures': ['ResNet', 'VGG', 'MobileNet'],
        'min_accuracy': 0.85,
        'min_dataset_size': 1000
    }

    connector = Phase5Connector(config)

    # Test model discovery
    models = connector.discover_trained_models()
    print(f"Found {len(models)} Phase 5 models")

    # Test pipeline validation
    validation_results = connector.validate_integration_pipeline()
    print(f"Pipeline validation: {validation_results}")

    return validation_results

if __name__ == "__main__":
    test_phase5_integration()