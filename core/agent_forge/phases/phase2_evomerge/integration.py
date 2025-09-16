"""
Integration layer for EvoMerge phase.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class EvoMergeIntegration:
    """Integration layer for EvoMerge with other phases."""

    def __init__(self):
        self.phase_name = "evomerge"
        self.input_contract = {
            'models': List[nn.Module],
            'count': 3,
            'parameters': 25_000_000,
            'tolerance': 0.01
        }
        self.output_contract = {
            'model': nn.Module,
            'metrics': Dict[str, float],
            'required_metrics': ['fitness', 'perplexity', 'accuracy']
        }

    def validate_input_from_cognate(self, cognate_output: Dict[str, Any]) -> bool:
        """Validate input from Phase 1 (Cognate)."""
        try:
            # Check for models
            if 'models' not in cognate_output:
                logger.error("No models found in Cognate output")
                return False

            models = cognate_output['models']

            # Check count
            if len(models) != self.input_contract['count']:
                logger.error(f"Expected {self.input_contract['count']} models, got {len(models)}")
                return False

            # Check parameters
            for i, model in enumerate(models):
                param_count = sum(p.numel() for p in model.parameters())
                expected = self.input_contract['parameters']
                tolerance = self.input_contract['tolerance']

                if abs(param_count - expected) / expected > tolerance:
                    logger.error(
                        f"Model {i} has {param_count} parameters, "
                        f"expected ~{expected} (tolerance: {tolerance})"
                    )
                    return False

            # Check specializations
            if 'specializations' in cognate_output:
                specs = cognate_output['specializations']
                if set(specs) != {'reasoning', 'memory_integration', 'adaptive_computation'}:
                    logger.warning(f"Unexpected specializations: {specs}")

            return True

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False

    def prepare_output_for_quietstar(self, evomerge_result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare output for Phase 3 (Quiet-STaR)."""
        try:
            # Extract model
            model = evomerge_result.get('model')
            if model is None:
                raise ValueError("No model in EvoMerge result")

            # Prepare output
            output = {
                'model': model,
                'phase_2_metrics': evomerge_result.get('metrics', {}),
                'evolution_history': {
                    'generations': evomerge_result.get('generation', 0),
                    'fitness': evomerge_result.get('fitness', 0.0),
                    'technique': evomerge_result.get('technique', 'evolutionary')
                },
                'ready_for_quietstar': True
            }

            # Add model statistics
            param_count = sum(p.numel() for p in model.parameters())
            output['model_stats'] = {
                'parameters': param_count,
                'layers': len(list(model.modules())),
                'device': next(model.parameters()).device.type
            }

            return output

        except Exception as e:
            logger.error(f"Output preparation failed: {e}")
            return {'error': str(e), 'ready_for_quietstar': False}

    def create_fallback_model(self, cognate_models: List[nn.Module]) -> nn.Module:
        """Create fallback model if evolution fails."""
        logger.warning("Creating fallback model using simple average merge")

        # Simple average merge as fallback
        merged = cognate_models[0].__class__()

        with torch.no_grad():
            for name, param in merged.named_parameters():
                # Average parameters
                avg_param = torch.zeros_like(param)
                for model in cognate_models:
                    avg_param += model.state_dict()[name]
                param.data = avg_param / len(cognate_models)

        return merged

    def save_phase_output(self, result: Dict[str, Any], output_dir: str):
        """Save phase output for later use."""
        output_path = Path(output_dir) / "phase2_evomerge"
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        if 'model' in result:
            model_path = output_path / "evolved_model.pt"
            torch.save(result['model'].state_dict(), model_path)
            logger.info(f"Saved evolved model to {model_path}")

        # Save metrics
        if 'metrics' in result:
            metrics_path = output_path / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(result['metrics'], f, indent=2)
            logger.info(f"Saved metrics to {metrics_path}")

        # Save configuration
        config_path = output_path / "config.json"
        config = {
            'phase': 'evomerge',
            'technique': result.get('technique', 'evolutionary'),
            'generation': result.get('generation', 0),
            'fitness': result.get('fitness', 0.0)
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_phase_output(self, output_dir: str, model_class) -> Dict[str, Any]:
        """Load saved phase output."""
        output_path = Path(output_dir) / "phase2_evomerge"

        result = {}

        # Load model
        model_path = output_path / "evolved_model.pt"
        if model_path.exists():
            model = model_class()
            model.load_state_dict(torch.load(model_path))
            result['model'] = model
            logger.info(f"Loaded model from {model_path}")

        # Load metrics
        metrics_path = output_path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                result['metrics'] = json.load(f)
            logger.info(f"Loaded metrics from {metrics_path}")

        # Load configuration
        config_path = output_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                result.update(config)

        return result